import os
import json
import pathlib
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import log_table_to_comet_experiment
import warnings
from transformers.utils.import_utils import _is_package_available
from configs.eval.task_configs.math_cot.utils import process_results
from unittest.mock import patch
from accelerate.utils.other import is_compiled_module
import contextlib
import functools
from pathlib import Path
import copy


from datasets import load_dataset
from apps.main.gen_utils.misc_utils import jlload, zip_
from lingua.tokenizer import SpecialTokens

# from transformers.trainer_callback import EarlyStoppingCallback


@dataclass
class SFTArgs:
    dump_dir: str
    tokenizer_name_or_path: str
    model_name_or_path: str
    save_dir: str
    run_name: str
    epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_len: int
    num_proc: int
    logging_steps: int
    eval_steps: Union[int, float]  # Also determines save_steps
    num_eval_samples: int
    fp16: bool
    bf16: bool
    tf32: bool
    seed: int
    eval_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    adam_beta1: float
    adam_beta2: float
    eval_delay: int

    # Data
    dataset: str
    dataset_base_path: str
    use_validation_set: bool
    use_test_set: bool

    # If true, the only special token used is <|START_OF_LATENT|> at the end of the prompt.
    # Used for raw data (non-latent) baselines
    only_special_token_delimiter: bool

    # Generation config / evaluation
    max_length: int  # Length of input prompt + max_new_tokens
    max_new_tokens: int
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    num_samples_per_prompt: int

    # VLLM generation
    use_vllm: bool
    vllm_device: str
    vllm_gpu_memory_utilization: float
    vllm_dtype: str
    vllm_enable_prefix_caching: bool
    vllm_max_model_len: int
    output_csv_path: str


"""Token embedding resize from AlpacaFarm."""


def stable_resize_token_embeddings_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    special_tokens_dict: dict,
):
    """Resize tokenizer and embedding together.

    For new tokens, the embedding value is the average of all old embedding vectors.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    stable_resize_token_embeddings(model, len(tokenizer))


def stable_resize_token_embeddings(
    model: PreTrainedModel, target_size: int, jitter_new_embeddings=False
):
    num_new_tokens = target_size - model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(target_size)

    if num_new_tokens > 0:

        @torch.inference_mode()
        def stable_init(embedding):
            embedding_data = embedding.weight.data
            embedding_avg = embedding_data[:-num_new_tokens].mean(dim=0, keepdim=True)
            embedding_data[-num_new_tokens:] = embedding_avg
            if jitter_new_embeddings:
                embedding_std = embedding_data[:-num_new_tokens].std(
                    dim=0, keepdim=True
                )
                # The random tensor must be of the same shape as the new embeddings.
                embedding_data[-num_new_tokens:] += (
                    torch.randn_like(embedding_data[-num_new_tokens:]) * embedding_std
                )

        input_embeddings = (
            model.get_input_embeddings()
        )  # Must grab this again after resize.
        output_embeddings = model.get_output_embeddings()
        # It doesn't matter if there's weight sharing or not; with sharing, the second init will overwrite the first.
        for embeddings in (input_embeddings, output_embeddings):
            stable_init(embeddings)


"""Utilities for in-the-loop vLLM generation."""


def is_vllm_available():
    return _is_package_available("vllm")


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


"""Generation callbacks with vLLM support (tested on single GPU, not multi-GPU yet).
TODO(@nband): support multi-GPU.

Based on TRL callbacks from 
https://github.com/huggingface/trl/blob/main/trl/trainer/callbacks.py
"""


def _generative_accuracy_completions_df(
    state: TrainerState,
    prompts: List[str],
    completions: List[str],
    is_correct: List[bool],
) -> pd.DataFrame:
    global_step = [str(state.global_step)] * len(prompts)
    data = list(zip(global_step, prompts, completions, is_correct))
    return pd.DataFrame(data, columns=["step", "prompt", "completion", "is_correct"])


def _generate_completions_with_eos(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    generation_config: Optional[GenerationConfig],
    batch_size: int = 1,
) -> list[str]:
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (list[str]): A list of input prompts for which completions are to be generated.
        model (PreTrainedModel): The pre-trained model to be used for generation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for encoding and decoding.
        accelerator (Accelerator): The accelerator to be used for model execution.
        generation_config (GenerationConfig): Configuration for text generation.
        batch_size (int, optional): The number of prompts to process in each batch. Default is 1.

    Returns:
        list[str]: A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx : idx + batch_size]
            tokenized_batch = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
                eos_token_id=tokenizer.eos_token_id,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations):
                # Remove prompt from generation
                generation = generation[len(prompt) :]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)

    return completions


class GenerativeAccuracyCallback(TrainerCallback):
    """
    A [`~transformers.TrainerCallback`] that draws samples from the model on a set of prompts and compares the generated samples to the ground truth.

    Usage:
    ```python
    trainer = DPOTrainer(...)
    generative_accuracy_callback = GenerativeAccuracyCallback(trainer=trainer)
    trainer.add_callback(generative_accuracy_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        eval_dataset (`datasets.Dataset` or `Dict[str, datasets.Dataset]`):
            The evaluation dataset to use for generating completions. Must include a `"prompt"` and `"completion"` column.
        answer_grader_fn (callable):
            A function that takes two strings and returns a boolean indicating whether the generated completion is correct.
            Usually involves extracting the answer from the generated completion and comparing it to the ground truth.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use for generating completions.
        num_prompts (`int` or `None`, *optional*, defaults to `None`):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples
            in the evaluation dataset.
        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generation.
        vllm_device (`str`, *optional*, defaults to `None`):
            The device to use for vLLM generation.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `None`):
            The GPU memory utilization to use for vLLM generation.
        vllm_dtype (`str`, *optional*, defaults to `None`):
            The dtype for the model used in vLLM generation.
        vllm_enable_prefix_caching (`bool`, *optional*, defaults to `True`):
            Whether to enable prefix caching for vLLM generation.
        vllm_max_model_len (`int`, *optional*, defaults to `None`):
            The maximum length of the model to use for vLLM generation.
        output_csv_path (`str`, *optional*, defaults to `None`):
            The path to save the CSV file containing the generated completions and their accuracy.
    """

    def __init__(
        self,
        answer_grader_fn: Callable[[str, str], bool],
        eval_dataset: Union[Dataset, Dict[str, Dataset]],
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        num_prompts: Optional[int] = None,
        use_vllm: bool = False,
        vllm_device: Optional[str] = None,
        vllm_gpu_memory_utilization: Optional[float] = None,
        vllm_dtype: Optional[float] = None,
        vllm_enable_prefix_caching: Optional[bool] = True,
        vllm_max_model_len: Optional[int] = None,
        output_csv_path: Optional[str] = None,
        num_samples_per_prompt: int = 1,
    ):
        self.answer_grader_fn = answer_grader_fn
        self.trainer = trainer
        self.generation_config = generation_config
        self.eval_dataset = eval_dataset
        self.use_vllm = use_vllm
        self.output_csv_path = output_csv_path
        self.num_samples_per_prompt = num_samples_per_prompt

        if num_prompts is not None and num_prompts != -1:
            if isinstance(self.eval_dataset, dict):
                for dataset_name, dataset in self.eval_dataset.items():
                    self.eval_dataset[dataset_name] = dataset.select(range(num_prompts))
            else:
                self.eval_dataset = self.eval_dataset.select(range(num_prompts))

        # Implementation based on https://github.com/huggingface/trl/blob/e3244d2d096ff1e2e248c931d06d39e165e20623/trl/trainer/grpo_trainer.py
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            assert (
                vllm_device is not None
            ), "`vllm_device` must be provided if `use_vllm` is True"

            if trainer.accelerator.is_main_process:
                device_type = PartialState().default_device.type
                device_module = getattr(torch, device_type)
                if vllm_device == "auto":
                    if device_module.device_count() == 1:
                        vllm_device = f"{device_type}:0"  # particular case when training with only 1 device: share it
                    else:
                        vllm_device = f"{device_type}:{trainer.accelerator.num_processes}"  # take the next GPU idx

                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == f"{device_type}"
                    and int(vllm_device.split(":")[1]) >= device_module.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {device_module.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"{device_type}:{idx}"
                    for idx in range(trainer.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )

                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )

                # For Ascend NPU (torch-npu), collective communication requires the establishment of a communication
                # group, and different processes must hold the same group number. However, multiple process groups will
                # be created internally within vLLM. This will cause the group id of the communication group on rank 0
                # to be different from that of other ranks, causing backward to hang on because the communication
                # domain cannot be established. So we need to patch it to make sure the group id of different ranks in
                # the training phase are the same.
                @contextlib.contextmanager
                def new_group_context():
                    new_group = torch.distributed.new_group
                    try:
                        torch.distributed.new_group = functools.partial(
                            new_group, use_local_synchronization=True
                        )
                        torch.npu.mem_get_info = functools.partial(
                            torch.npu.mem_get_info, device=vllm_device
                        )
                        yield
                    finally:
                        torch.distributed.new_group = new_group

                new_group_patch = (
                    new_group_context()
                    if device_type == "npu"
                    else contextlib.nullcontext()
                )
                with world_size_patch, profiling_patch, new_group_patch:
                    self.llm = LLM(
                        model=trainer.model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=vllm_gpu_memory_utilization,
                        dtype=vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=vllm_enable_prefix_caching,
                        max_model_len=vllm_max_model_len,
                    )

                # Sampling parameters
                self.sampling_params = SamplingParams(
                    max_tokens=self.generation_config.max_new_tokens,
                    n=1,  # TODO(@nband): support self-consistency, etc.
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    min_p=(
                        0.0
                        if self.generation_config.min_p is None
                        else self.generation_config.min_p
                    ),
                    repetition_penalty=self.generation_config.repetition_penalty,
                )

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.trainer.accelerator.wait_for_everyone()

    def _move_model_to_vllm(self):
        """TODO(@nband): support PEFT models if we want."""
        with unwrap_model_for_generation(
            self.trainer.model_wrapped, self.trainer.accelerator
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod

            state_dict = unwrapped_model.state_dict()
            if self.trainer.accelerator.is_main_process:
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights(state_dict.items())

    def _generate_completions_with_eos_vllm(
        self,
        prompts: List[str],
        ground_truth_completions: List[str],
    ) -> list[str]:
        """
        Generates completions for a list of pre-formatted prompts from the given model.
        """
        trainer = self.trainer
        self._move_model_to_vllm()
        if trainer.accelerator.is_main_process:
            # ordered_set_of_prompts = list(dict.fromkeys(prompts))
            all_outputs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
                use_tqdm=True,
            )

            # Decode to text, including special tokens
            generated_completions = [
                self.trainer.tokenizer.decode(
                    output.outputs[0].token_ids, skip_special_tokens=False
                )
                for output in all_outputs
            ]
            completions = list(zip_(ground_truth_completions, generated_completions))
            is_correct = [
                self.answer_grader_fn(completion=completion, ground_truth=ground_truth)
                for ground_truth, completion in completions
            ]
        else:
            # Postprocessing happens on the main process, so just include dummy values
            completions = [None] * len(prompts)
            is_correct = [None] * len(prompts)

        return is_correct, prompts, completions

    def on_evaluate_helper(
        self,
        model,
        tokenizer,
        accelerator,
        args,
        state,
        eval_dataset,
        eval_dataset_name="eval",
    ):
        eval_prompts = eval_dataset["prompt"].copy()
        eval_completions = eval_dataset["completion"].copy()

        if self.num_samples_per_prompt > 1:
            # Create multiple copies of the dataset using list multiplication
            print(f"Sampling {self.num_samples_per_prompt} times for eval set!")
            # Simply duplicate the dataset num_samples times
            # This is because VLLM does not seem faster (or even slower) for small models when using n > 1
            eval_prompts = list(eval_prompts) * self.num_samples_per_prompt
            eval_completions = list(eval_completions) * self.num_samples_per_prompt

        if self.use_vllm:
            is_correct, prompts, completions = self._generate_completions_with_eos_vllm(
                prompts=eval_prompts,
                ground_truth_completions=eval_completions,
            )
        else:
            with accelerator.split_between_processes(
                list(zip_(eval_prompts, eval_completions))
            ) as prompts_completions:
                prompts, completions = [elem[0] for elem in prompts_completions], [
                    elem[1] for elem in prompts_completions
                ]
                generated_completions = _generate_completions_with_eos(
                    prompts,
                    model=model,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    generation_config=self.generation_config,
                    batch_size=args.per_device_eval_batch_size,
                )
                completions = list(zip_(completions, generated_completions))
                is_correct = [
                    self.answer_grader_fn(
                        completion=completion, ground_truth=ground_truth
                    )
                    for ground_truth, completion in completions
                ]
                is_correct = gather_object(is_correct)
                prompts = gather_object(prompts)
                completions = gather_object(completions)

        # Logging
        if self.trainer.accelerator.is_main_process:
            accuracy = sum(is_correct) / len(is_correct)
            self.trainer.log({f"eval/{eval_dataset_name}_accuracy": accuracy})

            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    df = _generative_accuracy_completions_df(
                        state=state,
                        prompts=prompts,
                        completions=completions,
                        is_correct=is_correct,
                    )
                #     wandb.log(
                #         {
                #             f"eval/{eval_dataset_name}_model_completions": wandb.Table(
                #                 dataframe=df
                #             )
                #         }
                #     )

            # If provided, log to CSV
            if self.output_csv_path is not None:
                step_df_name = Path(self.output_csv_path) / f"{eval_dataset_name}_model_completions_step_{state.global_step}.csv"
                print(f"Logging to CSV at {step_df_name}")
                df.to_csv(step_df_name, index=False)

            if "comet_ml" in args.report_to:
                df = _generative_accuracy_completions_df(
                    state=state,
                    prompts=prompts,
                    completions=completions,
                    is_correct=is_correct,
                )
                log_table_to_comet_experiment(
                    name=f"eval/{eval_dataset_name}_model_completions.csv",
                    table=df,
                )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # At every evaluation step, we generate completions for the model and compare them with the ground truth.
        # Then we log this to the trainer.
        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        if isinstance(self.eval_dataset, dict):
            for dataset_name, dataset in self.eval_dataset.items():
                self.on_evaluate_helper(
                    model=model,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    args=args,
                    state=state,
                    eval_dataset=dataset,
                    eval_dataset_name=dataset_name,
                )
        else:
            self.on_evaluate_helper(
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                args=args,
                state=state,
                eval_dataset=self.eval_dataset,
                eval_dataset_name="eval",
            )


"""Dataset preprocessing functions.

We need to end the prompt in precisely the `response_template_ids` that we pass to the `DataCollatorForCompletionOnlyLM`.
Then the label on which we backpropagate will be exactly the `completion` field below.
This is necessary because the formatted `prompt` field below are passed to the accuracy evaluation callback `GenerativeAccuracyCallback`.
"""

RESPONSE_TEMPLATE_STR = SpecialTokens.START_OF_LATENT.value + SpecialTokens.PRIOR_PREFIX.value
RESPONSE_TEMPLATE_STR_ONLY_SPECIAL_TOKEN_DELIMITER = SpecialTokens.START_OF_LATENT.value


def prepare_latent_lingua_downstream_dataset(
    example, only_special_token_delimiter: bool
):
    prompt_and_completion = example["text"]
    prompt, completion = prompt_and_completion.split(RESPONSE_TEMPLATE_STR)

    if only_special_token_delimiter:
        # Remove the <|END_OF_LATENT|> token from the completion
        assert SpecialTokens.END_OF_LATENT.value in completion
        completion = completion.replace(SpecialTokens.END_OF_LATENT.value, "")

        # Add the <|START_OF_LATENT|> token to the prompt, which is a unique delimiter used to separate the prompt from the completion.
        prompt = prompt + RESPONSE_TEMPLATE_STR_ONLY_SPECIAL_TOKEN_DELIMITER
    else:
        prompt = prompt + RESPONSE_TEMPLATE_STR

    return {
        "prompt": prompt,
        "completion": completion,
    }


def prepare_dataset(dataset, num_proc, only_special_token_delimiter: bool):
    prepare_example_fn = partial(
        prepare_latent_lingua_downstream_dataset,
        only_special_token_delimiter=only_special_token_delimiter,
    )
    dataset = dataset.map(prepare_example_fn, num_proc=num_proc)
    return dataset


HF_REPO_NAME = "ryoungj/bootstrap-latent-thought-data"

def load_dataset_from_hf(dataset_name: str, dataset_path: str, use_validation_set: bool, use_test_set: bool, save_jsonl: bool = True):
    """
    Load a dataset from Hugging Face Hub.
    """
    
    def _load_split(split):
        dataset = load_dataset(HF_REPO_NAME, f"finetune_{dataset_name}", split=split)
        data_dir = os.path.join(dataset_path, dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, f"{split}.jsonl")
        if save_jsonl and not os.path.exists(data_path):
            print(f"Saving {split} dataset to {data_path}")
            dataset.to_json(data_path)
        return dataset
    
    # Create a DatasetDict to store the loaded splits
    dataset_dict = DatasetDict()
    
    # Load each requested split individually
    dataset_dict["train"] = _load_split("train")
    
    if use_validation_set:
        dataset_dict["validation"] = _load_split("validation")
    
    if use_test_set:
        dataset_dict["test"] = _load_split("test")
    
    return dataset_dict


def get_datasets(cfg: SFTArgs):
    train_and_eval_datasets = load_dataset_from_hf(
        cfg.dataset, cfg.dataset_base_path, cfg.use_validation_set, cfg.use_test_set
    )
    train_dataset = prepare_dataset(
        train_and_eval_datasets["train"], cfg.num_proc, cfg.only_special_token_delimiter
    )

    # Drop cols that are not "prompt" or "completion"
    train_dataset = train_dataset.remove_columns(
        set(train_dataset.column_names) - {"prompt", "completion"}
    )

    eval_datasets = {}
    eval_datasets_for_accuracy_callback = {}
    if cfg.use_validation_set:
        validation_dataset = prepare_dataset(
            train_and_eval_datasets["validation"],
            cfg.num_proc,
            cfg.only_special_token_delimiter,
        )
        eval_datasets["validation"] = validation_dataset
        eval_datasets_for_accuracy_callback["validation"] = copy.deepcopy(
            validation_dataset
        )

    if cfg.use_test_set:
        test_dataset = prepare_dataset(
            train_and_eval_datasets["test"],
            cfg.num_proc,
            cfg.only_special_token_delimiter,
        )
        eval_datasets["test"] = test_dataset
        eval_datasets_for_accuracy_callback["test"] = copy.deepcopy(test_dataset)

    return train_dataset, eval_datasets, eval_datasets_for_accuracy_callback


"""Prompt formatting and answer extraction functions."""


def formatting_prompts_func(example, eos_token=None):
    output_texts = []
    for prompt, response in zip(example["prompt"], example["completion"]):
        formatted = prompt + response

        if eos_token is not None:
            formatted += eos_token

        output_texts.append(formatted)

    return output_texts


def grade_exact_match(completion: str, ground_truth: str) -> bool:
    return process_results(doc={"solution": ground_truth}, results=[completion])[
        "exact_match"
    ]


def get_data_collator(
    tokenizer: AutoTokenizer, response_template_str: str = RESPONSE_TEMPLATE_STR
):
    response_template_ids = tokenizer.encode(
        response_template_str, add_special_tokens=False
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tokenizer
    )
    return collator


"""Training loop."""


def train_sft(cfg: SFTArgs):
    set_seed(cfg.seed)

    # Initialize tokenizer and model from local path with accelerate
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": PartialState().process_index},
    )

    # Collect special tokens. Only add if non-existent.
    special_tokens_dict = dict(additional_special_tokens=[])
    if (
        tokenizer.pad_token_id is None
        or tokenizer.pad_token_id == tokenizer.eos_token_id
    ):
        special_tokens_dict["pad_token"] = "[PAD]"

    if cfg.only_special_token_delimiter:
        # Should already have inititalized the special token delimiting the prompt and completion
        assert RESPONSE_TEMPLATE_STR_ONLY_SPECIAL_TOKEN_DELIMITER in tokenizer.vocab

    stable_resize_token_embeddings_and_tokenizer(model, tokenizer, special_tokens_dict)

    if not (pathlib.Path(cfg.model_name_or_path) / "model_written_marker.txt").exists():
        model.save_pretrained(cfg.model_name_or_path)
        tokenizer.save_pretrained(cfg.model_name_or_path)
        with open(pathlib.Path(cfg.model_name_or_path) / "model_written_marker.txt", "w") as f:
            f.write("model_written")
    else:
        print(f"Model already written to {cfg.model_name_or_path}")

    # Get datasets
    train_dataset, eval_datasets, eval_datasets_for_accuracy_callback = get_datasets(
        cfg
    )

    if cfg.only_special_token_delimiter:
        response_template_str = RESPONSE_TEMPLATE_STR_ONLY_SPECIAL_TOKEN_DELIMITER
    else:
        response_template_str = RESPONSE_TEMPLATE_STR

    print(
        f"Using response template string (i.e., the prompt ends on this string): {response_template_str}"
    )
    collator = get_data_collator(tokenizer, response_template_str)

    # Configure training
    training_args = SFTConfig(
        output_dir=cfg.save_dir,
        run_name=cfg.run_name,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        max_seq_length=cfg.max_seq_len,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        seed=cfg.seed,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        eval_delay=cfg.eval_delay,
        overwrite_output_dir=True,  # Overwrite existing checkpoints
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        # Just pass validation here, so the GenerativeAccuracyCallback is called only a single time
        eval_dataset=eval_datasets['validation'],
        formatting_func=partial(formatting_prompts_func, eos_token=tokenizer.eos_token),
        data_collator=collator,
        args=training_args,
    )

    # Add callback for computing accuracy and log-likelihoods
    generation_config = GenerationConfig(
        max_length=cfg.max_length,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Add eval datasets to the GenerativeAccuracyCallback
    generative_accuracy_callback = GenerativeAccuracyCallback(
        answer_grader_fn=grade_exact_match,
        eval_dataset=eval_datasets_for_accuracy_callback,
        trainer=trainer,
        generation_config=generation_config,
        num_prompts=cfg.num_eval_samples,
        use_vllm=cfg.use_vllm,
        vllm_device=cfg.vllm_device,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_dtype=cfg.vllm_dtype,
        vllm_enable_prefix_caching=cfg.vllm_enable_prefix_caching,
        vllm_max_model_len=cfg.vllm_max_model_len,
        output_csv_path=cfg.output_csv_path,
        num_samples_per_prompt=cfg.num_samples_per_prompt,
    )
    trainer.add_callback(generative_accuracy_callback)

    # Add early stopping callback
    # if cfg.early_stopping_patience is not None:
    #     early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
    #     trainer.add_callback(early_stopping_callback)

    # Start training
    trainer.train()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgs

    @dataclass
    class LMTransformerArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate SFTArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(SFTArgs(**file_cfg))
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train_sft(cfg)

    with open(os.path.join(cfg.dump_dir, "eval.completed"), "w") as f:
        f.write("success")



if __name__ == "__main__":
    main()
