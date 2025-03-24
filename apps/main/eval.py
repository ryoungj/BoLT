# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from collections import defaultdict
import json
import logging
from functools import partial
from pathlib import Path
import time
import copy
import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.vllm_causallms import VLLM
from typing import Any, List, Optional, Tuple, Union
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from omegaconf import OmegaConf
import torch
import contextlib
from apps.main.generate import (
    TransformerGenerator,
    PackedCausalTransformerGenerator,
    TransformerGeneratorArgs,
    load_transformer_generator,
    VLLMGenerator
)
from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.latent_bootstrap import bootstrap_latents, BootstrapLatentsArgs
from lingua.args import dump_config, dataclass_from_dict, load_config_file
from lingua.data import init_choice_state, setup_sources, VAL_DATA_FILE_PATTERN
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints, LINGUA_CONFIG_NAME
from lingua.metrics import MetricLogger, LoggingArgs
from lingua.tokenizer import Tokenizer, SpecialTokens
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
    dist_mean_dict,
)

from apps.main.gen_utils.chunk_gen import chunked_generation_wrapper

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


LATENT_TASK_VARIANT_PREFIX = "latent_"

@dataclass
class LMHarnessArgs:
    tasks: Optional[List[Any]] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[Union[int, float]] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234
    include_path: Optional[str] = None  # customized task config path to include
    gen_eval_on_latent_variant: bool = False  # whether to eval on the latent variant of generative tasks for latent models


@dataclass
class ComputeELBOArgs:
    # generation
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 1024

    # monte carlod
    num_mc_samples: List[int] = field(default_factory=lambda: [1])  # number of samples to compute the ELBO

    # chunking
    split_mode: str = "sentence"  # split by sentence by default
    units_per_chunk: Optional[int] = 8  # number of units (e.g., words) per chunk
    num_chunks: Optional[int] = None  # number of chunks to split the input into
    max_num_units_per_chunk: Optional[int] = None  # maximum number of units (e.g., words) per chunk
    min_num_units_per_chunk: Optional[int] = None  # minimum number of units (e.g., words) per chunk
    include_all_prefix_context: bool = True  # include all prefix context by default
    apply_random_chunking: bool = False  # no random chunking by default


@dataclass
class ValidationArgs:
    run_validation: bool = False  # Set default to False for backward compatibility
    max_num_samples: Optional[int] = None  # If None the whole validation file is used
    use_val_from_train_src: bool = True  # Use the validation set from training sources
    root_dir: str = ""
    sources: List[str] = field(default_factory=list) # Other sources to eval on
    downstream_root_dir: str = ""
    downstream_sources: List[str] = field(default_factory=list) # Dwonstream validation sources to eval on
    seed: int = 42
    max_seq_len: Optional[int] = None  # if None, use the model max seq len
    chunk_seq_len: bool = False  # whether to chunk long inputs into multiple chunks instead of truncating them
    compute_elbo_every_n_steps: Optional[int] = None  # the frequency of computing the ELBO, set to None to disable
    elbo_cfg: ComputeELBOArgs = field(default_factory=ComputeELBOArgs)

@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    use_vllm: bool = False
    generator: TransformerGeneratorArgs = field(
        default_factory=TransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)

    logging: LoggingArgs = field(default_factory=LoggingArgs)

    global_step: Optional[int] = None  # for in-training evaluation


def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])

        sampling_params = []
        unique_sampling_params = []
        for g in gen_args:
            params = copy.deepcopy(self.generator.default_sampling_params)
            params.temperature = g.get("temperature", 0. if not g.get("do_sample", True) else self.generator.default_sampling_params.temperature)
            params.top_p = g.get("top_p", self.generator.default_sampling_params.top_p)
            params.top_k = g.get("top_k", self.generator.default_sampling_params.top_k)
            params.stop = g.get("until", []) + (self.generator.default_sampling_params.stop or [])
            params.bad_words = g.get("bad_words", [])
            sampling_params.append(params)

            if params not in unique_sampling_params:
                unique_sampling_params.append(params)

        unique_sampling_params_str = "\n\t" + "\n\t".join([str(p) for p in unique_sampling_params])
        logger.info(f"Sampling generation with the following sampling params:{unique_sampling_params_str}")

        results = self.generator.generate(prompts, sampling_params=sampling_params)
        filtered_gen = []
        for r, p in zip(results, sampling_params):
            g = r.generation_outputs[0].generation  # assumed only one generation
            for e in p.stop:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]

        sampling_params = copy.deepcopy(self.generator.default_sampling_params)
        sampling_params.max_tokens = 1  # only compute the log likelihood of the continuation
        sampling_params.truncate_prompt_tokens = None  # disable truncation

        gen_results = self.generator.generate(inputs, sampling_params=sampling_params)
        results = []
        for idx, (p, gen) in enumerate(zip(prompts, gen_results)):
            p_len = len(self.generator.encode_fn(p))
            # Example:
            # - Prompt is like "p_1 p_2 p_3", continuation is like "a_1 a_2"
            # - Inputs is like "p_1 p_2 p_3 a_1 a_2"
            # - p_len is len(["bos", "p_1", "p_2", "p_3"]) -> 4 = len(prompt) + 1
            # - both ll and greedy are like [l_{p_1}, l_{p_2}, l_{p_3}, l_{a_1}, l_{a_2}] -> 5 = len(inputs) = len(prompt) + len(continuation)
            # - We need to compute the log likelihood of "a_1 a_2", which should be start from index 3 = len(prompt) = p_len - 1

            p_len -= 1  # the continuation tokens starts at the index of the last prompt token, which is p_len - 1
            results.append((sum(gen.prompt_loglikelihoods[p_len:]), all(gen.prompt_greedys[p_len:])))

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]

        sampling_params = copy.deepcopy(self.generator.default_sampling_params)
        sampling_params.max_tokens = 1  # only compute the log likelihood of the continuation
        sampling_params.truncate_prompt_tokens = None  # disable truncation

        gen_results = self.generator.generate(prompts, sampling_params=sampling_params)
        results = []
        for r in gen_results:
            results.append((sum(r.prompt_loglikelihoods),))
        return results


def compute_elbo(generator, texts, config: ComputeELBOArgs, seed=42):
    kwargs = asdict(config)
    gen_latent_kwargs = {
        "use_vllm": isinstance(generator, VLLMGenerator),
        "num_total_samples": max(kwargs["num_mc_samples"]),
        "compute_joint_likelihood": True,
        "compute_elbo": True,
        "num_elbo_samples": kwargs.pop("num_mc_samples"),
        "load_cache": False,
        "save_cache": False,
    }
    gen_latent_cfg = dataclass_from_dict(BootstrapLatentsArgs, gen_latent_kwargs)

    # set up the sampling params for the latent generation
    orig_sample_params = copy.deepcopy(generator.default_sampling_params)
    latent_sample_params = copy.deepcopy(orig_sample_params)
    latent_sample_params.temperature = kwargs.pop("temperature")
    latent_sample_params.top_p = kwargs.pop("top_p") or 1.0
    latent_sample_params.top_k = kwargs.pop("top_k") or -1
    latent_sample_params.max_tokens = kwargs.pop("max_gen_len")
    latent_sample_params.stop += gen_latent_cfg.generation_until
    latent_sample_params.include_stop_str_in_output = False
    generator.default_sampling_params = latent_sample_params

    logger.info(f"Generating latents with the following sampling params: {latent_sample_params}")

    gen_func = partial(bootstrap_latents, cfg=gen_latent_cfg, generator=generator, seed=seed)
    gen_func = chunked_generation_wrapper(gen_func, **kwargs)

    samples = [{"text": text} for text in texts]
    outputs, _ = gen_func(samples, shard_idx=0)
    elbo_keys = [k for k in outputs[0] if k.startswith("elbo")] 
    elbo_keys.append("num_suffix_token_truncated")
    elbos = [{k: o[k] for k in elbo_keys} for o in outputs]

    # restore the original sampling params
    generator.default_sampling_params = orig_sample_params
    return elbos


def eval_on_val(generator, val_args: ValidationArgs, train_cfg: OmegaConf, global_step: Optional[int] = None):
    val_srcs = {}
    if val_args.use_val_from_train_src:
        for src in train_cfg.data.sources:
            path = os.path.join(train_cfg.data.root_dir, src)
            val_srcs[path] = 1.0
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        val_srcs[path] = 1.0
    
    downstream_srcs = {}
    for src in val_args.downstream_sources:
        path = os.path.join(val_args.downstream_root_dir, src)
        downstream_srcs[path] = 1.0
    
    
    val_multi_state = init_choice_state("", val_srcs, val_args.seed, get_global_rank(), get_world_size(), VAL_DATA_FILE_PATTERN)
    val_path_to_iter = setup_sources(val_multi_state)
    downstream_multi_state = init_choice_state("", downstream_srcs, val_args.seed, get_global_rank(), get_world_size(), "*test.jsonl")
    downstream_path_to_iter = setup_sources(downstream_multi_state)

    srcs = {**val_srcs, **downstream_srcs}
    path_to_iter = {**val_path_to_iter, **downstream_path_to_iter}
    logger.info(f"Evaluating on validation sources: {list(srcs.keys())}")

    all_val_metrics = {}

    sampling_params = copy.deepcopy(generator.default_sampling_params)
    sampling_params.max_tokens = 1  # only compute the log likelihood of the continuation
    sampling_params.truncate_prompt_tokens = None  # disable truncation
    
    max_val_samples = val_args.max_num_samples
    if max_val_samples is not None:
        max_val_samples_per_gpu = max_val_samples // get_world_size()  # each GPU will sample this number of samples
        logger.info(f"Using a subset of {max_val_samples} samples from validation sources, {max_val_samples_per_gpu} samples per GPU.")
    else:
        max_val_samples_per_gpu = None
    
    max_seq_len = val_args.max_seq_len
    if max_seq_len is None:
        max_seq_len = generator.max_tokens - 1 # account for the bos token
    
    max_seq_len -= 2  # slack 
    tokenizer = generator.tokenizer

    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        is_downstream = src in downstream_srcs
        if is_downstream:
            start_latent_token = generator.encode_fn(SpecialTokens.START_OF_LATENT.value)[1]
            # end_latent_token = generator.encode_fn(SpecialTokens.END_OF_LATENT.value)[1]
            # prefix_token = generator.encode_fn(SpecialTokens.PRIOR_PREFIX.value)[1]
            # remove_tokens = [start_latent_token, end_latent_token, prefix_token]

        logger.info(f"Running validation on {src}...")
        start_indices = []
        for step, (content, state) in enumerate(jsonl_iterator):
            if state['current_iter'] > 0 or (max_val_samples_per_gpu is not None and step >= max_val_samples_per_gpu and not is_downstream):  # always use the entire validation set for downstream tasks
                break
            
            content_key = "text" if ("text" in content) else "content"
            content_text = content[content_key]
            
            content_tokens = generator.encode_fn(content_text)
            content_tokens = content_tokens[1:]  # remove the bos token
            
            # truncate or chunk the inputs if needed
            if len(content_tokens) > max_seq_len:
                if val_args.chunk_seq_len:
                    for i in range(0, len(content_tokens), max_seq_len):
                        content_text = tokenizer.decode(content_tokens[i:i+max_seq_len])
                else:
                    content_text = tokenizer.decode(content_tokens[:max_seq_len])
            
            if is_downstream:
                start_idx = content_tokens.index(start_latent_token)
                assert start_idx != -1, "Start of latent token not found"

                if train_cfg.data.latent_type is None:
                    for special_token in SpecialTokens:   # remove all special latent tokens
                        content_text = content_text.replace(special_token.value, "")
            else:
                start_idx = 0
            
            start_indices.append(start_idx)
            texts.append(content_text)

        # compute the likelihoods
        results = generator.generate(texts, sampling_params=sampling_params)
        metrics = defaultdict(list)
        for i, (r, start_idx) in enumerate(zip(results, start_indices)):
            ll = r.prompt_loglikelihoods
            ll = ll[start_idx:]
            tmp = sum(ll)
            metrics['nll'].append(tmp)
            metrics['nll_per_token'].append(tmp / len(ll))
            metrics['nll_per_char'].append(tmp / len(texts[i]))
            metrics['avg_seqlen'].append(len(ll))

        # compute elbo
        eval_elbo = val_args.compute_elbo_every_n_steps is not None and (
            global_step is None  # if global_step is None, we always compute the ELBO
            or global_step % val_args.compute_elbo_every_n_steps == 0   # every N steps
            or global_step == train_cfg.steps  # at the end of training
            or global_step in train_cfg.checkpoint.must_eval_steps  # at the end of training
        )
        if eval_elbo and train_cfg.data.latent_type == "random":
            elbos = compute_elbo(generator, texts, config=val_args.elbo_cfg, seed=val_args.seed)
            for i, elbo in enumerate(elbos):
                num_elbo_tokens = metrics['avg_seqlen'][i] - elbo.pop("num_suffix_token_truncated", 0)
                for k, v in elbo.items():
                    metrics[k].append(v)
                    metrics[f'{k}_per_token'].append(v / num_elbo_tokens)
                    metrics[f'{k}_per_char'].append(v / len(texts[i]))
        
        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])
        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done.\nMetrics: {metrics}")
        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1")
            name = f"{name}_1"
        all_val_metrics[name] = metrics
    return all_val_metrics


class FunctionEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)

def run_eval(cfg: EvalArgs, generator: TransformerGenerator, train_cfg: OmegaConf):
    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)
    

    # For debugging VLLM 
    # wrap = VLLM(pretrained="<MODEL_PATH>", tokenizer="<TOKENIZER_PATH>", dtype="bfloat16", load_format="lingua")

    val_results = None
    if cfg.validation.run_validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg, global_step=cfg.global_step)

    wrap = EvalHarnessLM(generator)
    
    harness_cfg_dict = asdict(cfg.harness)
    eval_on_latent_variant = harness_cfg_dict.pop("gen_eval_on_latent_variant", False) and train_cfg.data.latent_type in ["random", "prior"]
    include_path = harness_cfg_dict.pop("include_path", None)

    if include_path is not None:
        task_manager = TaskManager(verbosity=harness_cfg_dict["verbosity"], include_path=include_path)
        harness_cfg_dict["task_manager"] = task_manager

    if eval_on_latent_variant:
        assert include_path is not None, "include_path must be provided for customized latent variant evaluation"
        for task_config in harness_cfg_dict["tasks"]:
            task_name = task_config["task"]

            latent_task_name = f"{LATENT_TASK_VARIANT_PREFIX}{task_name}"
            logger.info(f"[Task: {task_name}] Evaluating on the latent variant for latent models, new task name: {latent_task_name}")
            task_config["task"] = latent_task_name

    if len(harness_cfg_dict["tasks"]) == 0:
        logger.warning("No tasks to evaluate, skipping evaluation")
        results = {"results": {}}
    else:
        results = simple_evaluate(wrap, **harness_cfg_dict)

    if get_global_rank() == 0:
        with open(Path(cfg.dump_dir) / "results.json", "w") as f:
            f.write(json.dumps(results, cls=FunctionEncoder))   # some items in eval harness are functions, need to convert them to string

        log_results = results["results"]
        log_results = {"eval/" + m.replace(".", "/"): v for m, v in log_results.items()}
        if cfg.global_step is not None:
            log_results["global_step"] = cfg.global_step
        logger.info(f"All evaluation results: {log_results}")

        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
            log_results.update({"val/" + m: v for m, v in val_results.items()})
    else:
        log_results = None

    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step
        print(
            json.dumps(timestamp | log_results),
            file=open(metric_log_path, mode="a"),
            flush=True,
        )

        val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=open(val_log_path, mode="a"),
                flush=True,
            )

        # Write sentinel file to mark eval completion
        eval_complete_path = Path(cfg.ckpt_dir) / "eval.complete"
        eval_complete_path.touch()
    del generator

    return log_results


def launch_eval(cfg: EvalArgs):
    # TODO: need to figure out the proper way of managing GPU memory (switching between train and eval)

    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
       
    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / LINGUA_CONFIG_NAME).exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not (consolidate_path.exists() and (consolidate_path / LINGUA_CONFIG_NAME).exists()):
            if get_global_rank() == 0:
                consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)
            else:
                # wait for the master to create the consolidate folder
                while not Path(consolidate_path / LINGUA_CONFIG_NAME).exists():
                    time.sleep(10)

            torch.distributed.barrier()
                    
    
    config = consolidate_path / LINGUA_CONFIG_NAME
    config = OmegaConf.load(config)
    tokenizer_path = config.data.tokenizer.path

    torch.distributed.barrier()
    logger.info("Loading model")
    if cfg.use_vllm:
        generator = VLLMGenerator(cfg.generator, consolidate_path, tokenizer_path)
    else:
        generator = PackedCausalTransformerGenerator(cfg.generator, consolidate_path, tokenizer_path)

    torch.distributed.barrier()
    logger.info("Model loaded")
    
    log_results = run_eval(cfg, generator, config)

    # Move model to CPU and delete it to free up memory, this is a temporary fix
    generator.clean_up()
    del generator
    return log_results


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
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
    file_cfg = load_config_file(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    with contextlib.ExitStack() as stack:
        metric_logger = stack.enter_context(
            MetricLogger(Path(cfg.dump_dir) / "metrics.jsonl", cfg)
        )
        eval_results = launch_eval(cfg)

        if get_global_rank() == 0:
            metric_logger.log(eval_results)


if __name__ == "__main__":
    main()
