# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Any, Union, Sequence
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from functools import partial
import json

from omegaconf import OmegaConf
from torch.nn import functional as F
import xformers

from vllm import LLM, SamplingParams

from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME, consolidate_checkpoints, CONSOLIDATE_FOLDER, CONFIG_NAME, LINGUA_CONFIG_NAME
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.logger import init_logger
from lingua.transformer import (
    Attention,
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)
from torch.nn.attention.flex_attention import create_block_mask
from apps.main.gen_utils.cpt_ckpt_utils import is_llama_pretrained_ckpt

logger = logging.getLogger()



@dataclass
class TransformerGeneratorArgs:
    n: int = 1  # number of samples to generate
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_tokens: Optional[int] = None
    max_prompt_len: Optional[int] = None
    max_gen_len: int = 1024  # Maximum number of tokens to generate
    until: List[str] = field(default_factory=list)
    strip_until: bool = False  # Whether to strip the until tokens from the generation
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    seed: Optional[int] = 42

    ## The below are only for the packed causal transformer generator
    max_pack_tokens: int = 2048  # Maximum number of tokens that can go through the model, for packed causal transformer this is the packed sequence length
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False
    logit_dtype: Optional[str] = "fp32"
    
    ### The below are only for the VLLM generator
    gpu_memory_utilization: float = 0.9
    max_num_seqs: Optional[int] = None

@dataclass
class GenerateArgs:
    model_dir: str = ""
    step: Optional[int] = None
    use_vllm: bool = False
    prompt_path: Optional[str] = None
    generation_prefix: Optional[str] = None  # Text added before the generated completion
    generation_until: Optional[List[str]] = field(default_factory=list)  # Text that indicates the end of generation
    generator: TransformerGeneratorArgs = field(default_factory=TransformerGeneratorArgs) 


@dataclass
class GenerationOutput:
    is_done: bool  # whether the generation is finished
    generation: str  # generated text
    generated_tokens: List[int]  # generated tokens
    generated_loglikelihoods: List[float]  # log likelihood of the generation
    generated_greedys: List[bool]  # whether the generated token is greedy


@dataclass
class GenerationResults:
    prompt_tokens: List[int]  # prompt tokens
    prompt_loglikelihoods: List[float]  # log likelihood of the prompt tokens
    prompt_greedys: List[bool]  # whether the prompt token is greedy
    generation_outputs: List[GenerationOutput]  # generation outputs    


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)  # (N, V)
    argmax_token = torch.argmax(logits, dim=-1)  # (N,)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)  # (N, V)

        if top_p is not None and top_p < 1.0:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None and top_k > 0:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        next_token = next_token.squeeze(-1)  # (N,)
        logprobs = torch.log(torch.gather(probs, -1, next_token.unsqueeze(-1))).squeeze(-1)  # (N,)
        is_greedy = (next_token == argmax_token)
    else:
        next_token = argmax_token
        logprobs = torch.zeros_like(next_token, dtype=logits.dtype)
        is_greedy = torch.ones_like(next_token, dtype=torch.bool)
    return next_token.view(shape[:-1]), logprobs.view(shape[:-1]), is_greedy.view(shape[:-1])


def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = torch.tensor(p, dtype=torch.long)
        l = p.size(0)
        res.append(p)
        lengths.append(l)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths


def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches


class KVCache(nn.Module):
    def __init__(self, bsz, seqlen, n_heads, head_dim, dtype, device):
        super().__init__()
        shape = (bsz, seqlen, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.offset = 0

    def update(self, k_val, v_val, tok_idx):
        # input_pos: [B], k_val: [B, S, H, D]
        self.k_cache.index_copy_(1, self.offset + tok_idx, k_val)
        self.v_cache.index_copy_(1, self.offset + tok_idx, v_val)
        return self.k_cache, self.v_cache


class TransformerGenerator:
    def __init__(self, cfg: TransformerGeneratorArgs, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def generate(self, texts: List[str], sampling_params: SamplingParams):
        raise NotImplementedError

class PackedCausalTransformerGenerator(TransformerGenerator):
    def __init__(
        self,
        cfg: TransformerGeneratorArgs,
        model_path: str,
        tokenizer_path: str,
    ):
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_pack_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        super().__init__(cfg, model_path, tokenizer_path)
        assert cfg.n == 1, "Only support generating one sample for now!"

        self.model, self.tokenizer = load_consolidated_model_and_tokenizer(model_path, device=cfg.device)
        self.encode_fn = partial(self.tokenizer.encode, add_bos=True, add_eos=False)

        # compatible with VLLM setups
        self.default_sampling_params = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature, 
            top_p=cfg.top_p or 1.0, top_k=cfg.top_k or -1,
            max_tokens=cfg.max_gen_len, truncate_prompt_tokens=cfg.max_prompt_len,
            stop=cfg.until, include_stop_str_in_output=not cfg.strip_until,
        )

        self.max_pack_tokens = cfg.max_pack_tokens
        self.max_tokens = min(cfg.max_tokens or np.inf, getattr(self.model, "max_seqlen", np.inf))
        
        self.device = cfg.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None

        if cfg.logit_dtype and cfg.logit_dtype != cfg.dtype:
            logger.info(f"Using {cfg.logit_dtype} for logits")
            self.logit_dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.logit_dtype]
        else:
            self.logit_dtype = self.dtype

        self.model.eval()

    def clear_cache(self, offset):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_pack_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor, max_gen_len: int):
        # The KV cache is a fixed size tensor of size max_pack_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + max_gen_len
        max_pack_tokens = self.max_pack_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_pack_tokens
        padded_lengths[-1] += max_pack_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        self.prefill_mask = create_block_mask(
            doc_mask_mod, 1, None, lengths.sum(), max_pack_tokens
        )

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)

    @torch.compiler.disable
    def setup_generation(self, lengths):
        # KV Cache offset is set to the start of the padded documents
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = self.padded_doc_start
        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = lengths.clone()
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = torch.arange(lengths.size(0), device=lengths.device)

    # From here on some methods for generation
    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor, max_gen_len: int):
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths, max_gen_len=max_gen_len)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask,
            attn_impl="flex_attention",
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token):
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        mask = doc_mask & caus_mask
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id,  # n_seqs
            mask=mask,
            attn_impl="sdpa",
        )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(self, prompts = None, prompt_token_ids = None, sampling_params: Optional[SamplingParams] = None):
        assert not (prompts is None and prompt_token_ids is None), "Either prompts or prompt_token_ids must be provided!"
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        max_until_size = max([len(e) for e in sampling_params.stop]) if sampling_params.stop else 1
        
        # Tokenize
        if prompt_token_ids is not None:
            tokenized_prompts = prompt_token_ids
        else:
            tokenized_prompts = [
                self.encode_fn(p) for p in prompts
            ]
        # Truncate
        max_prompt_len = sampling_params.truncate_prompt_tokens or min(
            self.max_tokens - sampling_params.max_tokens, self.max_pack_tokens - sampling_params.max_tokens
        )
        tokenized_prompts = [p[-max_prompt_len:] for p in tokenized_prompts]
        # Account for the generation in lengths
        padded_lengths = [len(p) + sampling_params.max_tokens for p in tokenized_prompts]
        it = batch_prompts(tokenized_prompts, self.max_pack_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)

        results = []
        seq_idx = 0
        for batch in it:
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            generated_loglikelihoods = [[] for _ in range(n_seqs)]
            generated_is_greedy = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)

            # Prefilling cache
            prompt_logits = self.prefill(packed_batch.unsqueeze(0), lengths, max_gen_len=sampling_params.max_tokens)
            prompt_logits = prompt_logits.to(self.logit_dtype)

            # Selecting last token in each prompt
            all_tokens, all_loglikelihood, all_greedy = sample_tokens(
                prompt_logits, sampling_params.temperature, sampling_params.top_p, sampling_params.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]
            start_loglikelihood = all_loglikelihood[:, lengths.cumsum(0) - 1]
            start_greedy = all_greedy[:, lengths.cumsum(0) - 1]

            for seq_id, (tok, loglik, gr) in enumerate(zip(start_token.squeeze(0).tolist(), start_loglikelihood.squeeze(0), start_greedy.squeeze(0))):  # squeeze batch dim (=1) due to sequence packing
                generated_tokens[seq_id].append(tok)
                generated_loglikelihoods[seq_id].append(loglik.cpu().to(torch.float32).item())
                generated_is_greedy[seq_id].append(gr.cpu().item())

            current_token = start_token
            for i in range(1, sampling_params.max_tokens):

                next_logits = self.generate_next_token(current_token)
                next_logits = next_logits.to(self.logit_dtype)

                next_token, next_loglikelihood, next_greedy = sample_tokens(
                    next_logits.clone(), sampling_params.temperature, sampling_params.top_p, sampling_params.top_k
                )

                for seq_id, (tok, loglik, gr) in enumerate(zip(next_token.squeeze(0).tolist(), next_loglikelihood.squeeze(0), next_greedy.squeeze(0))):   # squeeze batch dim (=1) due to sequence packing
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        generated_loglikelihoods[seq_id].append(loglik.cpu().to(torch.float32).item())
                        generated_is_greedy[seq_id].append(gr.cpu().item())
                        current_end_str = self.tokenizer.decode(
                            generated_tokens[seq_id][-max_until_size :]
                        )
                        contains_end_string = any(
                            [e in current_end_str for e in sampling_params.stop]
                        )
                        is_done[seq_id] = (
                            contains_end_string or tok == self.tokenizer.eos_id
                        )
                if all(is_done):
                    break

                current_token = next_token

            decoded_generations = [self.tokenizer.decode(g) for g in generated_tokens]
            if len(sampling_params.stop) > 0 and not sampling_params.include_stop_str_in_output:
                for u in sampling_params.stop:
                    decoded_generations = [d[:d.find(u)] if d.find(u) != -1 else d for d in decoded_generations]

            prompt_tokens = tokenized_prompts[seq_idx:seq_idx+n_seqs]
            seq_idx += n_seqs
            
            prompt_loglikelihoods = []
            prompt_greedys = []
            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist())
            ):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                prompt_loglikelihoods.append((-F.cross_entropy(x, y, reduction="none").cpu().to(torch.float32)).tolist())
                prompt_greedys.append((x.argmax(dim=-1) == y).cpu().tolist())
            

            for done, gen, tok, loglik, gr, p_tok, p_loglik, p_gr in zip(is_done, decoded_generations, generated_tokens, generated_loglikelihoods, generated_is_greedy, prompt_tokens, prompt_loglikelihoods, prompt_greedys):
                # assert (len(p_tok) - 1) == len(p_loglik) == len(p_gr), "Number of prompt tokens, loglikelihoods, and greedys must be the same, got {} {} {}".format(len(p_tok), len(p_loglik), len(p_gr))
                # FIXME: only one generation output for now
                generation_outputs = [
                    GenerationOutput(
                        is_done=done, generation=gen, 
                        generated_tokens=tok, generated_loglikelihoods=loglik, generated_greedys=gr,
                    )
                ]
                results.append(
                    GenerationResults(
                        prompt_tokens=p_tok, prompt_loglikelihoods=p_loglik, prompt_greedys=p_gr,
                        generation_outputs=generation_outputs,
                    ))

        return results
    
    def clean_up(self):
        self.model.cpu()
        #  TODO: should also clean the cache

        del self.model
        del self.tokenizer


class VLLMGenerator(TransformerGenerator):
    def __init__(
        self,
        cfg: TransformerGeneratorArgs,
        model_path: str,
        tokenizer_path: str,
    ):
        super().__init__(cfg, model_path, tokenizer_path)

        vllm_kwargs = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "load_format": "lingua",
            "seed": cfg.seed,
            "dtype": dict(fp32="float32", fp16="float16", bf16="bfloat16")[cfg.dtype],
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
        }

        if cfg.max_num_seqs is not None:
            vllm_kwargs["max_num_seqs"] = cfg.max_num_seqs
        
        self.llm = LLM(**vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.encode_fn = self.tokenizer.encode

        max_seqlen = min(
            cfg.max_tokens or np.inf,
            getattr(self.llm.llm_engine.model_config, "max_model_len", np.inf)
        )

        max_prompt_len = cfg.max_prompt_len or min(
            max_seqlen - cfg.max_gen_len, cfg.max_tokens or np.inf - cfg.max_gen_len
        )

        self.default_sampling_params = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature, 
            top_p=cfg.top_p or 1.0, top_k=cfg.top_k or -1,
            max_tokens=cfg.max_gen_len, truncate_prompt_tokens=max_prompt_len,
            logprobs=True, prompt_logprobs=0, # only the logprob of the prompt tokens themselves
            stop=cfg.until, include_stop_str_in_output=not cfg.strip_until,
            skip_special_tokens=False, spaces_between_special_tokens=False,
        )

        self.max_tokens = max_seqlen
        self.device = cfg.device

    def generate(self, prompts = None, prompt_token_ids = None, sampling_params: Optional[Union[SamplingParams, Sequence[SamplingParams]]] = None):
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        outputs = self.llm.generate(prompts=prompts, prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        results = []

        for output in outputs:
            # Extract prompt-related information
            prompt_tokens = output.prompt_token_ids

            if output.prompt_logprobs is not None:  # could be none if prompt tokens are too long
                if output.prompt_logprobs[0] is None:
                    start_idx = 1
                else:
                    start_idx = 0
                prompt_logprobs = [l[t].logprob for (t, l) in zip(prompt_tokens[start_idx:], output.prompt_logprobs[start_idx:])]
                prompt_greedy = [l[t].rank == 1 for (t, l) in zip(prompt_tokens[start_idx:], output.prompt_logprobs[start_idx:])]
            else:
                prompt_logprobs = []
                prompt_greedy = []

            # Extract generation-related information
            generation_outputs = []

            for i, gen_output in enumerate(output.outputs):
                generated_text = gen_output.text
                generated_tokens = gen_output.token_ids

                generated_logprobs = [l[t].logprob for (t, l) in zip(generated_tokens, gen_output.logprobs)]
                generated_greedy = [l[t].rank == 1 for (t, l) in zip(generated_tokens, gen_output.logprobs)]
                generation_outputs.append(GenerationOutput(
                    is_done=gen_output.finish_reason == "stop",
                    generation=generated_text,
                    generated_tokens=generated_tokens,
                    generated_loglikelihoods=generated_logprobs,
                    generated_greedys=generated_greedy,
                ))
            
            # Create GenerationResult object
            results.append(GenerationResults(
                prompt_tokens=prompt_tokens,
                prompt_loglikelihoods=prompt_logprobs,
                prompt_greedys=prompt_greedy,
                generation_outputs=generation_outputs,
            ))

        return results
    
    def clean_up(self):
        # TODO: fix this
        del self.llm


def load_consolidated_model_and_tokenizer(
    consolidated_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
    device="cuda",
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / LINGUA_CONFIG_NAME
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)

    logger.info(f"Loading model state dict from {ckpt_path}")
    if is_llama_pretrained_ckpt(ckpt_path):
        model.load_state_dict(st_dict)
    else:
        model.load_state_dict(st_dict["model"])

    if device == "cuda":
        model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer


def consolidate_and_load_model_and_tokenizer(model_dir: str, step: Optional[int] = None):
    ckpt_base_dir = Path(model_dir) / "checkpoints"
    if step is None:
        checkpoint = sorted(ckpt_base_dir.glob("*"))[-1]  # latest checkpoint
    else:
        checkpoint = ckpt_base_dir / f"{step:010d}"
    checkpoint_dir = ckpt_base_dir / checkpoint

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    consolidate_path = checkpoint_dir / CONSOLIDATE_FOLDER
    if not consolidate_path.exists():
        consolidate_path = consolidate_checkpoints(checkpoint_dir)

    config = consolidate_path / LINGUA_CONFIG_NAME
    config = OmegaConf.load(config)

    tokenizer_path = config.data.tokenizer.path
    return consolidate_path, tokenizer_path


def load_transformer_generator(gen_cfg: GenerateArgs):
    if len(gen_cfg.generation_until) > 0:
        gen_cfg.generator.until += gen_cfg.generation_until

    logger.info(f"Loading transformer generator with config: {gen_cfg.generator}")
    
    model_path, tokenizer_path = consolidate_and_load_model_and_tokenizer(gen_cfg.model_dir, gen_cfg.step)
    if gen_cfg.use_vllm:
        generator = VLLMGenerator(gen_cfg.generator, model_path, tokenizer_path)
    else:
        generator = PackedCausalTransformerGenerator(gen_cfg.generator, model_path, tokenizer_path)

    return generator
    

def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        GenerateArgs, cfg,
    )

    init_logger()
    logger.info(gen_cfg)

    generator = load_transformer_generator(gen_cfg)

    prompts = []
    if gen_cfg.prompt_path is not None:
        logger.info(f"Loading prompts from {gen_cfg.prompt_path}")

        if gen_cfg.prompt_path.endswith(".txt"):
            with open(gen_cfg.prompt_path, "r") as f:
                prompts = [line.strip() for line in f.readlines()]
        elif gen_cfg.prompt_path.endswith(".json"):
            with open(gen_cfg.prompt_path, "r") as f:
                prompts = json.load(f)
        elif gen_cfg.prompt_path.endswith(".py"):
            with open(gen_cfg.prompt_path, "r") as f:
                namespace = {}
                exec(f.read(), namespace)
                prompts = namespace['get_prompts']()
        else:
            raise ValueError(f"Unknown prompt file type: {gen_cfg.prompt_path}")
    else:
        # Allow multiple prompts
        while True:
            prompt = input("Enter a prompt (or press enter to finish): ")
            if not prompt:
                break
            prompts.append(prompt)

    if gen_cfg.generation_prefix is not None:
        prompts = [p + gen_cfg.generation_prefix for p in prompts]


    # Start generation
    start_time = time.time()
    results = generator.generate(prompts)
    end_time = time.time()

    # Calculate tokens per second
    # total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    total_tokens = sum(len(g.generated_tokens) for r in results for g in r.generation_outputs)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, r in enumerate(results):
        print("="*100)
        print(f"\nPrompt {i+1}: {prompts[i]}")
        for j, g in enumerate(r.generation_outputs):
            print("\n" + "-" * 100)
            print(f"Generated Text {i+1}.{j+1}:\n{g.generation}")
            # print("All loglikelihoods:", r.generated_loglikelihoods)
            print(f"Loglikelihood {i+1}.{j+1}: {sum(g.generated_loglikelihoods):.4f}")
            print(f"Done {i+1}.{j+1}: {g.is_done}")
            print(f"Greedy {i+1}.{j+1}: {all(g.generated_greedys)}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
