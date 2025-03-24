"""
Bootstrap the latents by sampling from a trained posterior, and potentially selected with Monte Carlo methods.
"""

import os
import logging
import time
import torch
import copy
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.special import logsumexp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from lingua.tokenizer import SpecialTokens
from apps.main.generate import GenerateArgs, load_transformer_generator, TransformerGeneratorArgs, consolidate_and_load_model_and_tokenizer
from apps.main.gen_utils.cache_utils import SQLiteLockConnection

logger = logging.getLogger()


@dataclass
class BootstrapLatentsArgs(GenerateArgs):
    use_vllm: bool = True
    suffix_delimiter: str = SpecialTokens.START_OF_LATENT.value + SpecialTokens.END_OF_LATENT.value  # delimiter between prefix and suffix in the prompt
    generation_prefix: str = SpecialTokens.START_OF_LATENT.value + SpecialTokens.POSTERIOR_PREFIX.value
    generation_until: List[str] = field(default_factory=lambda: [SpecialTokens.END_OF_LATENT.value])
    joint_prefix: str = SpecialTokens.START_OF_LATENT.value + SpecialTokens.PRIOR_PREFIX.value  # prefix for the joint likelihood estimation
    joint_suffix: str = SpecialTokens.END_OF_LATENT.value  # suffix for the joint likelihood estimation
    num_total_samples: int = 1  # number of samples to generate, where the final sample is selected with random sampling or Monte Carlo methods
    compute_joint_likelihood: bool = True  # whether to compute the joint likelihood
    compute_elbo: bool = False  # whether to use the sampled latents to compute the ELBO
    num_elbo_samples: List[int] = field(default_factory=lambda: [1])  # number of samples to compute the ELBO
    apply_monte_carlo: bool = False  # whether to apply Monte Carlo methods to select the latents
    print_debug_outs: bool = False  # whether to print the debug outputs
    load_model: bool = True  # whether to actually load the model, required for generation
    load_cache: bool = True  # whether to load the generated latents from the cache
    save_cache: bool = True  # whether to save the generated latents to the cache
    save_every_n_samples: int = 10000  # batch size to process the latents and save to the cache
    cache_dir: Optional[str] = None  # directory to save the cache, if not specified, save to the model dir
    cache_subdir: str = "cache"  # subdirectory to save the cache


class LatentCache(SQLiteLockConnection):
    def __init__(self, cache_path: str, log_joint_logprob: bool = True):
        self.log_joint_logprob = log_joint_logprob
        keys_type_dict, values_type_dict = self.latent_cache_type_dicts
        super().__init__(db_path=cache_path, keys_type_dict=keys_type_dict, values_type_dict=values_type_dict)

    @property
    def latent_cache_type_dicts(self):
        key_type_dict = {
            "model": "TEXT",
            "prompt": "TEXT",
            "temperature": "REAL",
            "top_p": "REAL",
            "top_k": "REAL",
            "max_prompt_len": "INTEGER",
            "max_tokens": "INTEGER",
            "max_gen_len": "INTEGER",
            "dtype": "TEXT",
        }
        value_type_dict = {
            "latent": "TEXT",
            "done": "BOOLEAN",
            "prompt_len": "INTEGER",
            "gen_len": "INTEGER",
            "gen_logprob": "REAL",
            "gen_greedy": "BOOLEAN",
        }
        if self.log_joint_logprob:
            value_type_dict["joint_logprob"] = "REAL"
            value_type_dict["log_importance_weight"] = "REAL"
        return key_type_dict, value_type_dict


def process_batch_samples(generator, samples, num_needed_per_sample, cached_outputs, cfg, input_key="chunked_text_pair", output_key="latent"):
    """Process a batch of samples, generating num_needed outputs for each sample."""
    # Prepare prompts
    prompts_to_generate = []
    for sample, num_needed in zip(samples, num_needed_per_sample):
        # simply copy the prompt multiple times - for some reason, vllm seems to be slower when using n > 1
        prompts_to_generate.extend([sample["formatted_prompt"]] * num_needed)

    # Generate latents
    gen_results = generator.generate(prompts_to_generate)
    
    # Process generation results
    outputs = []
    dones = []
    cur_idx = 0
    for i, (sample, num_needed, cached_outs) in enumerate(zip(samples, num_needed_per_sample, cached_outputs)):
        # append cached outputs
        sample_outputs = cached_outs.copy()
        sample_dones = [o["done"] for o in cached_outs]

        # append new generated outputs
        for r in gen_results[cur_idx:cur_idx + num_needed]:
            gen = r.generation_outputs[0]
            out = copy.deepcopy(sample)
            out[output_key] = gen.generation
            out["done"] = gen.is_done
            out["prompt_len"] = len(r.prompt_tokens)
            out["gen_len"] = len(gen.generated_tokens)
            out["gen_logprob"] = sum(gen.generated_loglikelihoods)
            out["gen_greedy"] = all(gen.generated_greedys)
            sample_outputs.append(out)
            sample_dones.append(gen.is_done)

        assert len(sample_outputs) == cfg.num_total_samples and len(sample_dones) == cfg.num_total_samples, f"Number of outputs & dones for sample {i} must be {cfg.num_total_samples}, got {len(outputs[i])} for outputs and {len(dones[i])} for dones"
        outputs.append(sample_outputs)
        dones.append(sample_dones)

        cur_idx += num_needed

    # Compute joint likelihood if needed
    if cfg.compute_joint_likelihood:
        encode_fn = generator.encode_fn
        max_tokens = generator.max_tokens
        
        tokenized_prompts_with_latents = []
        num_overflows = 0
        num_suffix_truncated = 0
        
        for sample, outs in zip(samples, outputs):
            tokenized_prompts_per_sample = []
            for out in outs:
                joint_prompt = (sample[input_key]["prefix"] + cfg.joint_prefix + 
                              out[output_key] + cfg.joint_suffix + 
                              sample[input_key]["suffix"])
                tokenized_prompts_per_sample.append(encode_fn(joint_prompt))
            
            max_joint_prompt_len = max(len(t) for t in tokenized_prompts_per_sample)
            all_within_context_limit = max_joint_prompt_len <= max_tokens
            
            prefix_len = len(encode_fn(sample[input_key]["prefix"]))   # BOS is also be included here
            if not all_within_context_limit:
                num_overflows += 1
                
                num_tokens_to_remove = max_joint_prompt_len - max_tokens
                num_prefix_tokens_to_remove = min(prefix_len, num_tokens_to_remove)
                num_suffix_tokens_to_remove = num_tokens_to_remove - num_prefix_tokens_to_remove
                prefix_len = prefix_len - num_prefix_tokens_to_remove
                
                if num_suffix_tokens_to_remove > 0:
                    num_suffix_truncated += 1
                
                truncated_prompts = []
                for t in tokenized_prompts_per_sample:
                    t = t[num_prefix_tokens_to_remove:]
                    if num_suffix_tokens_to_remove > 0:
                        t = t[:-num_suffix_tokens_to_remove]
                    truncated_prompts.append(t)
                tokenized_prompts_per_sample = truncated_prompts
            
            tokenized_prompts_with_latents.extend(tokenized_prompts_per_sample)
            sample["tokenized_prefix_len"] = prefix_len
            sample["num_suffix_token_truncated"] = (num_suffix_tokens_to_remove 
                if not all_within_context_limit else 0)
        
        if num_overflows > 0:
            logger.info(f"Likelihood estimation: {num_overflows}/{len(samples)} generations are overflowed "
                       f"and truncated to context limit of {max_tokens} tokens.")
            logger.warning(f"Number of suffix truncated: {num_suffix_truncated}/{len(samples)}")
        
        # Compute likelihoods
        param = copy.deepcopy(generator.default_sampling_params)
        param.truncate_prompt_tokens = None
        param.n = 1
        param.max_tokens = 1
        
        lik_results = generator.generate(prompt_token_ids=tokenized_prompts_with_latents, sampling_params=param)
        lik_results = [lik_results[i:i+cfg.num_total_samples] for i in range(0, len(lik_results), cfg.num_total_samples)]
        assert len(lik_results) == len(samples), "Number of likelihood results and samples must be the same"
        
        # Group likelihood results by sample
        for sample, outs, lik_outs in zip(samples, outputs, lik_results):
            for out, lik_out in zip(outs, lik_outs):
                prefix_bound = sample["tokenized_prefix_len"]
                out["joint_logprob"] = sum(lik_out.prompt_loglikelihoods[prefix_bound:])
                out["log_importance_weight"] = out["joint_logprob"] - out["gen_logprob"]
                out["num_suffix_token_truncated"] = sample["num_suffix_token_truncated"]
            
    return outputs, dones


def bootstrap_latents(samples: List[Dict[str, Any]], shard_idx: int, input_key="chunked_text_pair", output_key="latent", cfg: Optional[BootstrapLatentsArgs] = None, seed: int = 42, generator = None, **kwargs):
    # Check and set the config
    cfg.generator.seed = int(time.time() * 1000) % (2**32)  # randomize seed for generation, otherwise we might get same generations as in the cache
    if not cfg.generator.strip_until:
        logger.warning("Strip_until should be set to True to remove the end of latent tokens from the generation. Setting it to True...")
        cfg.generator.strip_until = True
    if generator is not None and generator.default_sampling_params.include_stop_str_in_output:
        logger.warning("Include stop string in output is set to True in the generator's sampling params. Setting it to False...")
        generator.default_sampling_params.include_stop_str_in_output = False
    
    if (cfg.apply_monte_carlo or cfg.compute_elbo) and not cfg.compute_joint_likelihood:
        cfg.compute_joint_likelihood = True
        logger.warning("Joint likelihood estimation is not computed when applying Monte Carlo methods or computing ELBO. Setting it to True...")

    if generator is None:
        if cfg.load_model:
            logger.info(f"Sampling latents with config: {cfg}")
            generator = load_transformer_generator(cfg)
            model_path = generator.model_path
        else:
            logger.warning("Generator is not loaded, skipping generation...")
            generator = None
            model_path, _ = consolidate_and_load_model_and_tokenizer(cfg.model_dir, cfg.step)
    else:
        model_path = generator.model_path

    # Setup the cache
    if cfg.cache_dir is None: 
        cfg.cache_dir = os.path.join(model_path, cfg.cache_subdir)  # save to the model dir so that it can be shared across different runs
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, f"bootstrap_latents_shard_{shard_idx}.db")
    cache = LatentCache(cache_path, log_joint_logprob=True)

    try:
        base_key_dict = {
            "model": str(model_path),
            "temperature": cfg.generator.temperature,
            "top_p": cfg.generator.top_p,
            "top_k": cfg.generator.top_k,
            "max_prompt_len": cfg.generator.max_prompt_len,
            "max_tokens": cfg.generator.max_tokens,
            "max_gen_len": cfg.generator.max_gen_len,
            "dtype": cfg.generator.dtype,
        }

        all_samples = []
        all_cache_key_dicts = []
        num_prompts_needed = []
        copy_keys = [input_key]
        
        for sample in samples:
            assert "prefix" in sample[input_key] and "suffix" in sample[input_key], "Input must contain prefix and suffix"

            if sample[input_key]["prefix"] in ["", "N/A"]:
                sample[input_key]["prefix"] = ""
                prompt = sample[input_key]["suffix"]
            else:
                prompt = sample[input_key]["prefix"] + cfg.suffix_delimiter + sample[input_key]["suffix"]
            prompt += cfg.generation_prefix
            
            sample["formatted_prompt"] = prompt
            all_cache_key_dicts.append({**base_key_dict, "prompt": prompt})
            all_samples.append(sample)

        # Batch cache query
        if cfg.load_cache:
            all_cached_outputs = cache.get_items_batch(
                all_cache_key_dicts,
                limit=cfg.num_total_samples,
                value_only=True,
                return_ids=True,
                batch_size=2000,
            )
        else:
            all_cached_outputs = [[] for _ in range(len(all_samples))]

        # Process cached results
        for sample, cached_outputs in zip(all_samples, all_cached_outputs):
            for o in cached_outputs:
                [o.update({k: sample[k]}) for k in copy_keys]
                o["db_cache_id"] = o.pop("id", None)

            if len(cached_outputs) < cfg.num_total_samples:
                num_needed = cfg.num_total_samples - len(cached_outputs)
                num_prompts_needed.append(num_needed)
            else:
                num_prompts_needed.append(0)

        logger.info(f"Generating {cfg.num_total_samples} latents for {len(all_samples)} samples, totalling {cfg.num_total_samples * len(all_samples)} prompts")
        num_cached_latents = sum(len(o) for o in all_cached_outputs)
        if num_cached_latents > 0:
            logger.info(f"Loaded {num_cached_latents} cached latents, generating latents for the remaining {sum(num_prompts_needed)} prompts")

        # Process in batches
        all_outputs = []
        all_dones = []
        
        # normalize the batch size by the number of samples to generate
        # and no need to batch if not saving to cache
        batch_size = cfg.save_every_n_samples // cfg.num_total_samples if cfg.save_cache else len(all_samples)
        num_batches = (len(all_samples) + batch_size - 1) // batch_size

        logger.info(f"Processing {num_batches} batches with, each batch contains {batch_size} x {cfg.num_total_samples} = {batch_size * cfg.num_total_samples} prompts")
        
        for batch_idx in tqdm(range(num_batches)):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(all_samples))
            batch_samples = all_samples[batch_start:batch_end]
            batch_needed = num_prompts_needed[batch_start:batch_end]
            batch_cached = all_cached_outputs[batch_start:batch_end]
            
            if sum(batch_needed) > 0:
                assert generator is not None, "Generator is not loaded, set load_model=True to load the model."
                batch_outputs, batch_dones = process_batch_samples(
                    generator=generator,
                    samples=batch_samples,
                    num_needed_per_sample=batch_needed,
                    cached_outputs=batch_cached,
                    cfg=cfg,
                    input_key=input_key,
                    output_key=output_key
                )
            else:
                batch_outputs = [outs.copy() for outs in batch_cached]
                batch_dones = [[o["done"] for o in outs] for outs in batch_outputs]
            
            all_outputs.extend(batch_outputs)
            all_dones.extend(batch_dones)
            
            # Save batch results to cache immediately if enabled
            if cfg.save_cache:
                added_latents = []
                updated_latents = []
                updated_ids = []
                for sample, outs in zip(batch_samples, batch_outputs):
                    sample_key_dict = {**base_key_dict, "prompt": sample["formatted_prompt"]}
                    for out in outs:
                        item = {"key_dict": sample_key_dict, "value_dict": out}
                        if out.get("db_cache_id", None) is not None:
                            updated_latents.append(item)
                            updated_ids.append(out.pop("db_cache_id"))
                        else:
                            added_latents.append(item)

                if updated_latents:
                    logger.info(f"Updating {len(updated_latents)} latents in for batch {batch_start//batch_size} (sample {batch_start} to {batch_end})")
                    cache.update_items(updated_latents, updated_ids)
                if added_latents:
                    logger.info(f"Saving {len(added_latents)} latents in batch {batch_start//batch_size} (sample {batch_start} to {batch_end})")
                    cache.add_items(added_latents)

        # Check for early terminations
        all_dones = [d for dones in all_dones for d in dones]  # flatten the list
        if sum(all_dones) < len(all_dones):
            num_early_terminated = len(all_dones) - sum(all_dones)
            logger.warning(f"{num_early_terminated}/{len(all_dones)} generations were early terminated")

        # Select final outputs
        final_outputs = []
        if cfg.num_total_samples == 1:
            logger.info("No sampling applied, using the single generation for each sample.")
            final_outputs = [out[0] for out in all_outputs]
        elif cfg.apply_monte_carlo:
            logger.info(f"Applying Monte Carlo methods to select the final latent out of {cfg.num_total_samples} samples.")
            rng = np.random.default_rng(seed)
            for samples, outs in zip(all_samples, all_outputs):
                imp_weight_dist = np.array([out["log_importance_weight"] for out in outs])
                imp_weight_dist = imp_weight_dist - np.max(imp_weight_dist)  # for numerical stability
                imp_weight_dist = np.exp(imp_weight_dist)
                imp_weight_dist = imp_weight_dist / np.sum(imp_weight_dist)
                logger.debug(f"Importance weighted distribution: {np.array2string(imp_weight_dist, precision=8)}")

                if cfg.print_debug_outs:
                    print("\n\n" + "=" * 100 + "\n\n")
                    print("\n\n" + "Prefix:\n" + samples[input_key]["prefix"])
                    print("\n\n" + "Suffix:\n" + samples[input_key]["suffix"])
                    print("\n\n" + ("\n\n" + "-" * 100 + "\n\n").join([f"Sample {idx} (prob: {imp_weight_dist[idx]:.8f}, imp: {o['log_importance_weight']:.8f}, gen: {o['gen_logprob']:.8f}, joint: {o['joint_logprob']:.8f}):\n{o[output_key]}" for idx, o in enumerate(outs)]))
                final_outputs.append(rng.choice(outs, p=imp_weight_dist))
        else:
            logger.info(f"Randomly sample 1 latent out of {cfg.num_total_samples} from each sample.")
            rng = np.random.default_rng(seed)
            for outs in all_outputs:
                final_outputs.append(rng.choice(outs, 1)[0])

        if cfg.compute_elbo:
            for sample, f_out, all_outs in zip(all_samples, final_outputs, all_outputs):
                log_imp_weights = np.array([out["log_importance_weight"] for out in all_outs])
                for n in cfg.num_elbo_samples:
                    f_out[f"elbo_{n}"] = logsumexp(log_imp_weights[:n]) - np.log(n)

        assert len(final_outputs) == len(all_samples), "Number of final outputs and samples must be the same"
        
        total_input_length = sum([o["prompt_len"] for o in final_outputs])
        total_output_length = sum([o["gen_len"] for o in final_outputs])
        stats = {
            "total_num_samples": len(final_outputs),
            f"total_{input_key}_length": total_input_length,
            f"total_{output_key}_length": total_output_length,
            f"avg_{input_key}_length": total_input_length / len(final_outputs),
            f"avg_{output_key}_length": total_output_length / len(final_outputs),
            "compression_ratio": total_output_length / total_input_length,
        }

        return final_outputs, stats
    finally:
        cache.close()