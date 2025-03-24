from dataclasses import dataclass
from typing import Optional
import os
from itertools import chain

import tiktoken

from apps.main.gen_utils.misc_utils import read_jsonl
from apps.main.gen_utils.misc_utils import get_logger

logger = get_logger(__name__)

@dataclass
class GenerationResults:
    shard_idx: int
    stats: dict
    success: bool
    error_message: Optional[str] = None


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, allowed_special="all"))


def get_token_stats(shard_outputs):
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    
    text_lengths = []
    latent_lengths = []
    total_num_samples = 0
    for output in shard_outputs:
        text_lengths.append(count_tokens(output["text"], tokenizer))
        latent_lengths.append(count_tokens(output["latent"], tokenizer))
        total_num_samples += 1

    total_text_tokens = sum(text_lengths)
    total_latent_tokens = sum(latent_lengths)

    stats = {
        "total_num_samples": total_num_samples,
        "total_text_length": total_text_tokens,
        "total_latent_length": total_latent_tokens,
        "avg_text_length": total_text_tokens / total_num_samples,
        "avg_latent_length": total_latent_tokens / total_num_samples,
        "compression_ratio": total_latent_tokens / total_text_tokens,
    }
    return stats


def load_shard_results(shard_idx, train_output_dir, val_output_dir):
    train_shard_file = os.path.join(train_output_dir, f"train_{shard_idx:08d}.jsonl.zst")
    val_shard_file = os.path.join(val_output_dir, f"val_{shard_idx:08d}.jsonl.zst")
    if os.path.exists(train_shard_file):
        train_shard_outputs = read_jsonl(train_shard_file)
        if os.path.exists(val_shard_file):
            val_shard_outputs = read_jsonl(val_shard_file)
        else:
            logger.warning(f"Validation shard {shard_idx} not found, skipping.")
            val_shard_outputs = []
        
        shard_outputs = chain(train_shard_outputs, val_shard_outputs)
        return get_token_stats(shard_outputs)
    else:
        logger.warning(f"Training shard {shard_idx} not found, skipping.")
        return None
    

def aggregate_stats(stats_list: list[dict]):
    agg_stats = {}
    for key in stats_list[0].keys():
        if key.startswith("total_"):
            agg_stats[key] = sum([s[key] for s in stats_list])
        elif key.startswith("avg_"):
            total_num_samples = agg_stats.get("total_num_samples", sum([s["total_num_samples"] for s in stats_list]))
            agg_stats[key] = sum([s[key] * s["total_num_samples"] for s in stats_list]) / total_num_samples
        elif key == "compression_ratio":
            pass
        else:
            logger.warning(f"Unknown key {key} for aggregation.")
    
    if "total_latent_length" in agg_stats and "total_text_length" in agg_stats:
        agg_stats["compression_ratio"] = agg_stats["total_latent_length"] / agg_stats["total_text_length"]
    
    return agg_stats


def aggregate_results(results: list[GenerationResults]):
    stats_list = []
    for result in results:
        if result.success:
            stats_list.append(result.stats)
        else:
            logger.error(f"Skipping shard {result.shard_idx} with error: {result.error_message}")

    return aggregate_stats(stats_list)