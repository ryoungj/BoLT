# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Prepare datasets for lingua. Given a dataset directory, transform it into a set of shards that can be used for training and validation.

Usage:

Sampled dataset:
python apps/main/prepare_data.py --data-config-dir ./configs/data --src-dataset-name dclm_baseline_sample_12b --tgt-dataset-name dclm_baseline_sample_3b --mode sampled_or_filtered --sample-ratio 0.25 --num-processes 16

Prepare final shuffled dataset used for training:
python apps/main/prepare_data.py --data-config-dir ./configs/data --src-dataset-name dclm_baseline_sample_3b --num-chunks 1 --num-val-samples 2000 --memory 64 --mode final_shuffled --tmp-dir ./data/tmp
"""

import argparse
import os
import time
import subprocess
import requests
import shutil
import uuid
import numpy as np
from multiprocessing import Pool
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
from functools import partial
from tqdm import tqdm
import json

from apps.main.gen_utils.misc_utils import read_jsonl, write_jsonl
from apps.main.gen_utils.misc_utils import filter_iterator, sample_iterator


DATA_SUFFIX = ".jsonl"
VAL_DATA_SUFFIX = f".val{DATA_SUFFIX}"

class DatasetFormat(str, Enum):
    ORIGINAL = "original"  # original downloaded format, e.g., .jsonl.zst
    PROCESSED = "processed"  # processed dataset, in similar format to original
    FINAL = "final"  # final shuffled dataset used for training, e.g., .jsonl

@dataclass
class DatasetStats:
    num_samples: int = 0
    content_word_length: int = 0
    src_num_samples: int = 0
    src_content_word_length: int = 0


@dataclass
class DatasetConfig:
    uuid: str
    name: str
    path: str
    sources: List[str]
    format: DatasetFormat
    stats: Optional[DatasetStats] = None
    process_kwargs: Optional[Dict[str, Any]] = None


def load_dataset_config(data_config_dir, dataset_name):
    dataset_config_path = os.path.join(data_config_dir, dataset_name + ".json")
    print(f"Loading dataset config from {dataset_config_path}")
    with open(dataset_config_path, "r") as f:
        return DatasetConfig(**json.load(f))

def save_dataset_config(dataset_config, data_config_dir, dataset_name):
    dataset_config_path = os.path.join(data_config_dir, dataset_name + ".json")
    print(f"Saving dataset config to {dataset_config_path}")
    with open(dataset_config_path, "w") as f:
        json.dump(asdict(dataset_config), f, indent=4)
    

def merge_shard_stats(shard_stats):
    return DatasetStats(
        num_samples=sum(result.num_samples for result in shard_stats),
        content_word_length=sum(result.content_word_length for result in shard_stats),
        src_num_samples=sum(result.src_num_samples for result in shard_stats),
        src_content_word_length=sum(result.src_content_word_length for result in shard_stats)
    )


def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def process_shard_file(shard_idx, shard_path, output_shard_path, content_key="text", sample_ratio=1.0, min_length=0, max_length=np.inf, keep_metadata=False, seed=42):
    seed += shard_idx
    inputs = read_jsonl(shard_path)
    
    total_length = 0
    total_num = 0

    word_pattern = re.compile(r'\b\w+\b')
    content_word_length_key = f"{content_key}_word_length"
    def _input_stats_calc_iter(_inputs):
        nonlocal total_length
        nonlocal total_num

        # for item in tqdm(_inputs, desc=f"Shard {shard_idx}: Processing", position=shard_idx):
        for item in _inputs:
            item[content_word_length_key] = len(word_pattern.findall(item[content_key]))
            total_length += item[content_word_length_key]
            total_num += 1

            yield item
    
    inputs = _input_stats_calc_iter(inputs)

    if min_length > 0 or max_length < np.inf:
        print(f"Shard {shard_idx}: Filtering inputs by length: " + (f"Min length = {min_length}, " if min_length > 0 else "") + (f"Max length = {max_length}" if max_length < np.inf else ""))
        outputs = filter_iterator(inputs, lambda x: min_length <= x[content_word_length_key] <= max_length)
    else:
        outputs = inputs
    
    if sample_ratio is not None and sample_ratio < 1:
        print(f"Shard {shard_idx}: Sampling {sample_ratio} of the inputs.")
        outputs = sample_iterator(outputs, sample_ratio, seed=seed)


    processed_length = 0
    processed_num = 0

    def _output_stats_calc_iter(_inputs):
        nonlocal processed_length
        nonlocal processed_num

        for item in _inputs:
            processed_length += item[content_word_length_key]
            processed_num += 1

            if keep_metadata:
                yield item
            else:
                # only keep the content and its word length
                yield {content_key: item[content_key], content_word_length_key: item[content_word_length_key]}
    
    outputs = _output_stats_calc_iter(outputs)
    write_jsonl(outputs, output_shard_path)
    print(f"Shard {shard_idx}: Processed shard {shard_path} with {processed_num}/{total_num} samples and {processed_length}/{total_length} words.")

    return DatasetStats(num_samples=processed_num, content_word_length=processed_length, src_num_samples=total_num, src_content_word_length=total_length)


def split_shard_file(shard_idx, shard_path, output_shard_path, split_slice, continuous_split=False, split_val_set=False):
    # copy the lines from the start to the end
    if shard_path.endswith(VAL_DATA_SUFFIX) and not split_val_set:
        start, end, step = 0, 1, 1
    else:
        start, end, step = map(int, split_slice.split(':'))
        if not (0 <= start < end <= step):
            raise ValueError(f"Invalid split slice format. Must satisfy 0 <= start < end <= step. Got {split_slice}")
    
    src_num_samples = 0
    src_num_content_words = 0
    num_samples = 0
    num_content_words = 0

    with open(output_shard_path, "w") as f:
        for idx, sample in enumerate(read_jsonl(shard_path)):
            src_num_samples += 1
            content_word_length = sample.get("text_word_length", 0) or sample.get("text_length", 0)
            src_num_content_words += content_word_length
            
            if not continuous_split:
                # If the line index (modulo step) falls within our desired range, write it
                if start <= (idx % step) < end:
                    f.write(json.dumps(sample) + "\n")
                    num_samples += 1
                    num_content_words += content_word_length

        if continuous_split:
            idx_start = src_num_samples * start // step
            idx_end = src_num_samples * end // step
            # Need to reopen the file for the second pass
            for idx, sample in enumerate(read_jsonl(shard_path)):
                if idx < idx_start:
                    continue
                if idx >= idx_end:
                    break
                    
                f.write(json.dumps(sample) + "\n")
                num_samples += 1
                content_word_length = sample.get("text_word_length", 0) or sample.get("text_length", 0)
                num_content_words += content_word_length

    return DatasetStats(num_samples=num_samples, content_word_length=num_content_words, src_num_samples=src_num_samples, src_content_word_length=src_num_content_words)


def main(
        mode, data_config_dir, src_dataset_name, tgt_dataset_name=None, tgt_dataset_basedir=None,
        num_chunks=32, num_val_samples=10000,  memory=8, seed=42, 
        split_slice=None, continuous_split=False,
        sample_ratio=1.0, min_length=0, max_length=np.inf, keep_metadata=False,
        num_processes=1, tmp_dir=None, overwrite=False
    ):
    src_dataset_config = load_dataset_config(data_config_dir, src_dataset_name)
    if tgt_dataset_name is None:
        tgt_dataset_name = src_dataset_name + f"_{mode}"

    if tgt_dataset_basedir is None:
        tgt_dataset_basedir = os.path.dirname(src_dataset_config.path)
    tgt_dataset_dir = os.path.join(tgt_dataset_basedir, tgt_dataset_name)

    if os.path.exists(tgt_dataset_dir):
        if not overwrite:
            print(f"Target dataset {tgt_dataset_dir} already exists. Set --overwrite to overwrite.")
            return
        else:
            print(f"Target dataset {tgt_dataset_dir} already exists. Overwriting...")
            shutil.rmtree(tgt_dataset_dir)
    os.makedirs(tgt_dataset_dir, exist_ok=True)

    if tmp_dir is not None:
        work_dir = tmp_dir
        tmp_dir = os.path.join(tmp_dir, str(uuid.uuid4()))
        os.makedirs(tmp_dir, exist_ok=True)
    else:
        work_dir = tmp_dir = os.environ["TMPDIR"]
    os.environ["TMPDIR"] = tmp_dir

    if "fineweb" in src_dataset_name or "open_web_math" in src_dataset_name or "finemath" in src_dataset_name:
        orig_extension = ".jsonl"
        cat_command = "cat"
    elif "dclm" in src_dataset_name:
        orig_extension = ".jsonl.zst"
        cat_command = "zstdcat"
    else:
        raise ValueError(f"Unknown dataset: {src_dataset_name}")

    if mode == "final_shuffled":
        assert not src_dataset_config.format == DatasetFormat.FINAL, "Should not apply final shuffling on final shuffled dataset."

        # only for saving here
        process_kwargs = {
            "num_val_samples": num_val_samples,
            "num_chunks": num_chunks,
            "seed": seed,
        }

        prefix = f"{tgt_dataset_name}.chunk."

        
        k_validation = num_val_samples // num_chunks

        # Setup terashuf
        terashuf_dir = setup_terashuf(work_dir)

        # Set up environment variables
        os.environ["MEMORY"] = f"{memory}"
        os.environ["SEED"] = f"{seed}"

        # Run the original shuffling and splitting command
        if num_chunks > 1:
            print(f"WARNING: using num_chunks = {num_chunks} now, please make sure to use the world size >= {num_chunks}, otherwise some chunks will be discarded.")

        terashuf_executable = os.path.join(terashuf_dir, "terashuf")
        run_command(
            f"ulimit -n 100000 && "
            f"find -L {src_dataset_config.path} -type f -name '*{orig_extension}' -print0 | xargs -0 {cat_command} | {terashuf_executable} | "
            f"split -n r/{num_chunks} -d --suffix-length 2 --additional-suffix {DATA_SUFFIX} - {tgt_dataset_dir}/{prefix}"
            # "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
        )

        # Create validation set and remove lines from chunks
        validation_file = f"{tgt_dataset_dir}/{tgt_dataset_name}{VAL_DATA_SUFFIX}"
        for i in range(num_chunks):
            chunk_file = f"{tgt_dataset_dir}/{prefix}{i:02d}{DATA_SUFFIX}"
            run_command(f"sed -i 's/}}{{\"bff/}}\\\n{{\"bff/g; s/}}{{\"text/}}\\\n{{\"text/g' {chunk_file}")  # fix json format
            if k_validation > 0:    
                run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
                run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

        processed_dataset_stats = DatasetStats(
            num_samples=src_dataset_config.stats["num_samples"],  # not changed, but should minus the validation set
            content_word_length=src_dataset_config.stats["content_word_length"],  # not changed, but should minus the validation set
            src_num_samples=src_dataset_config.stats["num_samples"],  
            src_content_word_length=src_dataset_config.stats["content_word_length"], 
        )
        processed_format = DatasetFormat.FINAL
    elif mode in ["sampled_or_filtered", "subset_split"]:
        if mode == "sampled_or_filtered":
            assert not src_dataset_config.format == DatasetFormat.FINAL, "Should not apply sampling or filtering on final shuffled dataset."
            process_func = process_shard_file
            process_kwargs = {
                "content_key": "text",
                "sample_ratio": sample_ratio,
                "min_length": min_length,
                "max_length": max_length,
                "keep_metadata": keep_metadata,
                "seed": seed,
            }
            processed_format = DatasetFormat.PROCESSED
            shard_file_extension = orig_extension
        elif mode == "subset_split":
            assert src_dataset_config.format == DatasetFormat.FINAL, "Split should be applied on final shuffled dataset."
            process_func = split_shard_file
            process_kwargs = {
                "split_slice": split_slice,
                "continuous_split": continuous_split,
            }
            processed_format = DatasetFormat.FINAL
            shard_file_extension = DATA_SUFFIX
        
        input_shard_paths = []
        output_shard_paths = []
        for shard_file_path in sorted(os.listdir(src_dataset_config.path)):
            if shard_file_path.endswith(shard_file_extension):
                input_shard_paths.append(os.path.join(src_dataset_config.path, shard_file_path))
                output_shard_paths.append(os.path.join(tgt_dataset_dir, shard_file_path))

        process_shard = partial(process_func, **process_kwargs)
        with Pool(num_processes) as pool:
            shard_stats = pool.starmap(
                process_shard, 
                [(idx, input_path, output_path) 
                 for idx, (input_path, output_path) in enumerate(zip(input_shard_paths, output_shard_paths))]
            )

        processed_dataset_stats = merge_shard_stats(shard_stats)
        print(f"Processed {processed_dataset_stats.num_samples}/{processed_dataset_stats.src_num_samples} samples and {processed_dataset_stats.content_word_length}/{processed_dataset_stats.src_content_word_length} words.")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    

    print("All tasks completed successfully!")

    tgt_dataset_config = DatasetConfig(
        uuid=str(uuid.uuid4()),
        name=tgt_dataset_name,
        path=tgt_dataset_dir,
        sources=[src_dataset_config.name],
        format=processed_format,
        stats=processed_dataset_stats,
        process_kwargs=process_kwargs,
    )

    save_dataset_config(tgt_dataset_config, data_config_dir, tgt_dataset_name)

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--data-config-dir", type=str, default="configs/dataset")
    parser.add_argument("--src-dataset-name", type=str, required=True)
    parser.add_argument("--tgt-dataset-name", type=str, required=False, default=None)
    parser.add_argument("--tgt-dataset-basedir", type=str, required=False, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num-val-samples", type=int, default=0)
    parser.add_argument("--split-slice", type=str, default=None, help="The split slice of the dataset in format start:end:step, e.g., 1:3:10 means for every 10 lines, keep lines 1-2")
    parser.add_argument("--continuous-split", action="store_true", default=False, help="Whether to split the dataset continuously.")
    parser.add_argument("--memory", type=float, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp-dir", type=str, required=False, default=None)
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="The ratio of samples to be sampled from the original dataset.")            
    parser.add_argument("--min-length", type=int, default=0, help="Minimum length of the text to be generated.")
    parser.add_argument("--max-length", type=int, default=np.inf, help="Maximum length of the text to be generated.")
    parser.add_argument("--keep-metadata", action="store_true", default=False, help="Whether to keep the metadata in the processed dataset.")
    parser.add_argument("--num-processes", type=int, default=1, help="The number of processes to use for generation.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to overwrite the target dataset if it already exists.")

    # TODO: save only meta data

    args = parser.parse_args()

    main(
        args.mode, args.data_config_dir, args.src_dataset_name, 
        tgt_dataset_name=args.tgt_dataset_name, tgt_dataset_basedir=args.tgt_dataset_basedir,
        num_chunks=args.num_chunks, num_val_samples=args.num_val_samples, keep_metadata=args.keep_metadata,
        memory=args.memory, seed=args.seed, 
        split_slice=args.split_slice, continuous_split=args.continuous_split, 
        sample_ratio=args.sample_ratio, min_length=args.min_length, max_length=args.max_length, 
        num_processes=args.num_processes, tmp_dir=args.tmp_dir, overwrite=args.overwrite
    )
