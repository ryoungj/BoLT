"""
Generate latents for a given dataset.
"""

import os
import shutil
import logging
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict, fields, field
from lingua.logger import init_logger
from typing import List, Dict, Tuple, Any, Optional, Callable, Union
from functools import partial
from multiprocessing import Pool
from itertools import islice, chain
import traceback
import numpy as np
from tqdm import tqdm
from collections import defaultdict


from lingua.args import dataclass_from_dict
from apps.main.prepare_data import DatasetConfig, DatasetFormat, DatasetStats, load_dataset_config, save_dataset_config, DATA_SUFFIX, VAL_DATA_SUFFIX
from apps.main.latent_bootstrap import BootstrapLatentsArgs, bootstrap_latents
from apps.main.gen_utils.misc_utils import read_jsonl, write_jsonl
from apps.main.gen_utils import *

logger = logging.getLogger()


ALL_LATENT_GENERATION_FUNCS = {
    **ALL_GENERATION_FUNC,
    "self_bootstrap": bootstrap_latents,
}

@dataclass
class GenLatentsArgs:
    src_dataset_name: str = "dclm_baseline_1.0"
    tgt_dataset_base_dir: Optional[str] = None
    tgt_dataset_name: str = "dclm_baseline_1.0_with_latents"
    dataset_config_dir: str = "configs/dataset"
    proc_val_set: bool = False  # whether to process the validation set
    tgt_num_shards: Optional[int] = None  # the number of output shards (excluding the validation set), if None, the number of shards will be the same as the source dataset
    # Data slicing and multi-processing are mutually exclusive because:
    # - Data slicing is for distributing work across separate jobs
    # - Multi-processing is for parallel processing within a single job
    data_slice: Optional[str] = None  # the slice of the dataset to be processed, useful when launching multiple jobs in parallel, format: start:end:slice, e.g., 1:3:10 means split dataset into 10 slices and process the 1 and 2 slices
    num_processes: int = 1  # the number of processes to use for data processing
    merge_shards: bool = True  # whether to merge the shards after processing
    overwrite: bool = False  # whether to overwrite the existing target dataset
    load_cache_results: bool = True  # whether to load cached results
    clean_cache: bool = False  # whether to clean up the cache directory after successful processing
    limit: Optional[Union[int, float]] = None  # the maximum number of samples or ratio of samples to process per workload, useful for debugging
    exec_seed: int = 42  # the seed to use for execution
    num_repeats: int = 1  # the number of times to repeat the generation for each sample
    reshuffle_repeats: bool = True  # reshuffle the repeats for each pass
    shuffle_seed: int = 42  # the seed to use for reshuffling the repeats

    chunk_seed: int = 42  # the seed to use for chunking the data
    chunk_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)   # how to chunk the data
    generation_method: str = "synthetic"
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)  # how to generate latents

    log_level: str = "INFO"


@dataclass
class Workload:
    global_workload_idx: int
    global_shard_idx: int
    local_num_slices: int
    local_slice_offset: int
    shard_file: str
    num_repeats: int = 1
    repeat_idx: int = 0

    def __str__(self):
        return f"Workload(global_workload_idx={self.global_workload_idx}, global_shard_idx={self.global_shard_idx}, local_num_slices={self.local_num_slices}, local_slice_offset={self.local_slice_offset}, num_repeats={self.num_repeats}, repeat_idx={self.repeat_idx}, shard_file={self.shard_file})"

    @property
    def result_file_suffix(self):
        suffix = f"_shard{self.global_shard_idx:02d}_slice{self.local_slice_offset:02d}_of_{self.local_num_slices:02d}"

        if self.num_repeats > 1:
            suffix += f"_repeat{self.repeat_idx:02d}_of_{self.num_repeats:02d}"
        return suffix


@dataclass
class GenLatentsStats(DatasetStats):
    total_text_token_length: int = 0
    total_latent_token_length: int = 0
    total_combined_text_latent_token_length: int = 0
    avg_text_token_length: float = 0
    avg_latent_token_length: float = 0
    avg_combined_text_latent_token_length: float = 0


@dataclass
class ExecResult:
    workload: Workload
    success: bool
    cache_output_path: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    error_msg: Optional[str] = None


def get_workloads(shard_files: List[str], data_slice: Optional[str] = None, num_repeats: int = 1) -> List[Workload]:
    """Get the workload of processing data shards."""
    num_total_shards = len(shard_files)

    if data_slice is None:
        start, end, num_total_slices = 0, num_total_shards, num_total_shards
    else:
        start, end, num_total_slices = map(int, data_slice.split(':'))
        assert 0 <= start < end <= num_total_slices, "Invalid data slice"

    workloads = []
    if num_total_slices > num_total_shards:
        assert num_total_slices % num_total_shards == 0, "The number of slices must be a multiple of the number of shards for uniform slicing."
        num_slices_per_shard = num_total_slices // num_total_shards   # e.g., 20 slices, 5 shards -> 4 slices per shard
        
        for repeat_idx in range(num_repeats):  # repeat the workload 
            global_workload_start_idx = repeat_idx * num_total_slices  # repeat after each pass of the entire dataset
            for i in range(start, end):
                workloads.append(
                    Workload(
                        global_workload_idx=global_workload_start_idx + i,
                        global_shard_idx=i // num_slices_per_shard, 
                        shard_file=shard_files[i // num_slices_per_shard],
                        local_num_slices=num_slices_per_shard, 
                        local_slice_offset=i % num_slices_per_shard,
                        num_repeats=num_repeats,
                        repeat_idx=repeat_idx,
                    )
                )
    else:
        assert num_total_shards % num_total_slices == 0, "The number of shards must be a multiple of the number of slices for uniform slicing."
        num_shards_per_slice = num_total_shards // num_total_slices  # e.g., 20 shards, 5 slices -> 4 shards per slice

        for repeat_idx in range(num_repeats):  # repeat the workload 
            global_workload_start_idx = repeat_idx * num_total_slices  # repeat after each pass of the entire dataset
            for i in range(start, end):
                workloads.extend([
                Workload(
                    global_workload_idx=global_workload_start_idx + i * num_shards_per_slice + j,
                    global_shard_idx=i * num_shards_per_slice + j, 
                    shard_file=shard_files[i * num_shards_per_slice + j], 
                    local_num_slices=1, 
                    local_slice_offset=0,
                    num_repeats=num_repeats,
                    repeat_idx=repeat_idx,
                )
                for j in range(num_shards_per_slice)
            ])
    return workloads


def distribute_workloads(shard_files: List[str], num_processes: int = 1, num_repeats: int = 1) -> List[List[Workload]]:
    """Distribute the workload of processing data slices across multiple processes."""
    dist_workloads = []
    assert num_processes % num_repeats == 0, "The number of processes must be a multiple of the number of repeats"
    num_data_slices = num_processes // num_repeats

    for i in range(num_data_slices):
        workloads = get_workloads(shard_files, data_slice=f"{i}:{i+1}:{num_data_slices}", num_repeats=num_repeats)
        dist_workloads.extend(workloads)
    
    num_workloads_per_process = (len(dist_workloads) + num_processes - 1) // num_processes
    dist_workloads = [dist_workloads[i * num_workloads_per_process:(i + 1) * num_workloads_per_process] for i in range(num_processes)]
    return dist_workloads


def execute_workloads(workloads: List[Workload], process_func: Callable, input_dir: str, cache_dir: str, limit: Optional[Union[int, float]] = None, chunk_seed: int = 42, chunk_kwargs: Dict[str, Any] = None, exec_seed: int = 42) -> List[ExecResult]:
    # FIXME: have to put here because the chunked_generation_wrapper is not picklable
    if chunk_kwargs is not None:
        chunk_kwargs["chunk_seed"] = chunk_seed  # update the chunk seed
        process_func = chunked_generation_wrapper(process_func, **chunk_kwargs)

    exec_results = []
    try:
        for workload in workloads:
            shard_file_path = os.path.join(input_dir, workload.shard_file)
            num_lines = sum(1 for _ in read_jsonl(shard_file_path))
            num_lines_per_slice = (num_lines + workload.local_num_slices - 1) // workload.local_num_slices
            start_line_idx = workload.local_slice_offset * num_lines_per_slice

            if limit is not None:
                if limit < 1:
                    limit = int(limit * num_lines)
                end_line_idx = min(start_line_idx + limit, num_lines)
            else:
                end_line_idx = num_lines if workload.local_slice_offset == workload.local_num_slices - 1 else (workload.local_slice_offset + 1) * num_lines_per_slice
            total_num_samples = end_line_idx - start_line_idx
            shard_slice_iter = islice(read_jsonl(shard_file_path), start_line_idx, end_line_idx)

            output_path = os.path.join(cache_dir, f"result{workload.result_file_suffix}.jsonl")

            shard_slice_iter = tqdm(shard_slice_iter, desc=f"Processing work {workload.global_workload_idx}", position=workload.global_workload_idx)

            outputs, proc_stats = process_func(shard_slice_iter, workload.global_workload_idx, seed=exec_seed + workload.global_workload_idx)

            write_jsonl(outputs, output_path)
            assert len(outputs) == total_num_samples, f"The number of outputs does not match the number of samples ({len(outputs)} vs {total_num_samples})"
            
            # post process
            proc_stats.pop("total_num_samples", None)
            proc_stats = {k.replace("_length", "_token_length"): v for k, v in proc_stats.items()}
            proc_stats.pop("compression_ratio", None)
            stats = GenLatentsStats(
                num_samples=total_num_samples,
                src_num_samples=total_num_samples,
                **proc_stats,
            )

            logger.info(f"Finished processing workload {workload.global_workload_idx}, stats: {stats}")
            exec_results.append(ExecResult(workload=workload, success=True, cache_output_path=output_path, stats=asdict(stats)))

        return exec_results
    except Exception as e:
        traceback.print_exc()   
        logger.error(f"Error executing workload {workload}: {str(e)}")
        exec_results.append(ExecResult(workload=workload, success=False, error_msg=str(e)))
        return exec_results


def merge_workload_stats(workload_stats: List[GenLatentsStats]) -> GenLatentsStats:
    keys = set(field.name for field in fields(GenLatentsStats))

    merged_stats = {}
    for key in keys:
        if key.startswith("avg_"):
            merge_func = np.mean
        else:
            merge_func = np.sum

        value = merge_func([stats.get(key, GenLatentsStats.__dataclass_fields__[key].default) for stats in workload_stats])
        merged_stats[key] = value.item() if isinstance(value, np.number) else value

    return GenLatentsStats(**merged_stats)


def merge_workload_results(workload_results: List[ExecResult], output_dir: str, shard_file_prefix: str, num_shards: int, num_repeats: int = 1, reshuffle_repeats: bool = True, shuffle_seed: int = 42):
    shard_to_workload_results = defaultdict(list)
    
    # filter out validation set first
    val_shard_file = shard_file_prefix + VAL_DATA_SUFFIX
    train_workload_results = []
    for result in workload_results:
        if result.workload.shard_file.endswith(VAL_DATA_SUFFIX):
            shard_to_workload_results[val_shard_file].append(result)
        else:
            train_workload_results.append(result)

    train_workload_results = sorted(train_workload_results, key=lambda x: x.workload.global_workload_idx)
    if num_repeats > 1 and reshuffle_repeats:
        assert len(train_workload_results) % num_repeats == 0, "The number of train workloads must be a multiple of the number of repeats, got {} workloads and {} repeats".format(len(train_workload_results), num_repeats)
        num_workloads_per_repeat = len(train_workload_results) // num_repeats
        
        shuffled_workload_results = []
        rng = np.random.default_rng(shuffle_seed)
        for i in range(num_repeats):
            repeat_workload_results = train_workload_results[i * num_workloads_per_repeat:(i + 1) * num_workloads_per_repeat]
            if i > 0:  # do not shuffle the first repeat
                rng.shuffle(repeat_workload_results)
            shuffled_workload_results.extend(repeat_workload_results)
        train_workload_results = shuffled_workload_results

    assert len(train_workload_results) % num_shards == 0, "The number of train workloads must be a multiple of the number of shards, got {} workloads and {} shards".format(len(train_workload_results), num_shards)
    num_workloads_per_shard = len(train_workload_results) // num_shards

    for i in range(num_shards):
        train_shard_file = shard_file_prefix + f".chunk.{i:02d}" + DATA_SUFFIX
        shard_to_workload_results[train_shard_file] = train_workload_results[i * num_workloads_per_shard:(i + 1) * num_workloads_per_shard]

    all_success = True
    succeeded_workload_stats = []
    for merged_shard_idx, (merged_shard_file, workload_results) in enumerate(sorted(shard_to_workload_results.items(), key=lambda x: x[0])):
        is_success = True
        if num_repeats == 1:
            # sanity check: without repeats, each output shard should correspond to the same input shards
            ref_workload = workload_results[0].workload
            attrs_match = lambda w1, w2: all(getattr(w1, a) == getattr(w2, a) for a in Workload.__dataclass_fields__.keys() if a not in ["local_slice_offset", "global_workload_idx"])
            assert all(attrs_match(ref_workload, result.workload) for result in workload_results), "All workloads must have matching attributes except for local_slice_offset"
            is_success &= (len(workload_results) == ref_workload.local_num_slices)

            if not is_success:
                 # print repetitive workloads
                local_slice_idx2workload = defaultdict(list)
                for r in workload_results:
                    local_slice_idx2workload[r.workload.local_slice_offset].append(r.workload)
                for local_slice_idx, workloads in local_slice_idx2workload.items():
                    if len(workloads) > 1:
                        raise RuntimeError(f"Repetitive workloads for local slice {local_slice_idx}: {workloads}. Please delete the repetitive entires and rerun the job.")


        is_success &= all(result.success for result in workload_results) 
        if is_success:
            # merge the results
            output_path = os.path.join(output_dir, merged_shard_file)
            
            output_paths = []
            output_iters = []
            for result in workload_results:
                output_paths.append(result.cache_output_path)
                output_iters.append(read_jsonl(result.cache_output_path))

            logger.info(f"Merging the following {len(output_iters)} results to {output_path}\n" + "\n".join([f"\t\t{p}" for p in output_paths]))
            
            
            write_jsonl(chain(*output_iters), output_path)   
            succeeded_workload_stats.extend([result.stats for result in workload_results if result.success])
        else:
            failed_workloads = [result.workload for result in workload_results if not result.success]
            logger.error(f"Merged shard {merged_shard_idx} ({merged_shard_file}) failed with workloads: {failed_workloads}")
            all_success = False

    agg_stats = merge_workload_stats(succeeded_workload_stats)
    return all_success, agg_stats


def main(args: GenLatentsArgs):
    # Preparation
    init_logger(level=args.log_level)
    src_dataset_config = load_dataset_config(args.dataset_config_dir, args.src_dataset_name)
    assert src_dataset_config.format == DatasetFormat.FINAL, "Source dataset must be in final format"
    
    if args.tgt_dataset_base_dir is None:
        args.tgt_dataset_base_dir = os.path.dirname(src_dataset_config.path)
    tgt_dataset_dir = os.path.join(args.tgt_dataset_base_dir, args.tgt_dataset_name)

    cache_dir = os.path.join(tgt_dataset_dir, "cache")
    cache_result_path = os.path.join(cache_dir, f"exec_results.jsonl")
    if os.path.exists(tgt_dataset_dir):
        if not args.overwrite:
            print(f"Target dataset {tgt_dataset_dir} already exists. Set --overwrite to overwrite.")
            return
        else:
            print(f"Target dataset {tgt_dataset_dir} already exists. Overwriting...")

            if not args.load_cache_results:
                # remove the whole target dataset including the cache files
                shutil.rmtree(tgt_dataset_dir)
            else:
                # do not remove the cache files
                for file in os.listdir(tgt_dataset_dir):
                    file_path = os.path.join(tgt_dataset_dir, file)
                    if os.path.isfile(file_path) and file.endswith("jsonl"):  # remove jsonl files
                        os.remove(file_path)
                    # if file != "cache":  # Compare string names directly
                    #     if os.path.isfile(file_path):
                    #         os.remove(file_path)
                    #     elif os.path.isdir(file_path):
                    #         shutil.rmtree(file_path)
    os.makedirs(tgt_dataset_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Collect data shards
    data_shard_files = []
    num_loaded_val_set = 0
    for shard_file in os.listdir(src_dataset_config.path):
        # shard_file = os.path.join(src_dataset_config.path, shard_file)
        if shard_file.endswith(VAL_DATA_SUFFIX):
            tgt_val_file = os.path.join(tgt_dataset_dir, shard_file)
            if args.proc_val_set:
                assert args.num_repeats == 1, "Repeating with validation set is not supported yet!"
                logger.info(f"The validation set {shard_file} will be processed.")
                data_shard_files.append(shard_file)
                num_loaded_val_set += 1
            else:
                logger.info(f"The validation set {shard_file} will not be processed and will be copied.")

                if not os. path.exists(tgt_val_file):
                    logger.info(f"Copying the validation file {shard_file} to {tgt_val_file}")
                    shutil.copy(os.path.join(src_dataset_config.path, shard_file), tgt_val_file)
                else:
                    logger.info(f"The validation file {tgt_val_file} already exists. Skip copying.")
        elif shard_file.endswith(DATA_SUFFIX):
            data_shard_files.append(shard_file)

    data_shard_files = sorted(data_shard_files)
    logger.info(f"Processing {len(data_shard_files)} data shards: {data_shard_files}")

    # Load cached results
    if args.load_cache_results and os.path.exists(cache_result_path):
        finished_exec_results = []
        finished_workloads = []
        for result in read_jsonl(cache_result_path):
            if result["success"]:
                workload = Workload(**result["workload"])
                result["workload"] = workload
                finished_exec_results.append(ExecResult(**result))
                finished_workloads.append(workload)
    else:
        finished_exec_results = []
        finished_workloads = []
    
    # Distribute workloads
    assert not (args.data_slice is not None and args.num_processes > 1), "Please use either data slicing (for multiple jobs) or multiple processes (for single job parallelization), not both."
    
    if args.data_slice is not None:
        # single process with data slicing
        workloads = get_workloads(data_shard_files, args.data_slice, num_repeats=args.num_repeats)
        dist_workloads = [workloads]
    elif args.num_processes > 1:
        # multiple processes
        dist_workloads = distribute_workloads(data_shard_files, args.num_processes, num_repeats=args.num_repeats)
    else:
        workloads = get_workloads(data_shard_files, num_repeats=args.num_repeats)
        dist_workloads = [workloads]

    running_dist_workloads = []
    for i, workloads in enumerate(dist_workloads):
        skipped_workloads = []
        running_workloads = []
        for j, w in enumerate(workloads):
            if w not in finished_workloads:
                running_workloads.append(w)
            else:
                skipped_workloads.append(w)
        work_load_str = f"Workload distribution for process {i}:"
        if len(skipped_workloads) > 0:
            work_load_str += f"\n\tSkipped workloads:\n\t\t" + "\n\t\t".join([str(w) for w in skipped_workloads])
        else:
            work_load_str += "\n\tSkipped workloads: None"
        
        if len(running_workloads) > 0:
            work_load_str += f"\n\tRunning workloads:\n\t\t" + "\n\t\t".join([str(w) for w in running_workloads])
        else:
            work_load_str += "\n\tRunning workloads: None"
        logger.info(work_load_str)
        running_dist_workloads.append(running_workloads)


    if args.generation_method == "self_bootstrap":
        args.generation_kwargs = {"cfg": dataclass_from_dict(BootstrapLatentsArgs, args.generation_kwargs)}

    if args.generation_method in ["synthetic"]:
        args.generation_kwargs["tmp_dir"] = cache_dir    

    logger.info(f"Using generation method {args.generation_method} with kwargs {args.generation_kwargs}")

    # exit(0)  # debug
    
    process_func = partial(ALL_LATENT_GENERATION_FUNCS[args.generation_method], **args.generation_kwargs)   
    exec_func = partial(execute_workloads, process_func=process_func, input_dir=src_dataset_config.path, cache_dir=cache_dir, limit=args.limit, chunk_seed=args.chunk_seed, chunk_kwargs=args.chunk_kwargs, exec_seed=args.exec_seed)
    
    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            running_dist_exec_results = pool.map(exec_func, running_dist_workloads)
    else:
        running_dist_exec_results = [exec_func(workloads) for workloads in running_dist_workloads]
    running_exec_results = [r for exec_results in running_dist_exec_results for r in exec_results]

    finished_exec_results.extend(running_exec_results)
    write_jsonl([asdict(r) for r in running_exec_results], cache_result_path, mode="a")

    if args.merge_shards:
        num_workload_shard_files = len(set([w.shard_file for w in chain(*dist_workloads)]))
        if args.tgt_num_shards is None:
            args.tgt_num_shards = num_workload_shard_files - num_loaded_val_set
        else:
            assert args.tgt_num_shards <= num_workload_shard_files - num_loaded_val_set, "Only support merging shards with less number of shards than the number of data shards, got {} workload shards and {} data shards".format(num_workload_shard_files, len(data_shard_files))
        
        all_success, agg_stats = merge_workload_results(finished_exec_results, tgt_dataset_dir, shard_file_prefix=args.tgt_dataset_name, num_shards=args.tgt_num_shards, num_repeats=args.num_repeats, reshuffle_repeats=args.reshuffle_repeats, shuffle_seed=args.shuffle_seed)
        if all_success:
            logger.info("All shards processed successfully.")
            if args.clean_cache:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleaned up cache directory {cache_dir}")
            
            tgt_dataset_config = DatasetConfig(
                uuid=src_dataset_config.uuid,
                name=args.tgt_dataset_name,
                path=tgt_dataset_dir,
                sources=[src_dataset_config.name],
                format=DatasetFormat.FINAL,
                stats=agg_stats,
                process_kwargs=asdict(args),
            )
            save_dataset_config(tgt_dataset_config, args.dataset_config_dir, args.tgt_dataset_name)
        else:
            logger.warning("Some shards failed to be processed. Please check the error messages and re-run the script.")
    else:
        failed_workloads = [w.success for w in finished_exec_results if not w.success]
        if len(failed_workloads) > 0:
            logger.warning(f"Some workloads failed to be processed: {failed_workloads}")
        else:
            logger.info("All workloads processed successfully.")


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(GenLatentsArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    main(cfg)
