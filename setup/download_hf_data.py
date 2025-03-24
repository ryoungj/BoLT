
import argparse
import os
import time
import subprocess
import requests
import json
from huggingface_hub import snapshot_download


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16, # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    if dataset == "open_web_math" or "finemath" in dataset:
        def parse_metadata(self, data, path, id_in_file):
            return {
                "text": data.pop("text", ""),
                "media": data.pop("media", []),
                "id": f"{path}/{id_in_file}",
                "metadata": json.loads(data.pop("metadata")) | data,  # remaining data goes into metadata
                }
        
        adapter = parse_metadata
    else:
        adapter = None
    
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
                adapter=adapter,
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()

def main(dataset, data_dir):
    # Configuration
    repo_setups = {
        "fineweb_edu": ("HuggingFaceFW/fineweHb-edu", None),
        "fineweb_edu_10bt": ("HuggingFaceFW/fineweb-edu", "sample/10BT/*"),
        "dclm_baseline_1.0": ("mlfoundations/dclm-baseline-1.0", "*.jsonl.zst"),
        "dclm_baseline_1.0_10prct": ("mlfoundations/dclm-baseline-1.0", "global-shard_01_of_10/*.jsonl.zst"),
        "open_web_math": ("open-web-math/open-web-math", None),
        "finemath-3plus": ("HuggingFaceTB/finemath", "finemath-3plus/*.parquet"),
        "finemath-4plus": ("HuggingFaceTB/finemath", "finemath-4plus/*.parquet"),
        "finemath-infiwebmath-3plus": ("HuggingFaceTB/finemath", "infiwebmath-3plus/*.parquet"),
        "finemath-infiwebmath-4plus": ("HuggingFaceTB/finemath", "infiwebmath-4plus/*.parquet"),
    }
    
    repo_id, allow_patterns = repo_setups[dataset]
    src_dir = f"{data_dir}/{dataset}"
    # Download dataset
    download_dataset(repo_id, src_dir, allow_patterns)

    if "fineweb" in dataset or "open_web_math" in dataset or "finemath" in dataset:
        parquet_to_jsonl(dataset, src_dir, src_dir, os.path.join(src_dir, "parsed"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--datadir", type=str, default="data/processed")
    args = args.parse_args()
    main(args.dataset, args.datadir)
