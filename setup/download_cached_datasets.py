import os
from argparse import ArgumentParser
from datasets import load_dataset


parser = ArgumentParser()
parser.add_argument("--subsets", type=str, nargs="+", required=True)
parser.add_argument("--dump-dir", type=str, required=True)
args = parser.parse_args()

dataset_repo_name = "ryoungj/bootstrap-latent-thought-data"
for subset in args.subsets:
    print(f"Downloading subset '{subset}'...")
    dataset = load_dataset(dataset_repo_name, subset, split="train")
    dataset_dir = os.path.join(args.dump_dir, subset)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset.to_json(os.path.join(dataset_dir, f"{subset}.chunk.00.jsonl"))

