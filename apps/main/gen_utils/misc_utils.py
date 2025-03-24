"""Adapted from https://github.com/mlfoundations/dclm/blob/main/baselines/core/file_utils.py

Some of which from @lxuechen's mlswissknife https://github.com/lxuechen/ml-swissknife/blob/e1866f94b89d7f813cfd24240dd5a5d42675a48b/ml_swissknife/utils.py#L232
"""

import os
import sys
import json
import zstandard as zstd
import gzip
import jsonlines
import io
from typing import BinaryIO, List
import logging
import numpy as np
import glob
import functools
import pathlib
from pathlib import Path as LocalPath
from typing import Union
import random
from typing import Optional, Sequence, Callable

PathOrIOBase = Union[str, pathlib.Path, io.IOBase]
makedirs = functools.partial(os.makedirs, exist_ok=True)


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.propagate = False
    
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def sample_iterator(iterator, ratio, seed=42, no_smaller=True):
    """Sample items from an iterator with a specified ratio.
    
    The actual sampling ratio may not be exactly the specified ratio, but we make sure the ratio is at least as large as the specified ratio with `no_smaller=True`.
    """
    def sampler():
        random.seed(seed)

        sampled_size = 0
        total_size = 0
        reservoir = []  # potentially yield items to maintain the ratio
        
        for item in iterator:
            total_size += 1
            if random.random() < ratio:
                sampled_size += 1
                yield item
            elif no_smaller:
                reservoir_size = max(int(total_size * ratio) - sampled_size, 0)

                if reservoir_size > 0:
                    if len(reservoir) < reservoir_size:
                        reservoir.append(item)
                    else:
                        replace_index = random.randint(0, total_size - sampled_size)
                        if replace_index < reservoir_size:
                            reservoir[replace_index] = item
        
        # If the ratio is not met, yield additional items from the reservoir
        while reservoir and sampled_size / total_size < ratio:
            sampled_size += 1
            yield reservoir.pop(0)
    
    return sampler()


def filter_iterator(iterator, filter_func):
    for item in iterator:
        if filter_func(item):
            yield item



def is_compressed(file_path: str):
    return any(file_path.endswith(z) for z in (".zst", ".zstd", ".gz"))


def _jsonl_bytes_reader(fh: BinaryIO):
    with io.TextIOWrapper(fh, encoding="utf-8") as text_reader:
        with jsonlines.Reader(text_reader) as jsonl_reader:
            for item in jsonl_reader:
                yield item

def read_jsonl(file_path: str):
    """Read a JSONL file from a given path (local or S3)."""
    path = LocalPath(file_path)

    if any(file_path.endswith(z) for z in (".zst", ".zstd")):
        with path.open('rb') as f:
            with zstd.ZstdDecompressor().stream_reader(f) as reader:
                for line in _jsonl_bytes_reader(reader):
                    yield line
    elif file_path.endswith(".gz"):
        with gzip.open(path, 'rb') as f:
            for line in _jsonl_bytes_reader(f):
                yield line
    else:
        with path.open('rb') as f:    
            for line in _jsonl_bytes_reader(f):
                yield line

def write_jsonl(data, file_path: str, mode: str = "w"):
    """Write data to a JSONL file at a given path (local or S3)."""
    path = LocalPath(file_path)

    if is_compressed(file_path):
        data = [json.dumps(d) for d in data]
        data = "\n".join(data).encode('utf8')

    if any(file_path.endswith(z) for z in (".zst", ".zstd")):
        with path.open("wb") as f:
            with zstd.ZstdCompressor().stream_writer(f) as writer:
                writer.write(data)
    elif file_path.endswith(".gz"):
        with path.open("wb") as f:
            f.write(gzip.compress(data))
    else:
        with path.open(mode) as f:
            for item in data:
                json_str = json.dumps(item)
                f.write(f"{json_str}\n")


def glob_files(path, suffixes, verbose=False):
    """
    Glob files based on a given path and suffix.

    :param path: path to glob. Can be local or S3 (e.g., s3://bucket-name/path/)
    :param suffix: suffix of files to match. Defaults to ".jsonl"
    :return: list of file paths matching the pattern
    """
    # Use glob for local paths
    matching_files = []
    for suffix in suffixes:
        search_pattern = f"{path.rstrip('/')}/**/*{suffix}"
        matching_files.extend(glob.glob(search_pattern, recursive=True))
        if verbose:
            print("matching files with suffix: ", suffix)
            print(matching_files)

    return matching_files


def parse_json_file(file_path):
    documents = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                # text = json_obj['text']
                # documents.append(text)
                documents.append(json_obj)
            except (json.JSONDecodeError, KeyError) as e:
                logging.info(f"Error processing line: {e}")
    return documents


def _make_w_io_base(f: PathOrIOBase, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            makedirs(f_dirname)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f: PathOrIOBase, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(
    obj: Union[str, dict, list], f: PathOrIOBase, mode="w", indent=4, default=str
):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def jload(f: PathOrIOBase, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jldump(seq: Sequence[dict], f: PathOrIOBase, mode="w", indent=4, default=str):
    """Dump a sequence of dictionaries into a .jsonl file."""
    if not all(isinstance(item, dict) for item in seq):
        raise ValueError("Input is not of type Sequence[dict].")
    f = _make_w_io_base(f, mode)
    for item in seq:
        f.write(json.dumps(item, indent=indent, default=default) + "\n")
    f.close()


def readlines(f: Union[str, pathlib.Path, io.IOBase], mode="r", strip=True):
    f = _make_r_io_base(f, mode)
    lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    f.close()
    return lines

def jlload(f: PathOrIOBase, mode="r", strip=True):
    """Load a .jsonl file into a list of dictionaries."""
    return [json.loads(line) for line in readlines(f, mode=mode, strip=strip)]


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.
    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.
    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)