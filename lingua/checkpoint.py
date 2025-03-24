# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.format_utils import (
    torch_save_to_dcp,
    dcp_to_torch_save,
)
import torch.optim.optimizer

from lingua.distributed import get_is_master

logger = logging.getLogger("CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

LINGUA_CONFIG_NAME = "params_lingua.json"
CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")


@dataclass
class SaveEvery:
    every: int = 1000
    keep: int = 0


@dataclass
class CheckpointArgs:
    dump: SaveEvery = field(default_factory=SaveEvery)
    eval: SaveEvery = field(default_factory=SaveEvery)
    keep_ckpt_steps: List[int] = field(default_factory=list)
    must_eval_steps: List[int] = field(default_factory=list)
    path: Optional[str] = None
    init_ckpt_path: Optional[str] = None
    load_optim_from_init_ckpt: bool = False  # if true, will also load optimizer state from init ckptsssss
    continue_training_from_init: bool = False


def _get_key_step(name: str):
    return int(re.findall(RE_DIGITS, name)[-1])


def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME / LINGUA_CONFIG_NAME).exists():
        consolidate_path.mkdir(exist_ok=True)
        logger.info(f"Consolidating to: {str(consolidate_path)}")
        dcp_to_torch_save(ckpt_dir, str(consolidate_path / CONSOLIDATE_NAME))
        (consolidate_path / CONFIG_NAME).write_text(
            (Path(ckpt_dir) / CONFIG_NAME).read_text()
        )
        (consolidate_path / LINGUA_CONFIG_NAME).write_text(
            (Path(ckpt_dir) / LINGUA_CONFIG_NAME).read_text()
        )
        logger.info("Consolidated !")
    return consolidate_path



def load_from_checkpoint(ckpt_dir: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, model_key: str = "model", optim_key: str = "optim", raise_on_embd_mismatch: bool = True):
    """
    See unmerged lingua PR: https://github.com/facebookresearch/lingua/pull/47
    """
    if not (Path(ckpt_dir) / '.metadata').exists():
        raise ValueError(f"Please convert the checkpoint distcp format using `torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it")

    state_dict = {}
    if optimizer is not None:
        state_dict[model_key], state_dict[optim_key] = get_state_dict(model, optimizer)
    else:
        state_dict[model_key] = get_model_state_dict(model)
        if model_key == "": # If only loading a model directly, the key should be empty
            state_dict = state_dict.pop(model_key)

    # Add shape validation before loading
    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()

    def _clean_name(name: str):
        # Strip '_orig_mod.' prefix if present in the model parameter name
        return name.replace('_orig_mod.', '').replace('model.', '')

    metadata_state_dict = {_clean_name(name): data for name, data in metadata.state_dict_metadata.items()}
    for name, param in model.named_parameters():
        clean_name = _clean_name(name)
        assert clean_name in metadata_state_dict.keys(), (clean_name, metadata_state_dict.keys())
        saved_shape = metadata_state_dict[clean_name].size
        
        # Allow embedding layers to be resized (only first dimension can change)
        if param.shape != tuple(saved_shape):
            if clean_name in ['tok_embeddings.weight', 'output.weight']:
                if raise_on_embd_mismatch:
                    raise ValueError(f"Invalid embedding resize for {clean_name}: "
                                    f"checkpoint has shape {saved_shape}, "
                                    f"but model expects shape {param.shape}. ")
                else:
                    if param.shape[1:] != tuple(saved_shape)[1:]:
                        raise ValueError(f"Invalid embedding resize for {clean_name}: "
                                    f"checkpoint has shape {saved_shape}, "
                                    f"but model expects shape {param.shape}. "
                                    f"Only the vocabulary dimension can be resized.")

                    # for some reason, the dcp.load does not raise an error here, but probably not a safe operation
                    logger.warning(f"Resizing embedding {clean_name} from {saved_shape} to {param.shape}")
            else:
                raise ValueError(f"Shape mismatch for parameter {name} (cleaned: {clean_name}): "
                            f"checkpoint has shape {saved_shape}, "
                            f"but model expects shape {param.shape}")
    
    dcp.load(state_dict, checkpoint_id=ckpt_dir)


class CheckpointManager:
    def __init__(self, args: CheckpointArgs):
        self.path = args.path
        self.dump_every = args.dump
        self.eval_every = args.eval
        self.init_ckpt_path = args.init_ckpt_path
        self.keep_ckpt_steps = args.keep_ckpt_steps
        self.continue_training_from_init = args.continue_training_from_init

        os.makedirs(self.path, exist_ok=True)

        self.existing_saves = self.get_existing_saves()

    def get_existing_saves(self) -> List[Path]:
        folders = [
            p
            for p in Path(self.path).iterdir()
            if p.is_dir() and re.match(RE_FOLDER, p.name)
        ]
        folders.sort(key=lambda p: _get_key_step(p.name))
        return folders

    def clean_up(self):
        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_complete_folders = []
        eval_incomplete_folders = []
        all_eval_folders = []
        other_folders = []
        keep_folders = []
        for p in self.existing_saves:
            step = _get_key_step(p.name)
            is_dump = step % self.dump_every.every == 0
            is_eval = step % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval: # wait for evals to complete before removing them!
                all_eval_folders.append(p)
                if (p / "eval.complete").exists():
                    eval_complete_folders.append(p)
                else:
                    eval_incomplete_folders.append(p)

            if not (is_dump or is_eval):
                other_folders.append(p)
            if step in self.keep_ckpt_steps:
                keep_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval complete folders: {eval_complete_folders}")
        logger.info(f"Eval incomplete folders: {eval_incomplete_folders}")
        logger.info(f"Keep folders: {keep_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders_to_keep = set(all_eval_folders[-self.eval_every.keep:])
            eval_keep_with_incompletes = set(eval_folders_to_keep) | set(eval_incomplete_folders)
            if not eval_keep_with_incompletes.issubset(eval_folders_to_keep): 
                diff = eval_keep_with_incompletes - eval_folders_to_keep
                logger.warning(f"WARNING: Attempted to clean up eval folders, but some are not yet complete. Disk usage may be higher than expected. Incomplete folders: {diff}")
        else:
            eval_keep_with_incompletes = set(all_eval_folders)

        folder_to_keep = set(other_folders + dump_folders + keep_folders) | eval_keep_with_incompletes
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if dist.get_rank() == 0:
            for folder in folder_to_remove:
                def remove_dir_recursive(path):
                    for file in path.iterdir():
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            remove_dir_recursive(file)
                    path.rmdir()
                
                remove_dir_recursive(folder)
                # folder.rmdir()

        dist.barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(p.name))

    def get_last_step_path(self, dp_rank: int = 0) -> Optional[Path]:
        path = None
        for p in reversed(self.existing_saves):
            if (p / TRAIN_STATE_NAME.format(dp_rank)).is_file():
                path = p
                break
        return path

    def _create_folder(self, base_path: Path, folder_name: str) -> Path:
        folder = base_path / folder_name
        if get_is_master():
            folder.mkdir(parents=False, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return folder

    def _get_dp_tp_mesh(
        self, device_mesh: Optional[DeviceMesh] = None
    ) -> Tuple[int, int]:
        dp_rank = 0
        tp_rank = 0
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                dp_rank = device_mesh.get_local_rank("dp_replicate")
                if "dp_shard" in device_mesh.mesh_dim_names:
                    dp_rank = dp_rank * device_mesh["dp_replicate"].size() + device_mesh.get_local_rank("dp_shard")
            if "tp" in device_mesh.mesh_dim_names:
                tp_rank = device_mesh.get_local_rank("tp")
        return dp_rank, tp_rank

    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
    ):
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def save(
        self,
        model,
        optimizer,
        train_state,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> bool:

        # When creating directory check if only rank0 or is there other solution
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving to: {str(curr_save_dir)}")

        if dist.is_initialized():
            dist.barrier()

        logger.info("Saving...")
        state_dict = self.get_state_dict(model, optimizer)
        dcp.save(state_dict, checkpoint_id=curr_save_dir)
        logger.info("State dict saved!")

        if dist.is_initialized():
            dist.barrier()

        if get_is_master():
            config = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
            with open(curr_save_dir / LINGUA_CONFIG_NAME, "w") as f:
                json.dump(
                    config,
                    f,
                    indent=4,
                )

            # model params config compatible with huggingface
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    config["model"],
                    f,
                    indent=4,
                )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            logger.info(
                f"Saving train state to: {str(curr_save_dir / train_state_name)}"
            )
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        self.existing_saves.append(curr_save_dir)

        self.clean_up()

        if dist.is_initialized():
            dist.barrier()
        return True

    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: Optional[Path] = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        logger.info("Reloading train state")
        with open(path / train_state_name, "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        dcp.load(state_dict, checkpoint_id=path)
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")
