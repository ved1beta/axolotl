"""FSDP2-aware optimizer and scheduler checkpoint save/load.

The upstream HF Trainer._save_optimizer_and_scheduler relies on
accelerate's save_fsdp_optimizer which can fail when Axolotl's custom
fsdp2_prepare_model path is used.  This mixin replaces the blanket
try/except with a proper implementation that uses PyTorch's distributed
checkpoint APIs directly for FSDP2, and falls back to the upstream
Trainer for all other distributed backends.
"""

import os
import warnings

import torch
from transformers import Trainer

from axolotl.utils.logging import get_logger

SCHEDULER_NAME = "scheduler.pt"

LOG = get_logger(__name__)


def _is_fsdp2(trainer: Trainer) -> bool:
    """Return True when the trainer is running under FSDP version 2."""
    fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    return (
        getattr(trainer, "is_fsdp_enabled", False)
        and fsdp_plugin is not None
        and getattr(fsdp_plugin, "fsdp_version", 1) == 2
    )


def _get_state_dict_type(trainer: Trainer) -> str:
    """Return the configured state dict type string ('FULL_STATE_DICT', etc.)."""
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    sdt = getattr(fsdp_plugin, "state_dict_type", None)
    if sdt is None:
        return "FULL_STATE_DICT"
    return sdt.name if hasattr(sdt, "name") else str(sdt)


def _save_fsdp2_optimizer(trainer: Trainer, output_dir: str) -> None:
    """Save optimizer state under FSDP2 using PyTorch DCP APIs."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    is_full = _get_state_dict_type(trainer) == "FULL_STATE_DICT"

    sd_options = StateDictOptions(
        full_state_dict=is_full,
        cpu_offload=getattr(fsdp_plugin.state_dict_config, "offload_to_cpu", False),
        broadcast_from_rank0=getattr(
            fsdp_plugin.state_dict_config, "rank0_only", False
        ),
    )

    optim_state = get_optimizer_state_dict(
        trainer.model, trainer.optimizer, options=sd_options
    )

    if is_full:
        if trainer.args.process_index == 0:
            optim_file = os.path.join(output_dir, "optimizer.bin")
            LOG.info("Saving FSDP2 optimizer state (FULL) to %s", optim_file)
            torch.save(optim_state, optim_file)
    else:
        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

        ckpt_dir = os.path.join(output_dir, "optimizer_0")
        os.makedirs(ckpt_dir, exist_ok=True)
        LOG.info("Saving FSDP2 optimizer state (SHARDED) to %s", ckpt_dir)
        dist_cp.save(
            state_dict={"optimizer": optim_state},
            storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
            planner=DefaultSavePlanner(),
        )


def _load_fsdp2_optimizer(trainer: Trainer, checkpoint: str) -> None:
    """Load optimizer state under FSDP2 using PyTorch DCP APIs."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    is_full = _get_state_dict_type(trainer) == "FULL_STATE_DICT"

    sd_options = StateDictOptions(
        full_state_dict=is_full,
        cpu_offload=getattr(fsdp_plugin.state_dict_config, "offload_to_cpu", False),
        broadcast_from_rank0=getattr(
            fsdp_plugin.state_dict_config, "rank0_only", False
        ),
    )

    if is_full:
        optim_state = None
        optim_file = os.path.join(checkpoint, "optimizer.bin")
        if trainer.args.process_index == 0 or not getattr(
            fsdp_plugin.optim_state_dict_config, "rank0_only", False
        ):
            LOG.info("Loading FSDP2 optimizer state (FULL) from %s", optim_file)
            optim_state = torch.load(optim_file, weights_only=True)
    else:
        import torch.distributed.checkpoint as dist_cp

        ckpt_dir = os.path.join(checkpoint, "optimizer_0")
        if not os.path.isdir(ckpt_dir) and "optimizer" in checkpoint:
            ckpt_dir = checkpoint
        LOG.info("Loading FSDP2 optimizer state (SHARDED) from %s", ckpt_dir)

        optim_state = get_optimizer_state_dict(
            trainer.model, trainer.optimizer, options=sd_options
        )
        optim_state = {"optimizer": optim_state}
        dist_cp.load(
            optim_state,
            checkpoint_id=ckpt_dir,
            storage_reader=dist_cp.FileSystemReader(ckpt_dir),
        )
        optim_state = optim_state["optimizer"]

    set_optimizer_state_dict(
        trainer.model, trainer.optimizer, optim_state, options=sd_options
    )


class CheckpointSaveMixin(Trainer):
    """Mixin providing FSDP2-aware optimizer/scheduler checkpoint save and load."""

    def _save_optimizer_and_scheduler(self, output_dir):
        if _is_fsdp2(self):
            try:
                _save_fsdp2_optimizer(self, output_dir)
            except Exception as exc:
                LOG.error(
                    "Failed to save FSDP2 optimizer state: %s. "
                    "Checkpoint resumption will not restore optimizer state.",
                    exc,
                )
                raise

            if self.args.should_save:
                with warnings.catch_warnings(record=True):
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME),
                    )
        else:
            super()._save_optimizer_and_scheduler(output_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return

        if _is_fsdp2(self):
            optim_file = os.path.join(checkpoint, "optimizer.bin")
            optim_dir = os.path.join(checkpoint, "optimizer_0")
            scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME)

            has_optim = os.path.isfile(optim_file) or os.path.isdir(optim_dir)
            has_scheduler = os.path.isfile(scheduler_file)

            if has_optim and has_scheduler:
                LOG.info(
                    "Resuming FSDP2 optimizer + scheduler from checkpoint: %s",
                    checkpoint,
                )
                _load_fsdp2_optimizer(self, checkpoint)

                with warnings.catch_warnings(record=True):
                    self.lr_scheduler.load_state_dict(
                        torch.load(scheduler_file, weights_only=True)
                    )
            elif has_optim:
                LOG.warning(
                    "Found optimizer checkpoint but no scheduler state at %s",
                    checkpoint,
                )
                _load_fsdp2_optimizer(self, checkpoint)
            else:
                LOG.warning(
                    "No FSDP2 optimizer checkpoint found at %s — "
                    "training will start with a fresh optimizer.",
                    checkpoint,
                )
        else:
            super()._load_optimizer_and_scheduler(checkpoint)
