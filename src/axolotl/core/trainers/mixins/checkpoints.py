"""FSDP2-aware checkpoint save/load.

accelerate.save_fsdp_model branches between a single pytorch_model_fsdp.bin
file and a pytorch_model_fsdp_0/ directory based on
fsdp_plugin.state_dict_type.  Under FSDP2 that attribute is not
guaranteed to agree between the save run and the resume run, so the
file that load_fsdp_model looks for may never have been written and
resuming crashes with FileNotFoundError.

This mixin bypasses save_fsdp_model and writes the model state dict
ourselves as pytorch_model_fsdp.bin using Axolotl's monkeypatched
Accelerator.get_state_dict, which always gathers FSDP2 DTensors to
full tensors on rank 0.  Optimizer save/load stays on accelerate's
save_fsdp_optimizer / load_fsdp_optimizer, which use PyTorch DCP APIs.
"""

import os
import warnings

import torch
from transformers import Trainer

from axolotl.utils.logging import get_logger

SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_FILE = "pytorch_model_fsdp.bin"

LOG = get_logger(__name__)


def _is_fsdp2(trainer: Trainer) -> bool:
    """Return True when the trainer is running under FSDP version 2."""
    fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    return (
        getattr(trainer, "is_fsdp_enabled", False)
        and fsdp_plugin is not None
        and getattr(fsdp_plugin, "fsdp_version", 1) == 2
    )


def _save_fsdp2_model(trainer: Trainer, output_dir: str) -> None:
    """Save the FSDP2 model as pytorch_model_fsdp.bin on rank 0.

    The monkeypatched Accelerator.get_state_dict gathers all DTensors
    into full tensors on rank 0, regardless of fsdp_plugin.state_dict_type.
    We always produce FSDP_MODEL_FILE so that accelerate.load_fsdp_model
    (called by HF Trainer under FULL_STATE_DICT) finds it on resume.
    """
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    if trainer.args.process_index == 0:
        output_file = os.path.join(output_dir, FSDP_MODEL_FILE)
        LOG.info("Saving FSDP2 model state to %s", output_file)
        torch.save(state_dict, output_file)


class CheckpointSaveMixin(Trainer):
    """Mixin providing FSDP2-aware checkpoint save and load."""

    def _save_optimizer_and_scheduler(self, output_dir):
        if _is_fsdp2(self):
            from accelerate.utils import save_fsdp_optimizer

            os.makedirs(output_dir, exist_ok=True)

            _save_fsdp2_model(self, output_dir)

            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                output_dir,
            )

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
            from accelerate.utils import load_fsdp_optimizer

            optim_file = os.path.join(checkpoint, "optimizer.bin")
            optim_dir = os.path.join(checkpoint, "optimizer_0")
            scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME)

            has_optim = os.path.isfile(optim_file) or os.path.isdir(optim_dir)
            has_scheduler = os.path.isfile(scheduler_file)

            if not has_optim:
                LOG.warning(
                    "No FSDP2 optimizer checkpoint found at %s — "
                    "training will start with a fresh optimizer.",
                    checkpoint,
                )
                return

            LOG.info("Resuming FSDP2 optimizer from checkpoint: %s", checkpoint)
            load_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.optimizer,
                self.model,
                checkpoint,
            )

            if has_scheduler:
                with warnings.catch_warnings(record=True):
                    self.lr_scheduler.load_state_dict(
                        torch.load(scheduler_file, weights_only=True)
                    )
            else:
                LOG.warning(
                    "Found optimizer checkpoint but no scheduler state at %s",
                    checkpoint,
                )
        else:
            super()._load_optimizer_and_scheduler(checkpoint)
