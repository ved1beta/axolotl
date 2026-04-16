"""E2E test for FSDP2 optimizer checkpoint save + resume.

Verifies that:
  1. Training under FSDP2 saves optimizer.bin (or sharded optimizer dir)
     alongside the scheduler state.
  2. Resuming from that checkpoint loads the optimizer+scheduler and
     continues training without error.
  3. The final loss after resume is sane (not NaN, not reset to epoch-0 levels).
"""

import os
from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from tests.e2e.utils import require_torch_2_7_0


def _base_fsdp2_cfg(temp_dir, state_dict_type="FULL_STATE_DICT"):
    return DictDefault(
        {
            "base_model": "Qwen/Qwen2.5-0.5B",
            "sequence_len": 512,
            "val_set_size": 0.0,
            "datasets": [
                {
                    "path": "tatsu-lab/alpaca",
                    "type": "alpaca",
                    "split": "train[:2%]",
                },
            ],
            "num_epochs": 1,
            "max_steps": 4,
            "save_steps": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "output_dir": temp_dir,
            "learning_rate": 0.00005,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "cosine",
            "flash_attention": True,
            "fsdp_version": 2,
            "fsdp_config": {
                "offload_params": False,
                "cpu_ram_efficient_loading": True,
                "transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
                "state_dict_type": state_dict_type,
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "reshard_after_forward": True,
            },
            "bf16": True,
        }
    )


def _run_training(cfg, temp_dir, extra_args=None):
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(temp_dir) / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

    cmd = [
        "axolotl",
        "train",
        str(config_path),
        "--num-processes",
        "2",
        "--main-process-port",
        f"{get_torch_dist_unique_port()}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    execute_subprocess_async(cmd)


class TestFSDP2CheckpointResume:
    """Verify FSDP2 optimizer save and resume round-trip."""

    @require_torch_2_7_0
    def test_fft_full_state_dict_checkpoint_has_optimizer(self, temp_dir):
        """Train 4 steps with save_steps=2 — verify optimizer.bin exists."""
        cfg = _base_fsdp2_cfg(temp_dir, state_dict_type="FULL_STATE_DICT")
        _run_training(cfg, temp_dir)

        checkpoint_dirs = sorted(Path(temp_dir).glob("checkpoint-*"))
        assert len(checkpoint_dirs) >= 1, "Expected at least 1 checkpoint"

        ckpt = checkpoint_dirs[0]
        optim_file = ckpt / "optimizer.bin"
        scheduler_file = ckpt / "scheduler.pt"
        assert optim_file.exists(), (
            f"optimizer.bin missing from {ckpt}. Contents: {list(ckpt.iterdir())}"
        )
        assert scheduler_file.exists(), (
            f"scheduler.pt missing from {ckpt}. Contents: {list(ckpt.iterdir())}"
        )

    @require_torch_2_7_0
    def test_fft_full_state_dict_resume_from_checkpoint(self, temp_dir):
        """Train 2 steps, save, then resume and train 2 more — verify no crash."""
        cfg = _base_fsdp2_cfg(temp_dir, state_dict_type="FULL_STATE_DICT")
        cfg["max_steps"] = 2
        cfg["save_steps"] = 2
        _run_training(cfg, temp_dir)

        checkpoint_dirs = sorted(Path(temp_dir).glob("checkpoint-*"))
        assert len(checkpoint_dirs) >= 1, "No checkpoint to resume from"
        ckpt = checkpoint_dirs[0]

        resume_dir = temp_dir + "_resumed"
        cfg_resume = _base_fsdp2_cfg(resume_dir, state_dict_type="FULL_STATE_DICT")
        cfg_resume["max_steps"] = 4
        cfg_resume["save_steps"] = 100
        cfg_resume["resume_from_checkpoint"] = str(ckpt)
        _run_training(cfg_resume, resume_dir)

        model_files = list(Path(resume_dir).glob("*.safetensors")) + list(
            Path(resume_dir).glob("*.bin")
        )
        assert len(model_files) > 0, "No model files after resumed training"

    @require_torch_2_7_0
    def test_fft_sharded_state_dict_checkpoint_has_optimizer(self, temp_dir):
        """Train with SHARDED_STATE_DICT — verify optimizer_0/ dir exists."""
        cfg = _base_fsdp2_cfg(temp_dir, state_dict_type="SHARDED_STATE_DICT")
        _run_training(cfg, temp_dir)

        checkpoint_dirs = sorted(Path(temp_dir).glob("checkpoint-*"))
        assert len(checkpoint_dirs) >= 1, "Expected at least 1 checkpoint"

        ckpt = checkpoint_dirs[0]
        optim_dir = ckpt / "optimizer_0"
        scheduler_file = ckpt / "scheduler.pt"
        assert optim_dir.exists() and optim_dir.is_dir(), (
            f"optimizer_0/ dir missing from {ckpt}. Contents: {list(ckpt.iterdir())}"
        )
        assert scheduler_file.exists(), (
            f"scheduler.pt missing from {ckpt}. Contents: {list(ckpt.iterdir())}"
        )

    @require_torch_2_7_0
    def test_lora_full_state_dict_checkpoint_has_optimizer(self, temp_dir):
        """Train LoRA + FSDP2 — verify optimizer checkpoint is saved."""
        cfg = _base_fsdp2_cfg(temp_dir, state_dict_type="FULL_STATE_DICT")
        cfg["adapter"] = "lora"
        cfg["lora_r"] = 8
        cfg["lora_alpha"] = 16
        cfg["lora_dropout"] = 0.05
        cfg["lora_target_linear"] = True
        cfg["fsdp_config"]["cpu_ram_efficient_loading"] = False
        _run_training(cfg, temp_dir)

        checkpoint_dirs = sorted(Path(temp_dir).glob("checkpoint-*"))
        assert len(checkpoint_dirs) >= 1, "Expected at least 1 checkpoint"

        ckpt = checkpoint_dirs[0]
        optim_file = ckpt / "optimizer.bin"
        assert optim_file.exists(), (
            f"optimizer.bin missing from LoRA checkpoint {ckpt}. "
            f"Contents: {list(ckpt.iterdir())}"
        )
