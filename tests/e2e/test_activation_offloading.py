"""
E2E tests for activation offloading
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists


class TestActivationOffloading:
    """
    E2E test cases for activation offloading
    """

    @pytest.mark.parametrize(
        "adapter",
        ["lora", "qlora", None],
    )
    def test_activation_offloading(
        self,
        temp_dir,
        adapter,
        monkeypatch,
    ):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                    "eos_token": "<|im_end|>",
                },
                "datasets": [
                    {
                        "chat_template": "chatml",
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 10,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": "auto",
                "gradient_checkpointing": True,
                "activation_offloading": True,
                "save_first_step": False,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
            }
        )
        if adapter == "lora":
            cfg["adapter"] = "lora"
        if adapter == "qlora":
            cfg["adapter"] = "qlora"
            cfg["load_in_4bit"] = True

        # Record OffloadActivations state at the start of each training_step.
        # Regression guard for #3638: tracker / dedup map / forward stash must
        # be empty at the start of every step. With the leak (pre-fix), these
        # grow monotonically and pin GPU memory until OOM.
        from axolotl.core.trainers.mixins import activation_checkpointing as ac_mod

        recorded_states: list[dict] = []
        original_training_step = ac_mod.ActivationOffloadingMixin.training_step

        def recording_training_step(self, *args, **kwargs):
            ctx = self.activation_offload_context
            if isinstance(ctx, ac_mod.OffloadActivations):
                recorded_states.append(
                    {
                        "step": self._offload_step_counter,
                        "tracker": len(ctx.tracker),
                        "storage_dedup": len(ctx.storage_to_tensor_id),
                        "fwd_stash": len(getattr(ctx, "fwd_stash", {})),
                        "bwd_tensor_stash": len(getattr(ctx, "bwd_tensor_stash", {})),
                        "bwd_ev_stash": len(getattr(ctx, "bwd_ev_stash", {})),
                    }
                )
            return original_training_step(self, *args, **kwargs)

        monkeypatch.setattr(
            ac_mod.ActivationOffloadingMixin,
            "training_step",
            recording_training_step,
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        # All recorded pre-step states must be clean: cross-step state never
        # carries over.
        assert recorded_states, "no training_step recorded — test setup wrong"
        for rec in recorded_states:
            assert rec["tracker"] == 0, (
                f"OffloadActivations.tracker not empty at start of step "
                f"{rec['step']}: {rec} — cross-step leak (#3638) regressed"
            )
            assert rec["storage_dedup"] == 0, (
                f"OffloadActivations.storage_to_tensor_id not empty at start "
                f"of step {rec['step']}: {rec}"
            )
            assert rec["fwd_stash"] == 0, (
                f"OffloadActivations.fwd_stash not empty at start of step "
                f"{rec['step']}: {rec}"
            )
