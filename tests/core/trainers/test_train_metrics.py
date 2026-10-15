"""Tests for opt-in MFU and MoE expert-load logging."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from axolotl.core.trainers import base as trainer_base
from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.trainers.utils import (
    ExpertLoadTracker,
    is_experts_module,
    peak_flops_per_gpu,
    training_flops_per_token,
)


@pytest.fixture(name="moe_model")
def fixture_moe_model():
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    return Qwen3MoeForCausalLM(config)


def test_peak_flops_lookup():
    assert peak_flops_per_gpu("NVIDIA H100 80GB HBM3") == 989e12
    assert peak_flops_per_gpu("NVIDIA H100 PCIe") == 756e12
    assert peak_flops_per_gpu("NVIDIA GB200") == 2.5e15
    assert peak_flops_per_gpu("NVIDIA GeForce RTX 3090") is None


def test_flops_per_token_dense_excludes_embeddings():
    model = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 8, bias=False))
    assert training_flops_per_token(model) == 6 * 32


def test_flops_per_token_moe_counts_active_experts(moe_model):
    experts = [m for m in moe_model.modules() if is_experts_module(m)]
    expert_params = sum(p.numel() for m in experts for p in m.parameters())
    non_embed = moe_model.num_parameters() - moe_model.model.embed_tokens.weight.numel()

    assert len(experts) == 2
    assert training_flops_per_token(moe_model) == 6 * (non_embed - expert_params / 2)


def test_expert_load_stats(moe_model):
    tracker = ExpertLoadTracker(moe_model)
    moe_model.train()
    assert tracker.stats() == {}

    moe_model(torch.randint(0, 32, (2, 5)))
    assert all(c.sum().item() == 2 * 5 * 2 for c in tracker.counts)

    for counts in tracker.counts:
        counts.copy_(torch.tensor([2.0, 2.0, 0.0, 0.0]))
    tracker.counts[1].copy_(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert tracker.stats() == {
        "moe/expert_load_max_over_mean": 1.5,
        "moe/expert_load_max_over_mean_worst_layer": 2.0,
        "moe/expert_load_cv": 0.5,
    }
    assert all(c.sum().item() == 0 for c in tracker.counts)


def test_expert_load_ignores_eval_forward(moe_model):
    tracker = ExpertLoadTracker(moe_model)
    moe_model.eval()
    moe_model(torch.randint(0, 32, (1, 4)))
    assert tracker.counts == [None, None]


def _log(trainer, logs):
    forwarded = []
    with (
        patch.object(trainer_base, "is_main_process", return_value=False),
        patch.object(
            trainer_base.Trainer,
            "log",
            lambda _self, logs, start_time=None: forwarded.append(dict(logs)),
        ),
    ):
        AxolotlTrainer.log(trainer, logs)
    return forwarded[0]


def _trainer(**attrs) -> AxolotlTrainer:
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(include_tkps=False)
    trainer._stored_metrics = defaultdict(dict)
    for key, value in attrs.items():
        setattr(trainer, key, value)
    return trainer


def test_log_adds_expert_load_only_when_enabled(moe_model):
    assert not any(k.startswith("moe/") for k in _log(_trainer(), {"loss": 1.0}))

    tracker = ExpertLoadTracker(moe_model)
    moe_model.train()
    moe_model(torch.randint(0, 32, (1, 4)))
    logs = _log(_trainer(_expert_load=tracker), {"loss": 1.0})
    assert "moe/expert_load_cv" in logs


def test_log_mfu_from_total_token_rate():
    trainer = _trainer(
        state=SimpleNamespace(tokens={}),
        _tkps_prev_trainable=None,
        _tkps_prev_time=None,
        _mfu_prev_total=None,
        _mfu_flops_per_token=1e9,
        _mfu_peak_flops=1e14,
    )
    trainer.args.include_tkps = True
    trainer._sync_token_counts = lambda: None

    for now, total in ((0.0, 0), (2.0, 20_000)):
        trainer.state.tokens = {
            "trainable": torch.tensor(total // 2),
            "total": torch.tensor(total),
        }
        with patch.object(trainer_base.time, "perf_counter", return_value=now):
            logs = _log(trainer, {"loss": 1.0})

    # 10k tokens/s * 1 GFLOP/token over a 100 TFLOP/s peak
    assert logs["mfu"] == 0.1
