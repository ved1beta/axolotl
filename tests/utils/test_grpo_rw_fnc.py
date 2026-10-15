import os

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.trl import TRLConfig


def test_get_rollout_func_loads_successfully():
    """Test that a valid rollout function can be loaded"""
    rollout_func = GRPOStrategy.get_rollout_func("os.path.join")
    assert callable(rollout_func)
    assert rollout_func == os.path.join


def test_get_rollout_func_invalid_module_raises_error():
    """Test that invalid module path raises clear ValueError"""
    with pytest.raises(ValueError, match="Rollout function .* not found"):
        GRPOStrategy.get_rollout_func("nonexistent_module.my_func")


def test_get_reward_func_unknown_hub_repo_raises_error(monkeypatch):
    """A reward function that is neither importable nor a hub repo gets a clear ValueError"""

    def repo_not_found(repo_id, repo_type=None):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
        raise RepositoryNotFoundError(
            "404 Client Error", response=httpx.Response(404, request=request)
        )

    monkeypatch.setattr("axolotl.core.trainers.grpo.snapshot_download", repo_not_found)
    with pytest.raises(ValueError, match="Reward function .* not found") as excinfo:
        GRPOStrategy.get_reward_func("nonexistent_module.my_reward")
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_get_reward_func_unknown_offline_raises_error(monkeypatch):
    """Offline, an unknown reward function still gets a clear ValueError"""
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(ValueError, match="Reward function .* not found") as excinfo:
        GRPOStrategy.get_reward_func("nonexistent_module.my_reward")
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_overlong_penalty_regions():
    trl = TRLConfig(max_completion_length=100, overlong_penalty_cache=20)
    (reward_funcs,) = GRPOStrategy.set_trainer_args(DictDefault(trl=trl.model_dump()))
    assert len(reward_funcs) == 1
    assert reward_funcs[0]([[1] * 80, [1] * 90, [1] * 120]) == [0.0, -0.5, -1.0]


def test_overlong_penalty_appends_reward_weight():
    trl = TRLConfig(
        max_completion_length=100,
        reward_funcs=["os.path.join"],
        reward_weights=[2.0],
        overlong_penalty_cache=20,
        overlong_penalty_weight=0.5,
    )
    cfg = DictDefault(trl=trl.model_dump(), context_parallel_size=1)
    assert len(GRPOStrategy.set_trainer_args(cfg)[0]) == 2
    assert GRPOStrategy.set_training_args_kwargs(cfg)["reward_weights"] == [2.0, 0.5]


def test_overlong_penalty_validation():
    with pytest.raises(ValueError, match="overlong_penalty_cache"):
        TRLConfig(max_completion_length=100, overlong_penalty_cache=100)
    with pytest.raises(ValueError, match="max_completion_length"):
        TRLConfig(overlong_penalty_cache=20)
