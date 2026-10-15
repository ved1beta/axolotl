"""Utils for Axolotl trainers"""

import torch
from torch import nn

from axolotl.utils.distributed import is_distributed


def trainable_tokens_per_sec_per_gpu(
    prev_trainable: float | None,
    curr_trainable: float,
    world_size: int,
    elapsed: float,
) -> float | None:
    """Effective per-GPU trainable-token throughput over a logging window.

    ``curr_trainable``/``prev_trainable`` are the cumulative trainable-token
    counter (SUM-reduced across all ranks) at this log and the previous one, so
    the delta covers every gradient-accumulation microbatch and the elapsed wall
    time captures in-window overhead. Returns None when there is no prior window.
    """
    if prev_trainable is None or elapsed <= 0:
        return None
    return (curr_trainable - prev_trainable) / max(1, world_size) / elapsed


# Dense bf16 peak FLOP/s per GPU; first substring match against the device name wins.
PEAK_BF16_FLOPS = (
    ("GB200", 2.5e15),
    ("B200", 2.25e15),
    ("H200", 989e12),
    ("H100 PCIe", 756e12),
    ("H100", 989e12),
    ("A100", 312e12),
)


def peak_flops_per_gpu(device_name: str) -> float | None:
    for key, flops in PEAK_BF16_FLOPS:
        if key in device_name:
            return flops
    return None


def is_experts_module(module: nn.Module) -> bool:
    # transformers' @use_experts_implementation sets is_concatenated on every experts class.
    return hasattr(module, "num_experts") and hasattr(module, "is_concatenated")


def training_flops_per_token(model: nn.Module) -> float:
    """6 * active non-embedding params (forward + backward); attention scores are ignored."""

    def numel(params):
        # ZeRO-3 partitions params to numel 0 and keeps the full size in ds_numel.
        return sum(getattr(p, "ds_numel", p.numel()) for p in params)

    embeddings = {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters()
    }
    active = numel(p for p in model.parameters() if id(p) not in embeddings)
    for m in model.modules():
        if is_experts_module(m):
            top_k = getattr(getattr(m, "config", None), "num_experts_per_tok", None)
            if top_k:
                active -= numel(m.parameters()) * (1 - top_k / m.num_experts)
    return 6.0 * active


class ExpertLoadTracker:
    """Counts tokens routed to each expert per MoE layer, kept on device until ``stats()``.

    Hooks the experts modules' ``top_k_index`` input, which every experts implementation
    (eager, grouped_mm, scattermoe, sonicmoe, expert parallel) receives from the router.
    """

    def __init__(self, model: nn.Module):
        self.counts: list[torch.Tensor | None] = []
        for module in model.modules():
            if is_experts_module(module):
                module.register_forward_pre_hook(
                    self._hook(len(self.counts)), with_kwargs=True
                )
                self.counts.append(None)

    def _hook(self, layer: int):
        def hook(module, args, kwargs):
            if not module.training:
                return
            top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
            idx = top_k_index.reshape(-1).long()
            if self.counts[layer] is None:
                # expert parallel shrinks num_experts to the local shard; routing ids stay global
                num_experts = getattr(module, "num_experts_global", module.num_experts)
                self.counts[layer] = torch.zeros(num_experts, device=idx.device)
            # index_add_ rather than bincount: bincount syncs on CUDA to size its output.
            self.counts[layer].index_add_(  # type: ignore[union-attr]
                0, idx, torch.ones_like(idx, dtype=torch.float32)
            )

        return hook

    def stats(self) -> dict[str, float]:
        """Load-balance scalars since the last call; collective when distributed."""
        if not self.counts or any(c is None for c in self.counts):
            return {}
        counts = torch.stack(self.counts)  # type: ignore[arg-type]
        for c in self.counts:
            c.zero_()  # type: ignore[union-attr]
        if is_distributed():
            torch.distributed.all_reduce(counts)
        load = counts / counts.mean(dim=-1, keepdim=True).clamp_min(1e-9)
        max_over_mean = load.max(dim=-1).values
        cv = load.std(dim=-1, unbiased=False)
        mean_max, worst_max, mean_cv = torch.stack(
            [max_over_mean.mean(), max_over_mean.max(), cv.mean()]
        ).tolist()
        return {
            "moe/expert_load_max_over_mean": round(mean_max, 4),
            "moe/expert_load_max_over_mean_worst_layer": round(worst_max, 4),
            "moe/expert_load_cv": round(mean_cv, 4),
        }


def sanitize_kwargs_for_tagging(tag_names, kwargs=None):
    if isinstance(tag_names, str):
        tag_names = [tag_names]

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names

    return kwargs


def sanitize_kwargs_for_ds_tagging(dataset_tags, kwargs=None):
    if isinstance(dataset_tags, str):
        dataset_tags = [dataset_tags]

    if (dataset_tags is not None) and (kwargs is not None):
        if "dataset_tags" not in kwargs:
            kwargs["dataset_tags"] = dataset_tags
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], list):
            kwargs["dataset_tags"].extend(dataset_tags)
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], str):
            dataset_tags.append(kwargs["dataset_tags"])
            kwargs["dataset_tags"] = dataset_tags

    return kwargs
