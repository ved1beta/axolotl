"""Helpers for the Q-GaLore optimizer integration.

Q-GaLore (arxiv 2407.08296) projects gradients into a low-rank subspace using a
periodically-refreshed projection matrix P. The upstream wheel
(``q-galore-torch``) exposes ``QGaLoreAdamW8bit``; it discovers which parameters
to project by reading a ``rank`` key on each ``param_group``. This module
builds those param-groups from an Axolotl config.

The companion INT8-weight-wrapping recipe from the paper is not yet wired up
(see ``check_qgalore`` in :mod:`axolotl.utils.schemas.validation`).
"""

from __future__ import annotations

from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _name_matches(name: str, patterns: list[str]) -> bool:
    return any(pat in name for pat in patterns)


def build_qgalore_param_groups(
    model: nn.Module,
    target_modules: list[str] | None,
    *,
    rank: int,
    update_proj_gap: int,
    scale: float,
    proj_type: str,
    proj_quant: bool,
    proj_bits: int,
    proj_group_size: int,
    cos_threshold: float,
    gamma_proj: int,
    queue_size: int,
) -> list[dict]:
    """Split ``model``'s trainable parameters into two groups for Q-GaLore.

    The first group carries the Q-GaLore projection settings (``rank``,
    ``update_proj_gap`` etc.). The second is a plain AdamW group for everything
    that wasn't matched by ``target_modules`` (norms, biases, embeddings, …).

    ``target_modules`` is a list of substring patterns matched against
    parameter names — identical semantics to ``optim_target_modules`` for the
    upstream HuggingFace GaLore integration.
    """
    if not target_modules:
        target_modules = ["attn", "mlp"]

    galore_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Only 2D weight matrices benefit from the low-rank projection; 1D
        # tensors (norms, biases) go to the plain AdamW group.
        if param.dim() == 2 and _name_matches(name, target_modules):
            galore_params.append(param)
        else:
            other_params.append(param)

    if not galore_params:
        raise ValueError(
            "Q-GaLore: no parameters matched optim_target_modules="
            f"{target_modules!r}. Check the pattern list against the model's "
            "parameter names."
        )

    LOG.info(
        "Q-GaLore param groups: %d projected, %d plain (target_modules=%s)",
        len(galore_params),
        len(other_params),
        target_modules,
    )

    galore_group = {
        "params": galore_params,
        "rank": rank,
        "update_proj_gap": update_proj_gap,
        "scale": scale,
        "proj_type": proj_type,
        "quant": proj_quant,
        "quant_n_bit": proj_bits,
        "quant_group_size": proj_group_size,
        "cos_threshold": cos_threshold,
        "gamma_proj": gamma_proj,
        "queue_size": queue_size,
    }
    plain_group = {"params": other_params}
    return [galore_group, plain_group]
