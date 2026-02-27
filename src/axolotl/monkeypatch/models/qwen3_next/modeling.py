"""Monkeypatch for Qwen3_Next model to pass position_ids to linear attention.

Qwen3-Next and Qwen3.5 share the same GatedDeltaNet architecture — this module
delegates to the shared implementation in qwen3_5/modeling.py.
"""

from axolotl.monkeypatch.models.qwen3_5.modeling import _apply_packing_patches
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_qwen3_next_modeling_packing():
    """Apply all Qwen3Next model patches."""
    _apply_packing_patches("qwen3_next", "Qwen3Next")
