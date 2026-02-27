"""Monkeypatch for Qwen3_5 and Qwen3_5Moe models to pass position_ids to linear attention.

Both model variants (dense qwen3_5 and MoE qwen3_5_moe) share the same GatedDeltaNet
architecture, so a single set of forward implementations is patched into both.
"""

import importlib
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from fla.modules.conv import causal_conv1d as fla_causal_conv1d
except ImportError:
    fla_causal_conv1d = None


def get_cu_seqlens(position_ids):
    """
    Adapted from transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids.

    https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/modeling_flash_attention_utils.py#L316
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )

    return cu_seq_lens_q


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _inject_fla_kernels(module) -> None:
    """Inject FLA kernels into a modeling module, bypassing is_flash_linear_attention_available."""
    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        module.FusedRMSNormGated = FusedRMSNormGated
        module.chunk_gated_delta_rule = chunk_gated_delta_rule
        module.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        module.is_fast_path_available = True
    except ImportError:
        module.chunk_gated_delta_rule = None
        module.fused_recurrent_gated_delta_rule = None
        module.FusedRMSNormGated = None


def _patched_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.FloatTensor:
    """Shared decoder layer forward — threads position_ids into linear attention."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # MoE layers return (hidden_states, router_logits) — unpack
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states


def _make_gated_delta_forward(apply_mask_fn):
    """
    Factory: returns the patched GatedDeltaNet forward, closing over the module-specific
    apply_mask_to_padding_states function so the body stays model-type agnostic.
    """

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = apply_mask_fn(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # Compute cu_seqlens once early — reused by causal_conv1d AND chunk_gated_delta_rule
        cu_seqlens = None
        if not use_precomputed_states and position_ids is not None:
            cu_seqlens = get_cu_seqlens(position_ids=position_ids)

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (
            x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value)
        )

        mixed_qkv = torch.cat((query, key, value), dim=-1)  # [B, T, D]

        if use_precomputed_states:
            # Inference single-token path: causal_conv1d_update expects [B, D, T]
            mixed_qkv = mixed_qkv.transpose(1, 2)
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            mixed_qkv = mixed_qkv.transpose(1, 2)
        else:
            if cache_params is not None:
                # Cache expects [B, D, T]
                mixed_qkv_t = mixed_qkv.transpose(1, 2)
                cache_params.conv_states[self.layer_idx] = F.pad(
                    mixed_qkv_t,
                    (self.conv_kernel_size - mixed_qkv_t.shape[-1], 0),
                )

            if fla_causal_conv1d is not None:
                # FLA Triton kernel: [B, T, D] in/out, cu_seqlens resets state at boundaries
                mixed_qkv, _ = fla_causal_conv1d(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cu_seqlens=cu_seqlens,
                )
            else:
                # PyTorch fallback — no cu_seqlens, conv state leaks across packed sequences
                LOG.warning_once(
                    "FLA causal_conv1d not available. Falling back to PyTorch conv1d "
                    "which does not support cu_seqlens for packed sequences."
                )
                mixed_qkv = mixed_qkv.transpose(1, 2)
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
                mixed_qkv = mixed_qkv.transpose(1, 2)

        # mixed_qkv is [B, T, D] in all paths from here
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        return self.out_proj(core_attn_out)

    return patched_forward


# ---------------------------------------------------------------------------
# Unified patch entry point
# ---------------------------------------------------------------------------


def _apply_packing_patches(model_type: str, cls_prefix: str) -> None:
    """
    Apply all sample-packing patches for a qwen3_5 variant.

    Args:
        model_type:  transformers model_type string, e.g. "qwen3_5" or "qwen3_5_moe"
        cls_prefix:  class name prefix, e.g. "Qwen3_5" or "Qwen3_5Moe"
    """
    module_name = f"transformers.models.{model_type}.modeling_{model_type}"

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        LOG.warning(f"{model_type} not found in transformers, skipping packing patches")
        return

    _inject_fla_kernels(module)

    getattr(module, f"{cls_prefix}DecoderLayer").forward = _patched_decoder_forward

    gated_cls = getattr(module, f"{cls_prefix}GatedDeltaNet")
    gated_cls.forward = _make_gated_delta_forward(module.apply_mask_to_padding_states)

    LOG.info(f"Applied {cls_prefix} packing patch (fla_causal_conv1d={'available' if fla_causal_conv1d else 'unavailable'})")


# ---------------------------------------------------------------------------
# Public API — called from patch_manager.py
# ---------------------------------------------------------------------------


def patch_qwen3_5_modeling_packing():
    _apply_packing_patches("qwen3_5", "Qwen3_5")


def patch_qwen3_5_moe_modeling_packing():
    _apply_packing_patches("qwen3_5_moe", "Qwen3_5Moe")
