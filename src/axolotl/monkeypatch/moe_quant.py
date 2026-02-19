"""
Quantization helpers for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, many MoE models store expert weights as fused 3D nn.Parameter
tensors instead of individual nn.Linear modules. BnB 4-bit quantization only targets
nn.Linear, so these expert weights are skipped during model loading, causing OOM.

Two mechanisms are provided:
1. quantize_moe_params_during_loading: context manager that patches transformers v5's
   set_param_for_module to quantize 3D expert params immediately as they land on GPU,
   preventing OOM accumulation during from_pretrained.
2. quantize_moe_expert_params: post-load fallback for older transformers without
   core_model_loading, or to catch anything the context manager missed.
"""

from contextlib import contextmanager

import bitsandbytes as bnb
import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_EXPERT_PARAM_KEYWORDS = ("gate_up_proj", "down_proj")


@contextmanager
def quantize_moe_params_during_loading(bnb_config):
    """Quantize 3D MoE expert params on-the-fly as they land on GPU during loading.

    Patches transformers v5's set_param_for_module so each 3D expert parameter
    (gate_up_proj / down_proj) is immediately quantized to 4-bit after being placed
    on the GPU, preventing full-precision tensors from accumulating and causing OOM.

    Falls back to a no-op if transformers.core_model_loading is unavailable
    (pre-v5 transformers); quantize_moe_expert_params handles that case instead.

    Yields:
        list[str]: full parameter paths that were quantized during loading.
    """
    try:
        import transformers.core_model_loading as cml
    except ImportError:
        yield []
        return

    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    quant_type = (
        getattr(bnb_config, "bnb_4bit_quant_type", "nf4") if bnb_config else "nf4"
    )
    compress_statistics = (
        getattr(bnb_config, "bnb_4bit_use_double_quant", True) if bnb_config else True
    )
    quantized_paths: list[str] = []
    orig_fn = cml.set_param_for_module

    def _patched(model, target_name, param_value, *args, **kwargs):
        orig_fn(model, target_name, param_value, *args, **kwargs)
        if not (
            param_value is not None
            and isinstance(param_value, torch.Tensor)
            and param_value.ndim >= 3
            and any(kw in target_name for kw in _EXPERT_PARAM_KEYWORDS)
        ):
            return
        module_path, _, param_name = target_name.rpartition(".")
        try:
            module_obj = model.get_submodule(module_path) if module_path else model
        except AttributeError:
            return
        param = getattr(module_obj, param_name, None)
        if not isinstance(param, torch.nn.Parameter):
            return
        if param.device.type in ("cpu", "meta"):
            return
        if hasattr(module_obj, "parametrizations") and param_name in getattr(
            module_obj, "parametrizations", {}
        ):
            return  # already quantized
        try:
            # Checkpoint tensors (safetensors mmap) may be non-contiguous views.
            # BnB's CUDA kernel requires a contiguous owning tensor, so clone first.
            if not param.data.is_contiguous():
                setattr(
                    module_obj,
                    param_name,
                    torch.nn.Parameter(
                        param.data.clone().contiguous(),
                        requires_grad=param.requires_grad,
                    ),
                )
            replace_parameter_4bit(
                module_obj,
                param_name,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )
            quantized_paths.append(target_name)
            torch.cuda.empty_cache()
        except Exception as exc:  # pylint: disable=broad-except
            LOG.warning("Failed to quantize %s during loading: %s", target_name, exc)

    cml.set_param_for_module = _patched
    try:
        yield quantized_paths
    finally:
        cml.set_param_for_module = orig_fn


def find_unquantized_expert_params(model):
    """Find 3D+ nn.Parameter tensors that BnB quantization skipped.

    Returns:
        List of (module, param_name) tuples to quantize.
    """
    params_to_quantize = []
    for _, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim >= 3 and any(
                kw in param_name for kw in ("experts", "gate_up_proj", "down_proj")
            ):
                params_to_quantize.append((module, param_name))
    return params_to_quantize


def quantize_moe_expert_params(model, quant_type=None, compress_statistics=None):
    """Quantize 3D nn.Parameter expert weights that BnB skips during model loading.

    Reads quant_type and compress_statistics from the model's quantization_config
    when not explicitly provided, so that the same settings used for nn.Linear
    quantization are applied to the MoE expert parameters.
    """
    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    params_to_quantize = find_unquantized_expert_params(model)
    if not params_to_quantize:
        return False

    # Derive settings from model's BnB config if not explicitly provided
    if quant_type is None or compress_statistics is None:
        bnb_config = getattr(model.config, "quantization_config", None)
        if bnb_config is not None:
            if quant_type is None:
                quant_type = getattr(bnb_config, "bnb_4bit_quant_type", "nf4")
            if compress_statistics is None:
                compress_statistics = getattr(
                    bnb_config, "bnb_4bit_use_double_quant", True
                )
    # Final defaults
    if quant_type is None:
        quant_type = "nf4"
    if compress_statistics is None:
        compress_statistics = True

    count = 0
    for module, param_name in params_to_quantize:
        replace_parameter_4bit(
            module,
            param_name,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        count += 1

    torch.cuda.empty_cache()
    LOG.info(
        "Quantized %d MoE expert parameters to 4-bit (quant_type=%s, compress_statistics=%s)",
        count,
        quant_type,
        compress_statistics,
    )
    return True
