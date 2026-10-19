"""Author model + official EXL3 Transformers integration; no alternative engines."""

import importlib.metadata
import gc
import traceback
import sys
from pathlib import Path

from .checkpoint import inspect_checkpoint
from .guards import DeviceGuard, dispatch_guard, unsafe_cuda_fault


def load(config, *, resources=None):
    # Runtime owns rollback when given this private partial-resource ledger.
    if resources is not None:
        return _load(config, resources)
    resources = {}
    try:
        return _load(config, resources)
    except BaseException as error:
        # Keep traceback locations while releasing references held by completed
        # loader frames, so failed admission/validation cannot pin model tensors.
        traceback.clear_frames(error.__traceback__)
        for handle in resources.get("handles", []):
            handle.remove()
        torch = resources.get("torch")
        resources.clear()
        gc.collect()
        if torch is not None and not unsafe_cuda_fault(error):
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                error.add_note(
                    "Partial-load cache cleanup failed: " + type(cleanup_error).__name__
                )
        raise


def _load(config, resources):
    _metadata, expected_names = inspect_checkpoint(config)
    # Nothing in this module starts a model until load() is called explicitly.
    import torch

    resources["torch"] = torch
    from exllamav3.integration.transformers import Exl3HfLinear, register
    from transformers import AutoModel, AutoProcessor

    if importlib.metadata.version("exllamav3").split("+")[0] != "1.6.2":
        raise RuntimeError("This bridge targets the ExLlamaV3 1.6.2 integration")
    resources["device"] = config.device
    resources["cuda_attempted"] = True
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU runtime is required; CPU inference is not supported")
    device = torch.device(config.device)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < config.minimum_free_bytes or config.allocator_limit_bytes > total:
        raise MemoryError("Fresh per-GPU free-memory admission failed")
    torch.cuda.set_per_process_memory_fraction(
        config.allocator_limit_bytes / total, device
    )
    register()
    guard = DeviceGuard(torch, device, thread_only=True)
    hook = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda module, args: guard.before(module, args, {})
    )
    resources["handles"] = [hook]
    try:
        with dispatch_guard(torch, device):
            model, info = AutoModel.from_pretrained(
                str(config.model),
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": str(device)},
                output_loading_info=True,
            )
            resources["model"] = model
    finally:
        original_error = sys.exception()
        try:
            hook.remove()
        except BaseException as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(
                "Loader hook cleanup failed: " + type(cleanup_error).__name__
            )
        else:
            resources["handles"].remove(hook)
    if (
        info.get("unexpected_keys")
        or set(info.get("missing_keys", [])) - {"lm_head.weight"}
        or info.get("mismatched_keys")
        or info.get("error_msgs")
    ):
        raise RuntimeError(
            "Checkpoint loading left missing, unexpected or mismatched trained weights"
        )
    model.tie_weights()
    model.eval()
    if Path(model.name_or_path).resolve() != config.model:
        raise RuntimeError(
            "Author engine would resolve a different model/processor path"
        )
    if (
        type(model).__name__ != "D1Model"
        or not callable(model.system_one)
        or not callable(model.system_one_batch)
    ):
        raise RuntimeError(
            "The loaded model does not expose the author's D1 decision API"
        )
    if model.lm_head.weight is not model.get_input_embeddings().weight:
        raise RuntimeError("The trained D1 output head is not tied to its embedding")
    if any(p.device != device or p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Trained parameters are not GPU BF16")
    named = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, Exl3HfLinear)
    }
    linears = list(named.values())
    if set(named) != expected_names or any(
        getattr(m, "inner", None) is None
        or m.inner.K != 8
        or not m.inner.mul1
        or m.inner.mcg
        for m in linears
    ):
        raise RuntimeError(
            "The language body is not the supported 166 native EXL3 q8/mul1 layers"
        )
    for module in linears:
        if module.cache_dequantized or module._dequantized is not None:
            raise RuntimeError("Persistent reconstructed EXL3 weights are not admitted")
        if any(v.device != device for v in module.buffers()):
            raise RuntimeError("Quantized EXL3 buffers are not on the selected GPU")
    engine = model.engine
    engine.token_budget = config.token_budget
    engine.processor = AutoProcessor.from_pretrained(
        str(config.model), trust_remote_code=True, local_files_only=True
    )
    guard = DeviceGuard(torch, device)
    handles = []
    resources["handles"] = handles
    counts = {
        "vision": 0,
        "language": 0,
        "exl3_layers": {name: 0 for name in expected_names},
    }
    for name, module in model.named_modules():

        def before(module, args, kwargs, label=name):
            guard.before(module, args, kwargs)
            if label in counts["exl3_layers"]:
                counts["exl3_layers"][label] += 1
            if label == "model.vision_tower":
                counts["vision"] += 1
            if label == "model.language_model.layers.0":
                counts["language"] += 1

        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(guard.after))
    return (
        model,
        torch,
        device,
        handles,
        {
            "exl3_linears": len(linears),
            "trained_head_tied": True,
            "max_path_tokens": model.config.text_config.max_position_embeddings,
            "token_budget": engine.token_budget,
            "device": str(device),
            "model_loads": 1,
            "forward_counts": counts,
            "image_batching": "author serial B1; only image-free single-question requests pack across states",
        },
    )
