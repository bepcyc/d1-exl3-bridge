"""Scoped guards; this module is imported only by explicit runtime loading."""

import threading
from collections.abc import Mapping


def unsafe_cuda_fault(error):
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "cuda error:",
            "hip error:",
            "illegal memory access",
            "device-side assert",
            "device is lost",
            "context is destroyed",
        )
    )


def tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from tensors(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item, torch)


class DeviceGuard:
    def __init__(self, torch, device, *, thread_only=False):
        self.torch, self.device = torch, device
        self.thread = threading.get_ident()
        self.thread_only = thread_only

    def before(self, module, args, kwargs):
        if self.thread_only and threading.get_ident() != self.thread:
            return
        values = (
            args,
            kwargs,
            tuple(module.parameters(recurse=False)),
            tuple(module.buffers(recurse=False)),
        )
        if any(t.device != self.device for t in tensors(values, self.torch)):
            raise RuntimeError("A learned forward reached CPU or another GPU")

    def after(self, module, args, output):
        if any(t.device != self.device for t in tensors(output, self.torch)):
            raise RuntimeError("A learned forward returned a CPU/foreign-GPU tensor")


def dispatch_guard(torch, device):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Guard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            learned = (
                "aten.mm.",
                "aten.addmm.",
                "aten.bmm.",
                "convolution",
                "scaled_dot_product",
                "flash_attention",
                "efficient_attention",
                "aten.embedding.",
                "layer_norm",
                "rms_norm",
            )
            if any(op in name for op in learned):
                if any(
                    t.device != device for t in tensors((args, kwargs or {}), torch)
                ):
                    raise RuntimeError(
                        "A learned tensor operation reached CPU or another GPU"
                    )
            return func(*args, **(kwargs or {}))

    return Guard()
