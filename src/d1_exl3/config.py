"""Explicit model/device/capacity policy; no machine-specific defaults."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    model: Path
    minimum_free_bytes: int
    allocator_limit_bytes: int
    device: str = "cuda:0"
    token_budget: int = 65536
    maximum_packed_tokens: int = 98048
    maximum_branches: int = 256
    maximum_vision_tiles: int = 11
    expected_model_sha256: str | None = None
    maximum_path_tokens: int | None = None
    exclusive_worker_native_cleanup: bool = False

    def __post_init__(self):
        if type(self.exclusive_worker_native_cleanup) is not bool:
            raise ValueError("exclusive_worker_native_cleanup must be a boolean")
        object.__setattr__(self, "model", Path(self.model).expanduser().resolve())
        if not self.device.startswith("cuda:") or not self.device[5:].isdigit():
            raise ValueError(
                "A single explicit cuda:N device is required (CUDA or ROCm Torch)"
            )
        for name in (
            "minimum_free_bytes",
            "allocator_limit_bytes",
            "token_budget",
            "maximum_packed_tokens",
            "maximum_branches",
            "maximum_vision_tiles",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.minimum_free_bytes <= self.allocator_limit_bytes:
            raise ValueError(
                "Admission must include allocator limit plus external-runtime headroom"
            )
        if self.expected_model_sha256 is not None and (
            len(self.expected_model_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expected_model_sha256)
        ):
            raise ValueError("expected_model_sha256 must be lowercase SHA256")
        if self.maximum_path_tokens is not None and (
            type(self.maximum_path_tokens) is not int or self.maximum_path_tokens <= 0
        ):
            raise ValueError("maximum_path_tokens must be positive when explicitly set")
