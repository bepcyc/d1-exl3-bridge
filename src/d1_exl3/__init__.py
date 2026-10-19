"""D1 typed decisions using genuine EXL3 language layers.

Importing this package loads no Torch, model, accelerator or image library.
"""

from .config import RuntimeConfig
from .runtime import D1Runtime

__all__ = ["D1Runtime", "RuntimeConfig"]
