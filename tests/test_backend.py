"""Loader rejection/order contracts with no accelerator dependencies."""

import contextlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from d1_exl3 import RuntimeConfig, backend
from d1_exl3.checkpoint import expected_linears

META = json.loads((ROOT / "upstream/config.json").read_text())
NAMES = expected_linears(META)


class Tensor:
    def __init__(self, device="cuda:0", dtype="BF16"):
        self.device, self.dtype = device, dtype


class Handle:
    removed = False

    def remove(self):
        self.removed = True


class Linear:
    def __init__(self):
        self.inner = types.SimpleNamespace(K=8, mul1=True, mcg=False)
        self.cache_dequantized = False
        self._dequantized = None
        self.buffer = Tensor(dtype="UINT8")

    def buffers(self, recurse=True):
        return [self.buffer]

    def parameters(self, recurse=True):
        return []

    def register_forward_pre_hook(self, *a, **k):
        self.before = a[0]
        return Handle()

    def register_forward_hook(self, *a, **k):
        return Handle()


class D1Model:
    def __init__(self, path):
        self.name_or_path = str(path)
        self.weight = Tensor()
        self.lm_head = types.SimpleNamespace(weight=self.weight)
        self.embedding = types.SimpleNamespace(weight=self.weight)
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(max_position_embeddings=32768)
        )
        self.engine = types.SimpleNamespace(token_budget=65536, processor=None)
        self.linears = {n: Linear() for n in NAMES}

    def tie_weights(self):
        pass

    def eval(self):
        return self

    def get_input_embeddings(self):
        return self.embedding

    def parameters(self, recurse=True):
        return [self.weight]

    def named_modules(self):
        return list(self.linears.items())

    def modules(self):
        return list(self.linears.values())

    def system_one(self, *a):
        raise AssertionError("No inference in loader tests")

    def system_one_batch(self, *a):
        raise AssertionError("No inference in loader tests")


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig(ROOT, 100, 90)
        self.model = D1Model(self.config.model)
        self.events = []
        self.handle = Handle()
        self.info = {}

        def load(path, **kwargs):
            self.events.append(("load", path, kwargs))
            return self.model, self.info

        transformer = types.ModuleType("transformers")
        transformer.AutoModel = types.SimpleNamespace(from_pretrained=load)
        transformer.AutoProcessor = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: object()
        )
        plugin = types.ModuleType("exllamav3.integration.transformers")
        plugin.Exl3HfLinear = Linear
        plugin.register = lambda: self.events.append(("register",))
        self.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda d: None,
            mem_get_info=lambda d: (200, 300),
            set_per_process_memory_fraction=lambda *a: None,
            empty_cache=lambda: self.events.append(("cleanup",)),
        )
        torch = types.ModuleType("torch")
        torch.cuda = self.cuda
        torch.device = lambda d: d
        torch.bfloat16 = "BF16"
        torch.Tensor = Tensor
        torch.nn = types.SimpleNamespace(
            modules=types.SimpleNamespace(
                module=types.SimpleNamespace(
                    register_module_forward_pre_hook=lambda hook: self.handle
                )
            )
        )
        self.patches = [
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "transformers": transformer,
                    "exllamav3": types.ModuleType("exllamav3"),
                    "exllamav3.integration": types.ModuleType("exllamav3.integration"),
                    "exllamav3.integration.transformers": plugin,
                },
            ),
            patch.object(backend, "inspect_checkpoint", return_value=(META, NAMES)),
            patch.object(
                backend.importlib.metadata, "version", return_value="1.6.2+test"
            ),
            patch.object(
                backend, "dispatch_guard", return_value=contextlib.nullcontext()
            ),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_registration_loader_arguments_and_head_are_preserved(self):
        model, torch, device, handles, receipt = backend.load(self.config)
        self.assertIs(model, self.model)
        self.assertEqual(self.events[0], ("register",))
        kwargs = self.events[1][2]
        self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
        self.assertTrue(
            kwargs["trust_remote_code"]
            and kwargs["local_files_only"]
            and kwargs["output_loading_info"]
        )
        self.assertEqual(receipt["exl3_linears"], 166)
        self.assertIs(model.lm_head.weight, model.embedding.weight)
        self.assertTrue(self.handle.removed)
        name = next(iter(self.model.linears))
        module = self.model.linears[name]
        module.before(module, (Tensor(),), {})
        self.assertEqual(receipt["forward_counts"]["exl3_layers"][name], 1)
        with self.assertRaises(RuntimeError):
            module.before(module, (), {"hidden": Tensor(device="cpu")})
        self.assertEqual(receipt["forward_counts"]["exl3_layers"][name], 1)

    def test_admission_precedes_registration_and_weight_load(self):
        self.cuda.mem_get_info = lambda d: (99, 300)
        with self.assertRaises(MemoryError):
            backend.load(self.config)
        self.assertFalse(any(event[0] in ("register", "load") for event in self.events))

    def test_wrong_replacement_names_rejected_even_with166_modules(self):
        key = next(iter(self.model.linears))
        module = self.model.linears.pop(key)
        self.model.linears["not_the_language_body"] = module
        with self.assertRaises(RuntimeError):
            backend.load(self.config)

    def test_incomplete_or_wrong_native_quantization_rejected(self):
        module = next(iter(self.model.linears.values()))
        for value in (
            None,
            types.SimpleNamespace(K=4, mul1=True, mcg=False),
            types.SimpleNamespace(K=8, mul1=False, mcg=False),
        ):
            with self.subTest(value=value):
                module.inner = value
                with self.assertRaises(RuntimeError):
                    backend.load(self.config)

    def test_cpu_quant_buffer_and_wrong_parameter_dtype_rejected(self):
        module = next(iter(self.model.linears.values()))
        module.buffer.device = "cpu"
        with self.assertRaises(RuntimeError):
            backend.load(self.config)
        module.buffer.device = "cuda:0"
        self.model.weight.dtype = "FP32"
        with self.assertRaises(RuntimeError):
            backend.load(self.config)

    def test_loading_errors_and_broken_tie_rejected(self):
        self.info["missing_keys"] = ["model.language_model.embed_tokens.weight"]
        with self.assertRaises(RuntimeError):
            backend.load(self.config)
        self.assertIn(("cleanup",), self.events)
        self.info.clear()
        self.model.lm_head.weight = Tensor()
        with self.assertRaises(RuntimeError):
            backend.load(self.config)

    def test_partial_model_load_removes_constructor_hook_and_cleans_cache(self):
        def failed(*a, **k):
            raise RuntimeError("Weight stream failed")

        sys.modules["transformers"].AutoModel.from_pretrained = failed
        with self.assertRaises(RuntimeError):
            backend.load(self.config)
        self.assertTrue(self.handle.removed)
        self.assertIn(("cleanup",), self.events)


if __name__ == "__main__":
    unittest.main()
