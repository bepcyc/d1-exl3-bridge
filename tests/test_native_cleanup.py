"""Exclusive native lifecycle with fake backends; no accelerator imports."""

import contextlib
import io
import sys
import types
import unittest
import weakref
import gc
import threading
from pathlib import Path
from queue import Empty
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from d1_exl3 import D1Runtime, RuntimeConfig
from d1_exl3 import runtime as module
from d1_exl3.cli import main


class NativeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.cuda = types.SimpleNamespace(
            device=lambda _: contextlib.nullcontext(),
            synchronize=lambda _: self.events.append("synchronize"),
            empty_cache=lambda: self.events.append("empty"),
            memory_allocated=lambda _: self.events.append("allocated") or 0,
            memory_reserved=lambda _: self.events.append("reserved") or 0,
            max_memory_allocated=lambda _: 123,
            max_memory_reserved=lambda _: 456,
        )
        self.torch = types.SimpleNamespace(cuda=self.cuda)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(module, "_resident_runtimes", {}))
        self.stack.enter_context(
            patch.object(module, "_exclusive_cleanup_failed", False)
        )
        self.loader = self.stack.enter_context(
            patch(
                "d1_exl3.backend.load",
                side_effect=lambda config, **kwargs: (
                    object(),
                    self.torch,
                    config.device,
                    [],
                    {"model_loads": 1, "max_path_tokens": 32768},
                ),
            )
        )
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "exllamav3.util.tensor": types.SimpleNamespace(
                        g_tensor_cache=types.SimpleNamespace(
                            drop_all=lambda: self.events.append("drop")
                        )
                    ),
                    "exllamav3.util.memory": types.SimpleNamespace(
                        free_mem=lambda: self.events.append("free")
                    ),
                },
            )
        )

    def runtime(self, exclusive=True):
        return D1Runtime(
            RuntimeConfig(Path("."), 100, 90, exclusive_worker_native_cleanup=exclusive)
        )

    def test_unopened_close_never_loads_or_touches_cuda(self):
        runtime = self.runtime()
        with patch.object(
            module.gc, "collect", side_effect=AssertionError("unopened GC")
        ):
            runtime.close()
        runtime.close()
        self.loader.assert_not_called()
        self.assertEqual(self.events, [])
        receipt = runtime.stats()["native_unload"]
        self.assertEqual(receipt["status"], "NOT_LOADED")
        self.assertIsNone(receipt["allocated_bytes_after"])
        self.assertIsNone(receipt["reserved_bytes_after"])

    def test_exact_cleanup_order_and_idempotent_receipt(self):
        runtime = self.runtime().load()
        runtime._handles = [
            types.SimpleNamespace(remove=lambda: self.events.append("hook"))
        ]
        runtime.close()
        self.assertEqual(
            self.events,
            [
                "hook",
                "synchronize",
                "empty",
                "drop",
                "free",
                "synchronize",
                "allocated",
                "reserved",
            ],
        )
        self.assertIsNone(runtime._model)
        self.assertEqual(runtime._handles, [])
        receipt = runtime.stats()["native_unload"]
        self.assertEqual(receipt["status"], "RELEASED")
        self.assertTrue(receipt["scratch_cache_cleared"])
        self.assertEqual(receipt["cleanup_errors"], [])
        self.assertEqual(receipt["allocated_bytes_after"], 0)
        self.assertEqual(receipt["reserved_bytes_after"], 0)
        runtime.close()
        self.assertEqual(runtime.stats()["native_unload"], receipt)

    def test_default_close_does_not_clear_global_native_caches(self):
        runtime = self.runtime(False).load()
        runtime.close()
        self.assertEqual(self.events, ["empty"])
        self.assertEqual(runtime.stats()["native_unload"]["status"], "NOT_REQUESTED")

    def test_exclusive_refuses_other_resident_and_loading_runtimes(self):
        first = self.runtime(False).load()
        second = self.runtime(False).load()
        exclusive = self.runtime()
        with self.assertRaisesRegex(RuntimeError, "dedicated"):
            exclusive.load()
        self.assertEqual(self.loader.call_count, 2)
        first.close()
        second.close()
        exclusive.load()
        exclusive.close()

    def test_exclusive_reservation_blocks_new_load_before_backend_dispatch(self):
        first = self.runtime().load()
        second = self.runtime(False)
        with self.assertRaisesRegex(RuntimeError, "dedicated"):
            second.load()
        self.assertEqual(self.loader.call_count, 1)
        first.close()
        second.load()
        second.close()

    def test_reservation_exists_during_backend_load(self):
        first = self.runtime()
        second = self.runtime(False)
        self.loader.side_effect = lambda config, **kwargs: self.assert_blocked(
            second, config
        )
        first.load()
        first.close()
        self.assertEqual(self.loader.call_count, 1)

    def assert_blocked(self, second, config):
        with self.assertRaisesRegex(RuntimeError, "dedicated"):
            second.load()
        return (
            object(),
            self.torch,
            config.device,
            [],
            {"model_loads": 1, "max_path_tokens": 32768},
        )

    def test_nonzero_actual_memory_is_not_release_and_blocks_reuse(self):
        self.cuda.memory_allocated = lambda _: 123
        runtime = self.runtime().load()
        runtime.close()
        receipt = runtime.stats()["native_unload"]
        self.assertEqual(receipt["allocated_bytes_after"], 123)
        self.assertEqual(receipt["status"], "NOT_RELEASED")
        with self.assertRaisesRegex(RuntimeError, "healthy"):
            self.runtime(False).load()

    def test_cleanup_exception_is_retained_and_blocks_reuse(self):
        native = sys.modules["exllamav3.util.memory"]
        native.free_mem = Mock(side_effect=RuntimeError("native cleanup failed"))
        runtime = self.runtime().load()
        runtime.close()
        receipt = runtime.stats()["native_unload"]
        self.assertEqual(receipt["status"], "CLEANUP_FAILED")
        self.assertIn("native cleanup failed", receipt["cleanup_errors"][0])
        self.assertIsNone(receipt["allocated_bytes_after"])
        with self.assertRaisesRegex(RuntimeError, "healthy"):
            self.runtime().load()

    def test_unsafe_cuda_context_skips_all_cuda_and_native_cleanup(self):
        runtime = self.runtime().load()
        runtime._unsafe_cuda = True
        runtime.close()
        self.assertEqual(self.events, [])
        self.assertEqual(
            runtime.stats()["native_unload"]["status"], "UNSAFE_CUDA_CONTEXT"
        )

    def test_failed_load_never_invents_release_or_initializes_cuda_on_close(self):
        self.loader.side_effect = RuntimeError("partial load")
        runtime = self.runtime()
        with self.assertRaisesRegex(RuntimeError, "partial"):
            runtime.load()
        runtime.close()
        self.assertEqual(self.events, [])
        self.assertEqual(
            runtime.stats()["native_unload"]["status"], "PRE_CUDA_LOAD_FAILED"
        )
        self.assertFalse(module._resident_runtimes)

    def test_cli_idle_default_is_300_and_always_closes(self):
        fake = Mock()
        fake.stats.return_value = {"native_unload": {"status": "NOT_LOADED"}}
        queue = Mock()
        queue.get.side_effect = Empty
        with (
            patch("d1_exl3.cli.D1Runtime", return_value=fake) as constructor,
            patch("d1_exl3.cli.lines", return_value=queue),
        ):
            result = main(
                [
                    "--model",
                    ".",
                    "--minimum-free-gib",
                    "2",
                    "--allocator-gib",
                    "1",
                    "--exclusive-worker-native-cleanup",
                ]
            )
        self.assertEqual(result, 0)
        queue.get.assert_called_once_with(timeout=300)
        fake.close.assert_called_once()
        self.assertTrue(constructor.call_args.args[0].exclusive_worker_native_cleanup)

    def test_cli_returns_failure_for_unproven_release(self):
        fake = Mock()
        fake.stats.return_value = {
            "native_unload": {"status": "NOT_RELEASED", "allocated_bytes_after": 123}
        }
        queue = Mock()
        queue.get.side_effect = Empty
        with (
            patch("d1_exl3.cli.D1Runtime", return_value=fake),
            patch("d1_exl3.cli.lines", return_value=queue),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            result = main(
                [
                    "--model",
                    ".",
                    "--minimum-free-gib",
                    "2",
                    "--allocator-gib",
                    "1",
                    "--exclusive-worker-native-cleanup",
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn("123", stderr.getvalue())

    def test_cleanup_option_requires_boolean(self):
        with self.assertRaises(ValueError):
            RuntimeConfig(Path("."), 100, 90, exclusive_worker_native_cleanup=1)

    def test_memory_stats_are_live_only_for_loaded_safe_runtime(self):
        runtime = self.runtime()
        self.assertIsNone(runtime.stats()["cuda_memory"])
        self.assertEqual(self.events, [])
        runtime.load()
        sampled = runtime.stats()
        self.assertTrue(sampled["cuda_memory_current"])
        self.assertEqual(sampled["cuda_memory"]["peak_allocated_bytes"], 123)
        self.assertEqual(sampled["cuda_memory"]["peak_reserved_bytes"], 456)
        runtime.close()
        events = list(self.events)
        retained = runtime.stats()
        self.assertFalse(retained["cuda_memory_current"])
        self.assertEqual(retained["cuda_memory"], sampled["cuda_memory"])
        self.assertEqual(self.events, events)

    def test_dropped_runtime_keeps_opaque_ownership_without_retaining_model(self):
        runtime = self.runtime().load()
        ref = weakref.ref(runtime)
        del runtime
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(len(module._resident_runtimes), 1)
        with self.assertRaisesRegex(RuntimeError, "dedicated"):
            self.runtime(False).load()

    def test_default_unsafe_chained_forward_blocks_admission_before_close(self):
        def fail(*args):
            try:
                raise RuntimeError("CUDA error: device-side assert")
            except RuntimeError as error:
                raise ValueError("wrapped model failure") from error

        runtime = self.runtime(False).load()
        runtime._model = types.SimpleNamespace(engine=object(), system_one=fail)
        self.torch.inference_mode = contextlib.nullcontext
        with (
            patch("d1_exl3.capacity.preflight"),
            patch(
                "d1_exl3.guards.dispatch_guard", return_value=contextlib.nullcontext()
            ),
        ):
            with self.assertRaisesRegex(ValueError, "wrapped"):
                runtime.system_one(None, {"x": {"type": "choice"}})
        with self.assertRaisesRegex(RuntimeError, "healthy"):
            self.runtime(False).load()
        runtime.close()
        self.assertEqual(self.events, [])
        self.assertEqual(
            runtime.stats()["native_unload"]["status"], "UNSAFE_CUDA_CONTEXT"
        )

    def test_partial_constructor_exception_chain_does_not_retain_model(self):
        refs = []

        class Model:
            pass

        def construct():
            model = Model()
            refs.append(weakref.ref(model))
            try:
                raise ValueError("inner constructor")
            except ValueError as error:
                raise RuntimeError("outer constructor") from error

        def load(config, *, resources):
            resources.update(
                torch=self.torch, device=config.device, cuda_attempted=True
            )
            construct()

        self.loader.side_effect = load
        runtime = self.runtime()
        with self.assertRaisesRegex(RuntimeError, "outer") as retained:
            runtime.load()
        self.assertIsInstance(retained.exception.__cause__, ValueError)
        self.assertIsNone(refs[0]())
        receipt = runtime.stats()["native_unload"]
        self.assertTrue(receipt["load_failed"])
        self.assertEqual(receipt["status"], "RELEASED")
        self.assertFalse(module._resident_runtimes)

    def test_post_load_failure_keeps_original_when_hook_cleanup_also_fails(self):
        def load(config, *, resources):
            resources.update(
                torch=self.torch,
                device=config.device,
                cuda_attempted=True,
                model=object(),
                handles=[
                    types.SimpleNamespace(
                        remove=Mock(side_effect=RuntimeError("hook failed"))
                    )
                ],
            )
            raise ValueError("original load failure")

        self.loader.side_effect = load
        runtime = self.runtime()
        with self.assertRaisesRegex(ValueError, "original load failure"):
            runtime.load()
        receipt = runtime.stats()["native_unload"]
        self.assertEqual(receipt["status"], "CLEANUP_FAILED")
        self.assertIn("hook failed", receipt["cleanup_errors"][0])
        self.assertTrue(module._exclusive_cleanup_failed)

    def test_interrupted_cleanup_retains_receipt_and_blocks_replacement(self):
        runtime = self.runtime().load()
        runtime._handles = [
            types.SimpleNamespace(
                remove=Mock(side_effect=KeyboardInterrupt("interrupted"))
            )
        ]
        with self.assertRaises(KeyboardInterrupt):
            runtime.close()
        self.assertEqual(runtime.stats()["native_unload"]["status"], "CLEANUP_FAILED")
        self.assertEqual(self.events, [])
        with self.assertRaisesRegex(RuntimeError, "healthy"):
            self.runtime().load()

    def test_exclusive_cleanup_reservation_blocks_concurrent_loading(self):
        runtime = self.runtime().load()
        entered = threading.Event()
        release = threading.Event()

        def synchronize(device):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test timeout")

        self.cuda.synchronize = synchronize
        thread = threading.Thread(target=runtime.close)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(RuntimeError, "dedicated"):
                self.runtime(False).load()
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(runtime.stats()["native_unload"]["status"], "RELEASED")

    def test_partial_measurement_is_retained_when_second_read_fails(self):
        self.cuda.memory_allocated = lambda _: 456
        self.cuda.memory_reserved = Mock(side_effect=RuntimeError("measurement failed"))
        runtime = self.runtime().load()
        runtime.close()
        receipt = runtime.stats()["native_unload"]
        self.assertEqual(receipt["status"], "CLEANUP_FAILED")
        self.assertEqual(receipt["allocated_bytes_after"], 456)
        self.assertIsNone(receipt["reserved_bytes_after"])

    def test_cuda_cleanup_failure_stops_subsequent_native_operations(self):
        self.cuda.synchronize = Mock(
            side_effect=RuntimeError("CUDA error: device is lost")
        )
        runtime = self.runtime().load()
        runtime.close()
        self.assertEqual(self.events, [])
        self.assertEqual(runtime.stats()["native_unload"]["status"], "CLEANUP_FAILED")

    def test_loaded_sibling_cannot_dispatch_sample_or_cleanup_after_process_failure(
        self,
    ):
        first = self.runtime(False).load()
        second = self.runtime(False).load()
        first._unsafe_cuda = True
        first.close()
        before = list(self.events)
        with self.assertRaisesRegex(RuntimeError, "healthy"):
            second.system_one(None, {"x": {"type": "choice"}})
        self.assertFalse(second.stats()["cuda_memory_current"])
        second.close()
        self.assertEqual(self.events, before)
        self.assertEqual(second.stats()["native_unload"]["status"], "PROCESS_UNUSABLE")

    def test_interrupt_during_terminal_state_transition_is_finalized(self):
        runtime = self.runtime().load()
        original = D1Runtime.__setattr__
        interrupted = False

        def assign(obj, name, value):
            nonlocal interrupted
            original(obj, name, value)
            if obj is runtime and name == "_closed" and value and not interrupted:
                interrupted = True
                raise KeyboardInterrupt("terminal transition")

        with patch.object(D1Runtime, "__setattr__", assign):
            with self.assertRaises(KeyboardInterrupt):
                runtime.close()
        self.assertEqual(runtime.stats()["native_unload"]["status"], "CLEANUP_FAILED")
        self.assertTrue(module._exclusive_cleanup_failed)
        self.assertEqual(self.events, [])

    def test_default_close_preserves_cleanup_exception_after_recording(self):
        runtime = self.runtime(False).load()
        self.cuda.empty_cache = Mock(
            side_effect=RuntimeError("ordinary cleanup failed")
        )
        with self.assertRaisesRegex(RuntimeError, "ordinary cleanup"):
            runtime.close()
        self.assertEqual(runtime.stats()["native_unload"]["status"], "CLEANUP_FAILED")
        self.assertTrue(module._exclusive_cleanup_failed)

    def test_context_manager_keeps_original_body_exception(self):
        runtime = self.runtime(False)
        self.cuda.empty_cache = Mock(
            side_effect=RuntimeError("ordinary cleanup failed")
        )
        with self.assertRaisesRegex(ValueError, "original body") as caught:
            with runtime:
                raise ValueError("original body")
        self.assertIn("Runtime cleanup also failed", caught.exception.__notes__[0])
        self.assertEqual(runtime.stats()["native_unload"]["status"], "CLEANUP_FAILED")

    def test_cli_default_cleanup_failure_returns_nonzero_and_emits_receipt(self):
        fake = Mock()
        fake.stats.return_value = {
            "native_unload": {
                "status": "CLEANUP_FAILED",
                "cleanup_errors": ["specific failure"],
            }
        }
        queue = Mock()
        queue.get.return_value = None
        with (
            patch("d1_exl3.cli.D1Runtime", return_value=fake),
            patch("d1_exl3.cli.lines", return_value=queue),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            result = main(
                ["--model", ".", "--minimum-free-gib", "2", "--allocator-gib", "1"]
            )
        self.assertEqual(result, 1)
        self.assertIn("specific failure", stderr.getvalue())

    def test_cli_error_path_retains_original_and_cleanup_receipt(self):
        fake = Mock()
        fake.close.side_effect = RuntimeError("cleanup exception")
        fake.stats.return_value = {
            "native_unload": {
                "status": "CLEANUP_FAILED",
                "cleanup_errors": ["specific failure"],
            }
        }
        queue = Mock()
        queue.get.side_effect = ValueError("original input")
        with (
            patch("d1_exl3.cli.D1Runtime", return_value=fake),
            patch("d1_exl3.cli.lines", return_value=queue),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaisesRegex(ValueError, "original input") as caught:
                main(
                    [
                        "--model",
                        ".",
                        "--minimum-free-gib",
                        "2",
                        "--allocator-gib",
                        "1",
                        "--exclusive-worker-native-cleanup",
                    ]
                )
        self.assertIn("specific failure", stderr.getvalue())
        self.assertIn("Runtime cleanup also failed", caught.exception.__notes__[0])

    def test_unsafe_preflight_failure_also_blocks_process(self):
        runtime = self.runtime(False).load()
        runtime._model = types.SimpleNamespace(engine=object())
        with patch(
            "d1_exl3.capacity.preflight",
            side_effect=RuntimeError("CUDA error: context is destroyed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA error"):
                runtime.system_one(None, {"x": {"type": "choice"}})
        self.assertTrue(module._exclusive_cleanup_failed)
        runtime.close()
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
