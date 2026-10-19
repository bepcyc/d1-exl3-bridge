"""Runtime contracts against original author readout; no Torch/model/GPU loads."""

import contextlib
import io
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d1_exl3 import D1Runtime, RuntimeConfig
from d1_exl3.capacity import check_tree
from d1_exl3.protocol import validate_response
from upstream import prompt
from upstream.api import SystemOneApi

QUESTION = {
    "arbitrary": {
        "type": "score",
        "instructions": "How strong is the evidence?",
        "criteria": ["none", "some", "strong"],
    }
}


class Tokenizer:
    bos_token = ""

    def encode(self, text, **kwargs):
        return list(text.encode())


class Engine(SystemOneApi):
    __module__ = "test_author.runner"

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.bos = ""
        self.state_style = "json"
        self.system = "none"
        self.lead = ""
        self.option_style = "desc"
        self.token_budget = 65536
        self.seen = []

    def render(self, state, q):
        return json.dumps(state) + q.instructions

    def _image_markup(self, count):
        return "<image>" * count

    def processor(self, **kwargs):
        return {
            "input_ids": types.SimpleNamespace(shape=(1, 55)),
            "pixel_values": types.SimpleNamespace(
                shape=(len(kwargs["images"][0]), 256, 768)
            ),
        }

    def _plan(self, lengths):
        # Deliberately native behavior: oversized individual items are accepted.
        return [list(range(len(lengths)))] if lengths else []

    def run(self, requests):
        self.seen.append(requests)
        return [
            ([[0.1, 0.3, 0.6] if q.type == "score" else [0.7, 0.3] for q in qs], 17)
            for _, qs, _ in requests
        ]


class FakeModel:
    def __init__(self):
        self.engine = Engine()

    def system_one(self, *args):
        return self.engine.system_one(*args)

    def system_one_batch(self, *args):
        return self.engine.system_one_batch(*args)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig(ROOT, 100, 90)
        self.model = FakeModel()
        self.torch = types.SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(
                device=lambda _: contextlib.nullcontext(),
                empty_cache=lambda: None,
                memory_allocated=lambda _: 0,
                memory_reserved=lambda _: 0,
                max_memory_allocated=lambda _: 0,
                max_memory_reserved=lambda _: 0,
            ),
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "test_author.prompt": prompt,
                "test_author.runner": types.SimpleNamespace(
                    cap_pixels=lambda image: image
                ),
            },
        )
        self.modules.start()
        self.loader = patch(
            "d1_exl3.backend.load",
            return_value=(
                self.model,
                self.torch,
                "cuda:0",
                [],
                {"model_loads": 1, "max_path_tokens": 32768},
            ),
        )
        self.load = self.loader.start()
        self.dispatch = patch(
            "d1_exl3.guards.dispatch_guard", return_value=contextlib.nullcontext()
        )
        self.dispatch.start()

    def tearDown(self):
        self.dispatch.stop()
        self.loader.stop()
        self.modules.stop()

    def test_one_resident_load_preserves_original_score_and_legend(self):
        with D1Runtime(self.config) as runtime:
            a = runtime.system_one({"context": ["user", 23]}, QUESTION)
            b = runtime.system_one("another dataset", QUESTION)
            self.assertAlmostEqual(a["answers"]["arbitrary"]["score"], 1.5)
            self.assertEqual(
                a["answers"]["arbitrary"]["legend"],
                {"0": "none", "1": "some", "2": "strong"},
            )
            self.assertEqual(
                b["answers"]["arbitrary"]["probabilities"],
                {"0": 0.1, "1": 0.3, "2": 0.6},
            )
            self.assertEqual(runtime.stats()["requests"], 2)
        self.assertEqual(self.load.call_count, 1)

    def test_arbitrary_state_and_mixed_request_order_are_forwarded(self):
        requests = [
            ({"nested": True}, QUESTION),
            ("second", {"x": {"type": "noul", "instructions": "Is it present?"}}),
        ]
        with D1Runtime(self.config) as runtime:
            values = runtime.system_one_batch(requests)
            self.assertEqual(
                [list(v["answers"]) for v in values], [["arbitrary"], ["x"]]
            )
            self.assertEqual(self.model.engine.seen[0][0][0], {"nested": True})
            self.assertEqual(self.model.engine.seen[0][1][0], "second")

    def test_raw_probabilities_use_original_author_api(self):
        with D1Runtime(self.config) as runtime:
            self.assertEqual(
                runtime.probabilities("state", list(QUESTION.values())),
                [[0.1, 0.3, 0.6]],
            )
            rows = runtime.probabilities_batch(
                [("a", list(QUESTION.values())), ("b", list(QUESTION.values()))]
            )
            self.assertEqual(rows, [[[0.1, 0.3, 0.6]], [[0.1, 0.3, 0.6]]])

    def test_multiple_images_and_arbitrary_questions_reach_original_api(self):
        images = [object(), object()]
        with D1Runtime(self.config) as runtime:
            answer = runtime.system_one(None, QUESTION, images)
            self.assertEqual(answer["answers"]["arbitrary"]["score"], 1.5)
            forwarded = self.model.engine.seen[0][0][2]
            self.assertIs(forwarded, images)

    def test_excess_image_tiles_fail_before_any_neural_forward(self):
        with D1Runtime(self.config) as runtime:
            with self.assertRaises(ValueError):
                runtime.system_one(None, QUESTION, [object()] * 12)
            self.assertEqual(self.model.engine.seen, [])

    def test_oversized_single_row_is_rejected_before_forward_without_reload(self):
        with D1Runtime(self.config) as runtime:
            with self.assertRaises(ValueError):
                runtime.system_one("x" * 33000, QUESTION)
            self.assertEqual(self.model.engine.seen, [])
            self.assertTrue(runtime.system_one("valid", QUESTION))
        self.assertEqual(self.load.call_count, 1)

    def test_failed_load_does_not_automatically_reload(self):
        self.load.side_effect = RuntimeError("load failure")
        runtime = D1Runtime(self.config)
        with self.assertRaises(RuntimeError):
            runtime.load()
        with self.assertRaises(RuntimeError):
            runtime.load()
        self.assertEqual(self.load.call_count, 1)
        runtime.close()
        runtime.close()

    def test_corrupt_native_response_poisoned_until_close(self):
        self.model.system_one = lambda *a: {"answers": {}}
        cleanups = []
        self.torch.cuda.empty_cache = lambda: cleanups.append(True)
        runtime = D1Runtime(self.config)
        with self.assertRaises(RuntimeError):
            runtime.system_one(None, QUESTION)
        with self.assertRaises(RuntimeError):
            runtime.system_one(None, QUESTION)
        self.assertEqual(self.load.call_count, 1)
        runtime.close()
        self.assertEqual(cleanups, [True])


class PureTests(unittest.TestCase):
    def test_import_and_help_do_not_import_optional_accelerator_packages(self):
        program = """import builtins,sys
real=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in ('torch','transformers','exllamav3','PIL'):raise AssertionError(name)
 return real(name,*a,**k)
builtins.__import__=guarded
import d1_exl3
from d1_exl3.cli import main
try: main(['--help'])
except SystemExit as e: assert e.code==0
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", program],
            check=False,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capacity_includes_trunk_and_does_not_trust_native_plan(self):
        with self.assertRaises(ValueError):
            check_tree(
                100,
                [200],
                path_limit=256,
                packed_limit=1000,
                branch_budget=512,
                max_branches=256,
            )
        with self.assertRaises(ValueError):
            check_tree(
                1,
                [513],
                path_limit=1000,
                packed_limit=1000,
                branch_budget=512,
                max_branches=256,
            )

    def test_probability_validation_rejects_nan_and_wrong_score_expectation(self):
        response = {
            "answers": {
                "arbitrary": {
                    "type": "score",
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                    "confidence": float("nan"),
                    "score": 1.5,
                    "legend": {"0": "none", "1": "some", "2": "strong"},
                }
            },
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        with self.assertRaises(RuntimeError):
            validate_response(response, QUESTION)
        response["answers"]["arbitrary"]["confidence"] = 0.6
        response["answers"]["arbitrary"]["score"] = 0.5
        with self.assertRaises(RuntimeError):
            validate_response(response, QUESTION)

    def test_jsonl_keeps_single_runtime_and_closes_on_eof(self):
        from d1_exl3 import cli

        class Fake:
            def __init__(self, config):
                self.calls = 0
                self.closed = False

            def system_one_batch(self, requests):
                self.calls += 1
                return [
                    {
                        "answers": {
                            name: {"type": "noul", "noul": 0.7} for name in r[1]
                        },
                        "usage": {"input_tokens": 2, "output_tokens": 0},
                    }
                    for r in requests
                ]

            def close(self):
                self.closed = True

            def stats(self):
                return {"native_unload": {"status": "NOT_REQUESTED"}}

        instance = Fake(None)
        stdout = io.StringIO()
        line = (
            json.dumps(
                {
                    "state": "dataset",
                    "questions": {"x": {"type": "noul", "instructions": "Present?"}},
                }
            )
            + "\n"
        )
        with (
            patch.object(cli, "D1Runtime", return_value=instance),
            patch.object(sys, "stdin", io.StringIO(line * 2)),
            contextlib.redirect_stdout(stdout),
        ):
            cli.main(
                [
                    "--model",
                    str(ROOT),
                    "--minimum-free-gib",
                    "2",
                    "--allocator-gib",
                    "1",
                ]
            )
        self.assertEqual(instance.calls, 2)
        self.assertTrue(instance.closed)
        self.assertEqual(
            len([json.loads(v) for v in stdout.getvalue().splitlines()]), 2
        )

    def test_idle_cli_closes_without_ever_loading_a_model(self):
        from d1_exl3 import cli
        from threading import Event

        stop = Event()

        def read(*args):
            stop.wait()
            return ""

        stream = types.SimpleNamespace(readline=read)

        class Fake:
            closed = False

            def close(self):
                self.closed = True

            def stats(self):
                return {"native_unload": {"status": "NOT_REQUESTED"}}

        instance = Fake()
        try:
            with (
                patch.object(cli, "D1Runtime", return_value=instance),
                patch.object(sys, "stdin", stream),
            ):
                cli.main(
                    [
                        "--model",
                        str(ROOT),
                        "--minimum-free-gib",
                        "2",
                        "--allocator-gib",
                        "1",
                        "--idle-seconds",
                        ".01",
                    ]
                )
        finally:
            stop.set()
        self.assertTrue(instance.closed)


if __name__ == "__main__":
    unittest.main()
