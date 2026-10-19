"""Resident JSONL frontend; typed D1 decisions, no chat/generation substitution."""

import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

from .config import RuntimeConfig
from .runtime import D1Runtime

MAX_LINE = 32 * 1024**2
MAX_IMAGE = 16 * 1024**2


def read_image(spec):
    from PIL import Image, ImageOps

    if not isinstance(spec, dict) or len(spec) != 1 or set(spec) - {"path", "base64"}:
        raise ValueError("An image must specify exactly one path or base64 value")
    if "path" in spec:
        with Path(spec["path"]).open("rb") as stream:
            payload = stream.read(MAX_IMAGE + 1)
    else:
        if not isinstance(spec["base64"], str) or len(spec["base64"]) > 4 * (
            (MAX_IMAGE + 2) // 3
        ):
            raise ValueError("Encoded image exceeds the input limit")
        payload = base64.b64decode(spec["base64"], validate=True)
    if len(payload) > MAX_IMAGE:
        raise ValueError("Image exceeds the input byte limit")
    with Image.open(io.BytesIO(payload)) as image:
        if (
            getattr(image, "n_frames", 1) != 1
            or image.width * image.height > 128 * 1024**2
        ):
            raise ValueError("A bounded single-frame image is required")
        return ImageOps.exif_transpose(image).convert("RGB")


def request_rows(value, owned):
    rows = value if isinstance(value, list) else [value]
    if not 1 <= len(rows) <= 256:
        raise ValueError("A JSONL batch must contain1..256requests")
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"state", "questions", "images"}:
            raise ValueError("Request keys are state, questions and optional images")
        images = [read_image(item) for item in row.get("images", [])]
        owned.extend(images)
        result.append((row.get("state"), row["questions"], images))
    return result, isinstance(value, list)


def lines(stream):
    queue = Queue(maxsize=1)

    def read():
        try:
            while True:
                row = stream.readline(MAX_LINE + 1)
                queue.put(row or None)
                if not row:
                    return
        except (OSError, ValueError) as exc:
            queue.put(exc)

    Thread(target=read, daemon=True, name="d1-jsonl-input").start()
    return queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-gib", type=float, required=True)
    parser.add_argument("--allocator-gib", type=float, required=True)
    parser.add_argument("--token-budget", type=int, default=65536)
    parser.add_argument("--idle-seconds", type=float, default=300)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--model-sha256")
    parser.add_argument("--maximum-tiles", type=int, default=11)
    parser.add_argument("--maximum-branches", type=int, default=256)
    parser.add_argument("--maximum-path-tokens", type=int)
    parser.add_argument("--maximum-packed-tokens", type=int, default=98048)
    args = parser.parse_args(argv)
    if args.idle_seconds <= 0:
        parser.error("--idle-seconds must be positive")
    config = RuntimeConfig(
        args.model,
        int(args.minimum_free_gib * 1024**3),
        int(args.allocator_gib * 1024**3),
        args.device,
        args.token_budget,
        maximum_vision_tiles=args.maximum_tiles,
        maximum_branches=args.maximum_branches,
        expected_model_sha256=args.model_sha256,
        maximum_path_tokens=args.maximum_path_tokens,
        maximum_packed_tokens=args.maximum_packed_tokens,
    )
    runtime = D1Runtime(config)
    queue = lines(getattr(sys.stdin, "buffer", sys.stdin))
    from queue import Empty

    try:
        while True:
            try:
                raw = queue.get(timeout=args.idle_seconds)
            except Empty:
                break
            if raw is None:
                break
            if isinstance(raw, Exception):
                raise raw
            if len(raw) > MAX_LINE:
                raise ValueError("JSONL line exceeds the input limit")
            value = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("Nonfinite JSON is not allowed")
                ),
            )
            owned = []
            try:
                requests, is_batch = request_rows(value, owned)
                with contextlib.redirect_stdout(sys.stderr):
                    responses = runtime.system_one_batch(requests)
                print(
                    json.dumps(
                        responses if is_batch else responses[0], allow_nan=False
                    ),
                    flush=True,
                )
            finally:
                for image in owned:
                    image.close()
    finally:
        runtime.close()
        if args.stats:
            print(json.dumps(runtime.stats(), allow_nan=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
