"""Pin explicitly selected D1 source metadata and checkpoint without loading a model."""

import argparse
import json
from pathlib import Path
import tempfile

from adapter import REVISION, admit_source, sha, validate_metadata_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-sha256",
        required=True,
        help="Expected full checkpoint SHA256 from a trusted source",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Explicit metadata basename; repeat as needed",
    )
    args = parser.parse_args()
    try:
        for name in args.metadata:
            validate_metadata_name(name)
        pins = {name: sha(args.source / name) for name in args.metadata}
        receipt = {
            "model_revision": REVISION,
            "model_sha256": args.model_sha256,
            "metadata_sha256": pins,
        }
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "manifest.json"
            temporary.write_text(json.dumps(receipt))
            admit_source(args.source, temporary)
        if sha(args.source / "model.safetensors") != args.model_sha256:
            raise ValueError("Source checkpoint SHA differs from the expected digest")
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
