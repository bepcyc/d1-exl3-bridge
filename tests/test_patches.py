"""Check source identity and exact patch application; never import Torch/transformers."""

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PatchTests(unittest.TestCase):
    def test_upstream_metadata_hashes_and_patch_output_ast(self):
        manifest = json.loads((ROOT / "upstream/manifest.json").read_text())
        expected_asts = json.loads(
            (ROOT / "patches/historical-ast-sha256.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, entry in manifest["files"].items():
                self.assertEqual(
                    hashlib.sha256((ROOT / "upstream" / name).read_bytes()).hexdigest(),
                    entry["sha256"],
                )
            for name in ["hybrid.py", "lfm2_vl.py"]:
                target = Path(directory) / name
                shutil.copyfile(ROOT / "upstream" / name, target)
                result = subprocess.run(
                    [
                        "patch",
                        "--batch",
                        "--fuzz=0",
                        "-p1",
                        "-i",
                        str(ROOT / "patches" / (name + ".patch")),
                    ],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                tree = ast.dump(ast.parse(target.read_text()))
                self.assertEqual(
                    hashlib.sha256(tree.encode()).hexdigest(), expected_asts[name]
                )
                self.assertIn("Modified by Viacheslav Rodionov", target.read_text())


if __name__ == "__main__":
    unittest.main()
