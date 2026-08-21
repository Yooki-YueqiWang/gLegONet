from __future__ import annotations

import json
import re
import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "paper"


class ReleaseContractTests(unittest.TestCase):
    def load(self, name: str) -> dict:
        return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))

    def test_all_config_entrypoints_exist_and_dry_run(self) -> None:
        configs = sorted(CONFIG_DIR.glob("*.json"))
        self.assertGreaterEqual(len(configs), 11)
        for path in configs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertTrue((ROOT / payload["entrypoint"]).is_file(), path.name)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_config.py"), "--config", str(path), "--dry-run"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_config_keys_are_declared_cli_options(self) -> None:
        for path in sorted(CONFIG_DIR.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = (ROOT / payload["entrypoint"]).read_text(encoding="utf-8")
            tree = ast.parse(source)
            options: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "add_argument":
                    continue
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        if argument.value.startswith("--"):
                            options.add(argument.value)
            configured = {f"--{key}" for key in payload["arguments"]}
            self.assertEqual(configured - options, set(), path.name)

    def test_figure3_k_boundary_counts_and_ranks(self) -> None:
        expected = {
            "mms_01_rosette.json": (420, 207),
            "mms_02_crescent.json": (620, 220),
            "mms_03_bunny.json": (420, 843),
            "mms_04_annular_star.json": (720, 832),
            "mms_05_pinwheel.json": (420, 644),
        }
        for name, (boundary_count, rank) in expected.items():
            args = self.load(name)["arguments"]
            count = args.get("Nb", args.get("Nb_outer", 0) + args.get("Nb_inner", 0))
            self.assertEqual(args["K"], 22, name)
            self.assertEqual(count, boundary_count, name)
            self.assertEqual(args["reduced_rank"], rank, name)

    def test_inverse_protocol_matches_manuscript(self) -> None:
        for name in ("inverse_peanut.json", "inverse_channel.json"):
            args = self.load(name)["arguments"]
            self.assertEqual(args["K"], 22)
            self.assertEqual(args["rank"], 80)
            self.assertEqual(args["n_pairs"], 16)
            self.assertEqual(args["T_id"], 0.012)
            self.assertEqual(args["dt_obs"], 5e-5)
            self.assertEqual(args["obs_list"], [240, 1000])
            self.assertEqual(args["coeff_ridge"], 1e-10)
            self.assertEqual(args["threshold"], 2e-3)
            self.assertEqual(args["max_threshold_iterations"], 5)

    def test_repository_text_contains_no_chinese_characters(self) -> None:
        han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
        suffixes = {".py", ".md", ".json", ".yml", ".yaml", ".cff", ".txt"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in {".git", "artifacts", "outputs", "checkpoints"} for part in path.parts):
                continue
            self.assertIsNone(han.search(path.read_text(encoding="utf-8")), str(path))

    def test_no_checkpoint_is_part_of_release_tree(self) -> None:
        ignored_roots = {".git", "checkpoints", "artifacts", "outputs"}
        checkpoint_files = [
            path for path in ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
            and not any(part in ignored_roots for part in path.parts)
        ]
        self.assertEqual(checkpoint_files, [])


if __name__ == "__main__":
    unittest.main()
