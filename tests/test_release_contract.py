from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_ENTRYPOINTS = {
    "training/laplace/train.py": {
        "--K", "--outdir", "--epochs", "--n-train", "--n-test", "--batch-size", "--lr",
        "--weight-decay", "--step-lr", "--gamma", "--init-scaled-diag", "--loss-mode",
        "--loss-eps", "--seed", "--device"
    },
    "training/transport/train.py": {
        "--K", "--n-grid", "--width", "--depth", "--act", "--epochs", "--n-train",
        "--n-test", "--batch-size", "--lr", "--outdir", "--weight-decay", "--step-size",
        "--gamma", "--u-max", "--u-small-max", "--small-frac", "--lambda-h", "--lambda-g",
        "--lambda-g-rel", "--lambda-second", "--lambda-zero", "--eval-every", "--device", "--seed"
    },
    "experiments/manufactured_solutions/mms_01_rosette.py": {
        "--K", "--Nb", "--reduced_rank", "--T", "--dt", "--laplace_checkpoint"
    },
    "experiments/manufactured_solutions/mms_02_crescent.py": {
        "--K", "--Nb_outer", "--Nb_inner", "--reduced_rank", "--T", "--dt", "--laplace_checkpoint"
    },
    "experiments/manufactured_solutions/mms_03_bunny.py": {
        "--K", "--Nb", "--Nb_lift", "--lift_solver", "--reduced_rank", "--T", "--dt", "--laplace_checkpoint"
    },
    "experiments/manufactured_solutions/mms_04_annular_star.py": {
        "--K", "--Nb_outer", "--Nb_inner", "--reduced_rank", "--T", "--dt", "--laplace_checkpoint"
    },
    "experiments/manufactured_solutions/mms_05_pinwheel.py": {
        "--K", "--Nb", "--reduced_rank", "--T", "--dt", "--laplace_checkpoint", "--transport_checkpoint"
    },
    "experiments/allen_cahn/run.py": {
        "--K", "--Nb_build", "--reduced_rank", "--T", "--dt", "--laplace_block_path", "--laplace_block_sign"
    },
    "experiments/burgers/run.py": {
        "--K", "--Nb_build", "--rank", "--T", "--dt", "--laplace_ckpt", "--burgers_ckpt", "--ref_N"
    },
    "experiments/burgers/model.py": {
        "--K", "--Nb_build", "--rank", "--T", "--dt", "--laplace_ckpt", "--burgers_ckpt"
    },
    "experiments/inverse_discovery/run.py": {
        "--shape", "--K", "--rank", "--Nx", "--Nb", "--n_pairs", "--T_id", "--T_roll",
        "--dt_obs", "--dt_solver", "--obs_list", "--coeff_ridge", "--threshold",
        "--max_threshold_iterations", "--active_tol", "--laplace_checkpoint", "--transport_checkpoint",
        "--obs_mode", "--n_repeats", "--reference_mode", "--baseline_modes", "--skip_projection_diagnostic"
    },
}


def declared_options(path: Path) -> dict[str, bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    options: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if not first.value.startswith("--"):
            continue
        required = any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        options[first.value] = required
    return options


class ReleaseContractTests(unittest.TestCase):
    def test_public_entrypoints_exist(self) -> None:
        for relative_path in PUBLIC_ENTRYPOINTS:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_scientific_inputs_are_required_cli_options(self) -> None:
        for relative_path, expected_required in PUBLIC_ENTRYPOINTS.items():
            options = declared_options(ROOT / relative_path)
            self.assertEqual(expected_required - options.keys(), set(), relative_path)
            not_required = {option for option in expected_required if not options[option]}
            self.assertEqual(not_required, set(), relative_path)

    def test_no_bundled_parameter_configuration(self) -> None:
        self.assertFalse((ROOT / "config" "s").exists())
        self.assertFalse((ROOT / "scripts" / ("run_" "config.py")).exists())

        forbidden = ("configs/" "paper", "scripts/" "run_config.py")
        suffixes = {".py", ".md", ".yml", ".yaml", ".cff", ".txt"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if ".git" in path.parts:
                continue
            text = path.read_text(encoding="utf-8").replace("\\", "/")
            for phrase in forbidden:
                self.assertNotIn(phrase, text, str(path))

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
