#!/usr/bin/env python3
"""Validate a completed paper-scale MMS-IV workflow run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_ARGUMENTS: dict[str, int | float] = {
    "K": 22,
    "Nx_eval": 120,
    "Nb_outer": 420,
    "Nb_inner": 300,
    "Nb_dense": 1600,
    "reduced_rank": 1000,
    "tau_rel": 1e-10,
    "T": 0.90,
    "dt": 0.0015,
    "Du": 0.02,
    "Dv": 0.012,
    "a_param": 0.8,
    "b_param": 0.6,
    "amp_u": 4.0,
    "amp_v": 3.2,
    "omega": 4.6,
    "fit_lam": 1e-11,
    "seed": 1234,
}

EXPECTED_TRAINING_ARGUMENTS: dict[str, int | float | str] = {
    "K": 22,
    "epochs": 80,
    "n_train": 20000,
    "n_test": 4000,
    "batch_size": 16,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "step_lr": 40,
    "gamma": 0.3,
    "init_scaled_diag": 0.1,
    "loss_mode": "sample_relative_mse",
    "loss_eps": 1e-14,
    "seed": 123,
    "dtype": "float64",
}

PAPER_METRICS = {
    "final_rel_l2": 5.77e-3,
    "mean_rel_l2": 6.12e-3,
    "max_rel_l2": 7.70e-3,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required output: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, atol: float = 1e-12) -> bool:
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Directory containing summary.json and table_metrics.json")
    args = parser.parse_args()

    summary = load_json(args.results / "summary.json")
    metrics = load_json(args.results / "table_metrics.json")
    failures: list[str] = []

    run_args = summary.get("args", {})
    for name, expected in EXPECTED_ARGUMENTS.items():
        actual = run_args.get(name)
        if actual is None or not close(float(actual), float(expected)):
            failures.append(f"argument {name}: expected {expected}, found {actual}")

    checkpoint_path = Path(str(run_args.get("laplace_checkpoint", "")))
    if not checkpoint_path.is_file():
        failures.append(f"checkpoint file is not readable: {checkpoint_path}")
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        training_args = checkpoint.get("config", {})
        for name, expected in EXPECTED_TRAINING_ARGUMENTS.items():
            actual = training_args.get(name)
            if isinstance(expected, str):
                matches = actual == expected
            else:
                matches = actual is not None and close(float(actual), float(expected))
            if not matches:
                failures.append(f"checkpoint training argument {name}: expected {expected}, found {actual}")
        if checkpoint.get("K") != 22 or checkpoint.get("M") != 1517:
            failures.append(
                f"checkpoint dimensions: expected K=22 and M=1517, found K={checkpoint.get('K')} and M={checkpoint.get('M')}"
            )
        checkpoint_test_error = checkpoint.get("best_test_rel_l2")
        if checkpoint_test_error is None or not math.isfinite(float(checkpoint_test_error)) or float(checkpoint_test_error) > 1e-8:
            failures.append(
                f"checkpoint held-out relative error: expected <= 1.000e-08, found {checkpoint_test_error}"
            )

    expected_dimensions = {
        "basis_dim": 1517,
        "constraint_rank": 327,
        "raw_null_dimension": 1190,
        "reduced_rank": 832,
    }
    for name, expected in expected_dimensions.items():
        actual = summary.get(name)
        if actual != expected:
            failures.append(f"{name}: expected {expected}, found {actual}")

    upper_bounds = {
        "constraint_residual": (summary.get("constraint_residual"), 1e-10),
        "mass_orthonormality_residual": (summary.get("mass_orthonormality_residual"), 2e-6),
        "final_mixed_bc": (summary.get("final_mixed_bc"), 1e-8),
        "E_boundary_rms": (metrics.get("E_boundary_rms"), 1e-8),
        "E_boundary_sc": (metrics.get("E_boundary_sc"), 1e-6),
    }
    checkpoint_diagnostics = summary.get("laplace_checkpoint_diagnostics", {})
    upper_bounds["laplace_relative_error"] = (checkpoint_diagnostics.get("relative_error_to_exact"), 1e-5)
    upper_bounds["laplace_off_diagonal_norm"] = (checkpoint_diagnostics.get("off_diagonal_norm"), 1e-8)

    for name, (actual, limit) in upper_bounds.items():
        if actual is None or not math.isfinite(float(actual)) or float(actual) > limit:
            failures.append(f"{name}: expected a finite value <= {limit:.3e}, found {actual}")

    error_tolerances = {
        "final_rel_l2": 2e-4,
        "mean_rel_l2": 1e-4,
        "max_rel_l2": 2e-4,
    }
    for name, paper_value in PAPER_METRICS.items():
        actual = metrics.get(name)
        tolerance = error_tolerances[name]
        if actual is None or not math.isfinite(float(actual)) or abs(float(actual) - paper_value) > tolerance:
            failures.append(
                f"{name}: expected {paper_value:.3e} +/- {tolerance:.1e}, found {actual}"
            )

    if failures:
        print("MMS-IV workflow verification: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("MMS-IV workflow verification: PASS")
    print("  boundary construction: M=1517, rank(C)=327, null=1190, retained r=832")
    print(
        "  relative L2 errors: "
        f"final={float(metrics['final_rel_l2']):.6e}, "
        f"mean={float(metrics['mean_rel_l2']):.6e}, "
        f"max={float(metrics['max_rel_l2']):.6e}"
    )
    print(
        "  boundary residuals: "
        f"rms={float(metrics['E_boundary_rms']):.6e}, "
        f"scaled={float(metrics['E_boundary_sc']):.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
