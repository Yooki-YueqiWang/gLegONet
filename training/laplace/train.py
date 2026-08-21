#!/usr/bin/env python3
"""
Train and verify the reusable ambient Laplace block on Q=[-1,1]^2.

This script implements the dissipative diagonal block described in the TeX
protocol:

    B_Delta(a) ~= Delta(Phi a),
    Delta phi_{k,l} = -pi^2 (k^2+l^2) phi_{k,l},

using the real trigonometric basis with radial cutoff k^2+l^2 <= K^2.

Outputs
-------
- config.json
- model_state.pt
- history.csv
- history.npz
- spectrum.csv
- laplace_dissipative_error_curve.png
- laplace_spectrum_compare.png
- laplace_dissipation_verify.png

Inspect the complete parameter interface with ``python train.py --help``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Basis utilities
# -----------------------------------------------------------------------------


def inv_softplus(
    y: torch.Tensor | float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Numerically stable inverse of softplus for positive y."""
    y_t = torch.as_tensor(y, dtype=dtype, device=device)
    return y_t + torch.log(-torch.expm1(-y_t))


def make_real_trig_basis_metadata(K: int) -> Dict[str, np.ndarray]:
    """Return metadata for an orthonormal real trigonometric basis on [-1,1]^2.

    We use the normalized mean inner product

        <f,g>_Q = (1/|Q|) int_Q f g dx,

    so the real basis is

        phi_0 = 1,
        sqrt(2) cos(pi(kx+ly)), sqrt(2) sin(pi(kx+ly))

    for one representative of each nonzero conjugate pair.  The representative
    half plane is k>0 or k=0,l>0.
    """
    entries: List[Tuple[int, int, str]] = [(0, 0, "const")]
    pairs: List[Tuple[int, int]] = []
    for k in range(-K, K + 1):
        for ell in range(-K, K + 1):
            if k == 0 and ell == 0:
                continue
            if k * k + ell * ell <= K * K and (k > 0 or (k == 0 and ell > 0)):
                pairs.append((k, ell))
    pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))

    for k, ell in pairs:
        entries.append((k, ell, "cos"))
        entries.append((k, ell, "sin"))

    k_arr = np.array([e[0] for e in entries], dtype=np.int64)
    ell_arr = np.array([e[1] for e in entries], dtype=np.int64)
    kind = np.array([e[2] for e in entries])
    r = np.sqrt(k_arr.astype(np.float64) ** 2 + ell_arr.astype(np.float64) ** 2)
    lambdas = (math.pi ** 2) * (r ** 2)
    sigmas = (1.0 + r) ** (-0.5)

    return {
        "k": k_arr,
        "ell": ell_arr,
        "kind": kind,
        "lambda": lambdas.astype(np.float32),
        "sigma": sigmas.astype(np.float32),
    }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class PositiveDiagonalLaplace(nn.Module):
    """Positive diagonal dissipative Laplacian in scaled coefficient variables.

    The learned operator is

        B(a) = - scale * d_theta * a,
        d_theta = softplus(raw_diag) * nonzero_mask.

    The training target is Delta a / scale = -(lambda/scale) * a.
    """

    def __init__(self, lambda_vec: torch.Tensor, init_scaled_diag: float = 0.10):
        super().__init__()
        if lambda_vec.ndim != 1:
            raise ValueError("lambda_vec must be one-dimensional")
        self.register_buffer("lambda_vec", lambda_vec.clone())
        scale = float(torch.max(lambda_vec).item())
        if scale <= 0.0:
            raise ValueError("The Laplace scale must be positive")
        self.register_buffer("scale", lambda_vec.new_tensor(scale))
        self.register_buffer("nonzero_mask", (lambda_vec > 0).to(dtype=lambda_vec.dtype))

        raw0 = inv_softplus(
            init_scaled_diag,
            dtype=lambda_vec.dtype,
            device=lambda_vec.device,
        ).expand_as(lambda_vec).clone()
        self.raw_diag = nn.Parameter(raw0)

    def scaled_diag(self) -> torch.Tensor:
        return F.softplus(self.raw_diag) * self.nonzero_mask

    def physical_diag(self) -> torch.Tensor:
        return self.scale * self.scaled_diag()

    def forward_scaled(self, a: torch.Tensor) -> torch.Tensor:
        return -self.scaled_diag().unsqueeze(0) * a

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.scale * self.forward_scaled(a)


# -----------------------------------------------------------------------------
# Training and verification
# -----------------------------------------------------------------------------


def sample_coefficients(batch_size: int, sigma_vec: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, sigma_vec.numel(), device=device) * sigma_vec.unsqueeze(0)


def laplace_training_loss(
    model: PositiveDiagonalLaplace,
    lambda_scaled: torch.Tensor,
    coefficients: torch.Tensor,
    loss_mode: str,
    loss_eps: float,
) -> torch.Tensor:
    if loss_mode == "sample_mse_scaled":
        pred_scaled = model.forward_scaled(coefficients)
        true_scaled = -lambda_scaled.unsqueeze(0) * coefficients
        return torch.mean((pred_scaled - true_scaled) ** 2)

    if loss_mode == "sample_relative_mse":
        pred = model(coefficients)
        true = -model.lambda_vec.unsqueeze(0) * coefficients
        numerator = torch.sum((pred - true) ** 2, dim=1)
        denominator = torch.sum(true ** 2, dim=1) + loss_eps
        return torch.mean(numerator / denominator)

    if loss_mode == "spectrum_log_mse":
        mask = model.lambda_vec > 0
        pred = torch.clamp(model.scaled_diag()[mask], min=loss_eps)
        true = torch.clamp(lambda_scaled[mask], min=loss_eps)
        return torch.mean((torch.log(pred) - torch.log(true)) ** 2)

    if loss_mode == "spectrum_rel_mse":
        mask = model.lambda_vec > 0
        pred = model.scaled_diag()[mask]
        true = lambda_scaled[mask]
        rel = (pred - true) / torch.clamp(true, min=loss_eps)
        return torch.mean(rel ** 2)

    raise ValueError(f"Unsupported loss_mode={loss_mode!r}")


@torch.no_grad()
def evaluate_laplace(
    model: PositiveDiagonalLaplace,
    test_coefficients: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    lambda_vec = model.lambda_vec.to(test_coefficients.device)
    denom_sq = 0.0
    err_sq = 0.0
    rate_denom_sq = 0.0
    rate_err_sq = 0.0
    n_violate = 0
    n_seen = 0

    for start in range(0, len(test_coefficients), batch_size):
        a = test_coefficients[start:start + batch_size]
        bsz = len(a)
        pred = model(a)
        true = -lambda_vec.unsqueeze(0) * a

        err_sq += torch.sum((pred - true) ** 2).item()
        denom_sq += torch.sum(true ** 2).item()

        rate_pred = torch.sum(a * pred, dim=1)
        rate_true = torch.sum(a * true, dim=1)
        rate_err_sq += torch.sum((rate_pred - rate_true) ** 2).item()
        rate_denom_sq += torch.sum(rate_true ** 2).item()
        n_violate += int(torch.sum(rate_pred > 1.0e-10).item())
        n_seen += bsz

    diag = model.physical_diag()
    lambda_nonzero = lambda_vec[lambda_vec > 0]
    diag_nonzero = diag[lambda_vec > 0]
    spectrum_rel = torch.linalg.norm(diag_nonzero - lambda_nonzero) / torch.linalg.norm(lambda_nonzero)

    return {
        "test_rel_l2": math.sqrt(err_sq / max(denom_sq, 1.0e-30)),
        "dissipation_rate_rel_error": math.sqrt(rate_err_sq / max(rate_denom_sq, 1.0e-30)),
        "dissipation_violation_fraction": n_violate / max(n_seen, 1),
        "spectrum_rel_error": float(spectrum_rel.item()),
    }


def save_history_csv(path: str, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_history(outdir: str, history: List[Dict[str, float]]) -> None:
    epochs = np.array([h["epoch"] for h in history])
    test_rel = np.array([h["test_rel_l2"] for h in history])
    spectrum_rel = np.array([h["spectrum_rel_error"] for h in history])
    diss_rate = np.array([h["dissipation_rate_rel_error"] for h in history])
    violation = np.array([h["dissipation_violation_fraction"] for h in history])

    plt.figure(figsize=(7.0, 4.5))
    plt.semilogy(epochs, test_rel, marker="o", markersize=3, label="operator rel-L2")
    plt.semilogy(epochs, spectrum_rel, marker="s", markersize=3, label="spectrum rel-L2")
    plt.semilogy(epochs, diss_rate, marker="^", markersize=3, label="dissipation-rate rel error")
    plt.xlabel("epoch")
    plt.ylabel("relative error")
    plt.title("Laplace block: dissipative error curve")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "laplace_dissipative_error_curve.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.semilogy(epochs, np.maximum(violation, 1.0e-16), marker="o", markersize=3)
    plt.xlabel("epoch")
    plt.ylabel("fraction of samples with <a,B(a)> > 0")
    plt.title("Laplace block: dissipativity violations")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "laplace_dissipation_verify.png"), dpi=220)
    plt.close()


def plot_spectrum(outdir: str, lambda_vec: np.ndarray, learned_diag: np.ndarray) -> None:
    idx = np.argsort(lambda_vec)
    lam = lambda_vec[idx]
    diag = learned_diag[idx]
    nonzero = lam > 0

    plt.figure(figsize=(7.0, 4.5))
    plt.semilogy(np.arange(np.sum(nonzero)), lam[nonzero], label="exact")
    plt.semilogy(np.arange(np.sum(nonzero)), diag[nonzero], linestyle="--", label="learned")
    plt.xlabel("sorted nonzero mode index")
    plt.ylabel(r"$\pi^2(k^2+\ell^2)$")
    plt.title("Laplace spectrum: exact vs learned")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "laplace_spectrum_compare.png"), dpi=220)
    plt.close()


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = make_real_trig_basis_metadata(args.K)
    M = len(meta["lambda"])
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    lambda_vec = torch.tensor(meta["lambda"], device=device, dtype=dtype)
    sigma_vec = torch.tensor(meta["sigma"], device=device, dtype=dtype)
    lambda_scaled = lambda_vec / torch.max(lambda_vec)
    train_coefficients = sample_coefficients(args.n_train, sigma_vec, device)
    test_coefficients = sample_coefficients(args.n_test, sigma_vec, device)

    model = PositiveDiagonalLaplace(lambda_vec, init_scaled_diag=args.init_scaled_diag).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_lr, gamma=args.gamma)

    config = vars(args).copy()
    config.update({
        "basis": "real trigonometric, mean-orthonormal on [-1,1]^2",
        "K": args.K,
        "M": M,
        "laplace_eigenvalue": "pi^2*(k^2+ell^2)",
        "training_prior_sigma": "(1+sqrt(k^2+ell^2))^(-1/2)",
    })
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    history: List[Dict[str, float]] = []
    best_rel = float("inf")
    best_state = None
    t0 = time.time()
    steps_per_epoch = max(1, math.ceil(args.n_train / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        permutation = torch.randperm(args.n_train, device=device)
        for start in range(0, args.n_train, args.batch_size):
            coefficients = train_coefficients[permutation[start:start + args.batch_size]]
            loss = laplace_training_loss(
                model=model,
                lambda_scaled=lambda_scaled,
                coefficients=coefficients,
                loss_mode=args.loss_mode,
                loss_eps=args.loss_eps,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())

        scheduler.step()
        metrics = evaluate_laplace(model, test_coefficients, args.batch_size)
        row = {
            "epoch": epoch,
            "train_mse_scaled": loss_sum / steps_per_epoch,
            "lr": opt.param_groups[0]["lr"],
            **metrics,
            "seconds": time.time() - t0,
        }
        history.append(row)

        if metrics["test_rel_l2"] < best_rel:
            best_rel = metrics["test_rel_l2"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "model_state": best_state,
                "K": args.K,
                "M": M,
                "lambda": meta["lambda"],
                "sigma": meta["sigma"],
                "config": config,
                "best_epoch": epoch,
                "best_test_rel_l2": best_rel,
            }, os.path.join(args.outdir, "model_state.pt"))

        print(
            f"[epoch {epoch:04d}] "
            f"train_mse={row['train_mse_scaled']:.3e} "
            f"test_rel={row['test_rel_l2']:.3e} "
            f"spec_rel={row['spectrum_rel_error']:.3e} "
            f"diss_rate_rel={row['dissipation_rate_rel_error']:.3e} "
            f"viol={row['dissipation_violation_fraction']:.1e}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    learned_diag = model.physical_diag().detach().cpu().numpy()
    spectrum_rows = []
    for i in range(M):
        spectrum_rows.append({
            "index": i,
            "k": int(meta["k"][i]),
            "ell": int(meta["ell"][i]),
            "kind": str(meta["kind"][i]),
            "lambda_exact": float(meta["lambda"][i]),
            "lambda_learned": float(learned_diag[i]),
            "abs_error": float(abs(learned_diag[i] - meta["lambda"][i])),
        })
    save_history_csv(os.path.join(args.outdir, "history.csv"), history)
    np.savez(os.path.join(args.outdir, "history.npz"), **{k: np.array([h[k] for h in history]) for k in history[0]})
    save_history_csv(os.path.join(args.outdir, "spectrum.csv"), spectrum_rows)
    plot_history(args.outdir, history)
    plot_spectrum(args.outdir, meta["lambda"], learned_diag)

    final_metrics = evaluate_laplace(model, test_coefficients, args.batch_size)
    with open(os.path.join(args.outdir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("[done] saved results to", args.outdir)
    print(json.dumps(final_metrics, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ambient dissipative Laplace block on Q=[-1,1]^2")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-train", type=int, required=True)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight-decay", type=float, required=True)
    p.add_argument("--step-lr", type=int, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--init-scaled-diag", type=float, required=True)
    p.add_argument(
        "--loss-mode",
        type=str,
        required=True,
        choices=["sample_relative_mse", "sample_mse_scaled", "spectrum_log_mse", "spectrum_rel_mse"],
    )
    p.add_argument("--loss-eps", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", type=str, required=True, choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
