#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the Burgers transport density by local Sobolev matching.

Goal
----
For conservative Burgers transport, the transferable object is the local
primitive density h(u), not a square-domain nonlinear coefficient map.  This
script trains a plain MLP density h_theta by the one-dimensional Sobolev target

    h_theta(u)  ~= u^3 / 6,
    h_theta'(u) ~= u^2 / 2,

on a prescribed range of point values.  The learned density can then be
re-integrated on any embedded domain Omega in a Section-3 weak/Galerkin transfer.

The script also reports ambient K=22 diagnostics, but these are diagnostics only:

    H_theta(a) = mean_Q h_theta(Phi a),
    P_theta(a) = grad_a H_theta(a),
    P_theta(a) ~= Pi_K(0.5*(Phi a)^2),
    Bx = Jx P_theta,  By = Jy P_theta.

The saved checkpoint uses a plain DensityNet state_dict compatible with existing
loaders that instantiate DensityNet(width, depth, act).

This fixed version avoids copying stale diagnostics into non-evaluation epochs and prints separate train/eval lines,
uses stronger small-signal weighting by default, and treats ambient coefficient
errors as diagnostics on the state amplitude actually relevant to the MMS rollout.

Recommended Windows PowerShell command:

python train_burgers_local_density_K22_plain_mlp.py `
  --K 22 `
  --n-grid 96 `
  --epochs 3000 `
  --steps-per-epoch 200 `
  --batch-size 8192 `
  --device cuda `
  --u-max 1.0 `
  --u-small-max 0.35 `
  --small-frac 0.70 `
  --lambda-g 1.0 `
  --lambda-g-rel 1.0 `
  --lambda-h 0.05 `
  --lambda-zero 1.0 `
  --outdir runs_ambient_burgers_K22_local_sobolev_plain
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    def conv(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: conv(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [conv(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv(payload), f, indent=2)


def save_history_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# K=22 real trigonometric ambient basis for diagnostics
# -----------------------------------------------------------------------------


def make_real_trig_basis_metadata(K: int) -> Dict[str, Any]:
    entries: List[Tuple[int, int, str]] = [(0, 0, "const")]
    pairs: List[Tuple[int, int]] = []
    for k in range(-K, K + 1):
        for ell in range(-K, K + 1):
            if k == 0 and ell == 0:
                continue
            if k * k + ell * ell <= K * K and (k > 0 or (k == 0 and ell > 0)):
                pairs.append((k, ell))
    pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))

    pair_indices: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for k, ell in pairs:
        cidx = len(entries)
        entries.append((k, ell, "cos"))
        sidx = len(entries)
        entries.append((k, ell, "sin"))
        pair_indices[(k, ell)] = (cidx, sidx)

    k_arr = np.array([e[0] for e in entries], dtype=np.int64)
    ell_arr = np.array([e[1] for e in entries], dtype=np.int64)
    kind_arr = np.array([e[2] for e in entries])
    r = np.sqrt(k_arr.astype(np.float64) ** 2 + ell_arr.astype(np.float64) ** 2)
    return {
        "entries": entries,
        "pairs": pairs,
        "pair_indices": pair_indices,
        "k": k_arr,
        "ell": ell_arr,
        "kind": kind_arr,
        "r": r.astype(np.float64),
        "sigma": ((1.0 + r) ** (-0.5)).astype(np.float32),
        "lambda": (math.pi ** 2 * r ** 2).astype(np.float32),
    }


def evaluate_basis_on_grid(meta: Dict[str, Any], n_grid: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, n_grid + 1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(-1.0, 1.0, n_grid + 1, device=device, dtype=dtype)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    cols: List[torch.Tensor] = []
    sqrt2 = math.sqrt(2.0)
    for k, ell, kind in zip(meta["k"], meta["ell"], meta["kind"]):
        if kind == "const":
            cols.append(torch.ones_like(Xf))
        else:
            phase = math.pi * (float(k) * Xf + float(ell) * Yf)
            if kind == "cos":
                cols.append(sqrt2 * torch.cos(phase))
            elif kind == "sin":
                cols.append(sqrt2 * torch.sin(phase))
            else:
                raise ValueError(kind)
    return torch.stack(cols, dim=1).contiguous()


def assemble_derivative_matrices(meta: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    M = len(meta["k"])
    Jx = torch.zeros(M, M, dtype=dtype, device=device)
    Jy = torch.zeros(M, M, dtype=dtype, device=device)
    for (k, ell), (cidx, sidx) in meta["pair_indices"].items():
        # Row-batch convention: out = coeff @ J.T; J=-partial.
        Jx[sidx, cidx] = math.pi * float(k)
        Jx[cidx, sidx] = -math.pi * float(k)
        Jy[sidx, cidx] = math.pi * float(ell)
        Jy[cidx, sidx] = -math.pi * float(ell)
    return Jx, Jy


# -----------------------------------------------------------------------------
# Plain density network
# -----------------------------------------------------------------------------


class DensityNet(nn.Module):
    """Plain pointwise Hamiltonian density h_theta(u)."""

    def __init__(self, width: int = 128, depth: int = 4, act: str = "gelu", init_last_scale: float = 1.0e-3):
        super().__init__()
        act_l = act.lower()
        if act_l == "gelu":
            Act = nn.GELU
        elif act_l == "silu":
            Act = nn.SiLU
        elif act_l == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"unknown activation: {act}")

        layers: List[nn.Module] = []
        in_features = 1
        for _ in range(depth):
            lin = nn.Linear(in_features, width)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, Act()])
            in_features = width
        last = nn.Linear(width, 1)
        # Small last layer avoids a large constant derivative bias at start.
        nn.init.normal_(last.weight, mean=0.0, std=float(init_last_scale))
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u.unsqueeze(-1)).squeeze(-1)


def density_derivative_values(model: nn.Module, u_values: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    with torch.enable_grad():
        u_req = u_values.detach().clone().requires_grad_(True)
        h = model(u_req)
        g = torch.autograd.grad(h.sum(), u_req, create_graph=create_graph)[0]
    return g


def density_h_g_g2(model: nn.Module, u_values: torch.Tensor, want_second: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    with torch.enable_grad():
        u = u_values.detach().clone().requires_grad_(True)
        h = model(u)
        g = torch.autograd.grad(h.sum(), u, create_graph=True)[0]
        g2 = None
        if want_second:
            g2 = torch.autograd.grad(g.sum(), u, create_graph=True)[0]
    return h, g, g2


# -----------------------------------------------------------------------------
# Local density sampling and loss
# -----------------------------------------------------------------------------


def sample_u_mixture(
    batch_size: int,
    u_max: float,
    u_small_max: float,
    small_frac: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_small = int(round(batch_size * small_frac))
    n_wide = batch_size - n_small
    chunks: List[torch.Tensor] = []
    if n_wide > 0:
        chunks.append((2.0 * torch.rand(n_wide, device=device, dtype=dtype) - 1.0) * float(u_max))
    if n_small > 0:
        # Oversample the small-signal regime where relative errors are most sensitive.
        chunks.append((2.0 * torch.rand(n_small, device=device, dtype=dtype) - 1.0) * float(u_small_max))
    u = torch.cat(chunks, dim=0)
    perm = torch.randperm(u.numel(), device=device)
    return u[perm]


def local_sobolev_loss(
    model: nn.Module,
    u: torch.Tensor,
    lambda_h: float,
    lambda_g: float,
    lambda_g_rel: float,
    lambda_second: float,
    lambda_zero: float,
    rel_floor_u: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    h, g, g2 = density_h_g_g2(model, u, want_second=(lambda_second > 0.0))
    h_true = u ** 3 / 6.0
    g_true = 0.5 * u ** 2

    loss_h = torch.mean((h - h_true) ** 2) / torch.clamp(torch.mean(h_true ** 2), min=1.0e-14)
    loss_g = torch.mean((g - g_true) ** 2) / torch.clamp(torch.mean(g_true ** 2), min=1.0e-14)

    # Relative-style derivative loss with a floor to prevent singular weights at u=0.
    floor = 0.5 * float(rel_floor_u) ** 2
    loss_g_rel = torch.mean(((g - g_true) ** 2) / (g_true ** 2 + floor ** 2))

    if lambda_second > 0.0 and g2 is not None:
        # Since h''(u)=u for the Burgers primitive.
        loss_second = torch.mean((g2 - u) ** 2) / torch.clamp(torch.mean(u ** 2), min=1.0e-14)
    else:
        loss_second = torch.zeros((), dtype=u.dtype, device=u.device)

    u0 = torch.zeros(1, device=u.device, dtype=u.dtype, requires_grad=True)
    h0 = model(u0)
    g0 = torch.autograd.grad(h0.sum(), u0, create_graph=True)[0]
    loss_zero = h0.pow(2).mean() + g0.pow(2).mean()

    loss = (
        float(lambda_h) * loss_h
        + float(lambda_g) * loss_g
        + float(lambda_g_rel) * loss_g_rel
        + float(lambda_second) * loss_second
        + float(lambda_zero) * loss_zero
    )
    metrics = {
        "loss_h": float(loss_h.detach().cpu()),
        "loss_g": float(loss_g.detach().cpu()),
        "loss_g_rel": float(loss_g_rel.detach().cpu()),
        "loss_second": float(loss_second.detach().cpu()),
        "loss_zero": float(loss_zero.detach().cpu()),
    }
    return loss, metrics


def evaluate_pointwise_density(model: nn.Module, u_max: float, n_eval: int, device: torch.device, dtype: torch.dtype, rel_floor_u: float) -> Dict[str, float]:
    model.eval()
    u = torch.linspace(-float(u_max), float(u_max), n_eval, device=device, dtype=dtype)
    return evaluate_pointwise_samples(model, u, rel_floor_u)


def evaluate_pointwise_samples(model: nn.Module, u: torch.Tensor, rel_floor_u: float) -> Dict[str, float]:
    """Evaluate the local density on a fixed held-out sample set."""
    model.eval()
    h, g, g2 = density_h_g_g2(model, u, want_second=True)
    h_true = u ** 3 / 6.0
    g_true = 0.5 * u ** 2
    g2_true = u
    floor = 0.5 * float(rel_floor_u) ** 2
    return {
        "point_h_rel_l2": float((torch.linalg.norm(h - h_true) / torch.clamp(torch.linalg.norm(h_true), min=1.0e-14)).detach().cpu()),
        "point_g_rel_l2": float((torch.linalg.norm(g - g_true) / torch.clamp(torch.linalg.norm(g_true), min=1.0e-14)).detach().cpu()),
        "point_g_rel_floor_rms": float(torch.sqrt(torch.mean(((g - g_true) ** 2) / (g_true ** 2 + floor ** 2))).detach().cpu()),
        "point_g_max_abs": float(torch.max(torch.abs(g - g_true)).detach().cpu()),
        "point_h_max_abs": float(torch.max(torch.abs(h - h_true)).detach().cpu()),
        "point_second_rel_l2": float((torch.linalg.norm(g2 - g2_true) / torch.clamp(torch.linalg.norm(g2_true), min=1.0e-14)).detach().cpu()),
    }


# -----------------------------------------------------------------------------
# Ambient K22 diagnostics only
# -----------------------------------------------------------------------------


def sample_coefficients(batch_size: int, sigma_vec: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(batch_size, sigma_vec.numel(), dtype=dtype, device=device) * sigma_vec.unsqueeze(0)


def project_grid_to_coeff(g_grid: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
    return g_grid @ Phi / Phi.shape[0]


def exact_projected_primitive(a: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
    u = a @ Phi.T
    return project_grid_to_coeff(0.5 * u * u, Phi)


def energy_and_grad_from_density(model: nn.Module, a_in: torch.Tensor, Phi: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        a = a_in.detach().clone().requires_grad_(True)
        u = a @ Phi.T
        H = model(u).mean(dim=1)
        grad_a = torch.autograd.grad(H.sum(), a, create_graph=create_graph)[0]
    return H, grad_a, u


def evaluate_ambient_diagnostics(
    model: nn.Module,
    Phi: torch.Tensor,
    Jx: torch.Tensor,
    Jy: torch.Tensor,
    sigma_vec: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    Pn = Pd = Bxn = Bxd = Byn = Byd = Bsumn = Bsumd = gn = gd = 0.0
    cons_vals: List[float] = []
    for _ in range(n_batches):
        a = sample_coefficients(batch_size, sigma_vec, device, dtype)
        P_true = exact_projected_primitive(a, Phi)
        _, P_pred, u = energy_and_grad_from_density(model, a, Phi, create_graph=False)
        Bx_pred, By_pred = P_pred @ Jx.T, P_pred @ Jy.T
        Bx_true, By_true = P_true @ Jx.T, P_true @ Jy.T
        Bsum_pred, Bsum_true = Bx_pred + By_pred, Bx_true + By_true

        Pn += torch.sum((P_pred - P_true) ** 2).item(); Pd += torch.sum(P_true ** 2).item()
        Bxn += torch.sum((Bx_pred - Bx_true) ** 2).item(); Bxd += torch.sum(Bx_true ** 2).item()
        Byn += torch.sum((By_pred - By_true) ** 2).item(); Byd += torch.sum(By_true ** 2).item()
        Bsumn += torch.sum((Bsum_pred - Bsum_true) ** 2).item(); Bsumd += torch.sum(Bsum_true ** 2).item()

        g_pred = density_derivative_values(model, u.detach(), create_graph=False)
        g_true = 0.5 * u.detach() * u.detach()
        gn += torch.sum((g_pred - g_true) ** 2).item(); gd += torch.sum(g_true ** 2).item()
        cons = torch.abs(torch.sum(a * Bsum_pred, dim=1)) / torch.clamp(torch.linalg.norm(a, dim=1) * torch.linalg.norm(Bsum_pred, dim=1), min=1.0e-30)
        cons_vals.append(float(torch.mean(cons).detach().cpu()))

    return {
        "ambient_primitive_rel_l2": math.sqrt(Pn / max(Pd, 1.0e-30)),
        "ambient_uux_rel_l2": math.sqrt(Bxn / max(Bxd, 1.0e-30)),
        "ambient_uuy_rel_l2": math.sqrt(Byn / max(Byd, 1.0e-30)),
        "ambient_sum_rel_l2": math.sqrt(Bsumn / max(Bsumd, 1.0e-30)),
        "ambient_pointwise_g_rel_l2": math.sqrt(gn / max(gd, 1.0e-30)),
        "ambient_l2_conservation_residual": float(np.mean(cons_vals)) if cons_vals else float("nan"),
    }


def evaluate_small_amplitude_diagnostics(
    model: nn.Module,
    Phi: torch.Tensor,
    Jx: torch.Tensor,
    Jy: torch.Tensor,
    sigma_vec: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int,
    batch_size: int,
    target_min: float,
    target_max: float,
) -> Dict[str, float]:
    model.eval()
    Pn = Pd = Bxn = Bxd = Byn = Byd = gn = gd = 0.0
    for _ in range(n_batches):
        a = sample_coefficients(batch_size, sigma_vec, device, dtype)
        with torch.no_grad():
            u0 = a @ Phi.T
            max_abs = torch.amax(torch.abs(u0), dim=1, keepdim=True).clamp_min(1.0e-12)
            target = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(target_min, target_max)
            a = a * (target / max_abs)
            P_true = exact_projected_primitive(a, Phi)
        _, P_pred, u = energy_and_grad_from_density(model, a, Phi, create_graph=False)
        Bx_pred, By_pred = P_pred @ Jx.T, P_pred @ Jy.T
        Bx_true, By_true = P_true @ Jx.T, P_true @ Jy.T
        Pn += torch.sum((P_pred - P_true) ** 2).item(); Pd += torch.sum(P_true ** 2).item()
        Bxn += torch.sum((Bx_pred - Bx_true) ** 2).item(); Bxd += torch.sum(Bx_true ** 2).item()
        Byn += torch.sum((By_pred - By_true) ** 2).item(); Byd += torch.sum(By_true ** 2).item()
        g_pred = density_derivative_values(model, u.detach(), create_graph=False)
        g_true = 0.5 * u.detach() * u.detach()
        gn += torch.sum((g_pred - g_true) ** 2).item(); gd += torch.sum(g_true ** 2).item()
    return {
        "small_primitive_rel_l2": math.sqrt(Pn / max(Pd, 1.0e-30)),
        "small_uux_rel_l2": math.sqrt(Bxn / max(Bxd, 1.0e-30)),
        "small_uuy_rel_l2": math.sqrt(Byn / max(Byd, 1.0e-30)),
        "small_pointwise_g_rel_l2": math.sqrt(gn / max(gd, 1.0e-30)),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_history(outdir: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    ep = np.array([r["epoch"] for r in rows], dtype=float)
    plt.figure(figsize=(7.4, 4.8))
    for key, label in [
        ("point_g_rel_l2", r"pointwise $h_\theta'$"),
        ("point_g_rel_floor_rms", r"relative-floor $h_\theta'$"),
        ("point_h_rel_l2", r"pointwise $h_\theta$"),
        ("point_second_rel_l2", r"pointwise $h_\theta''$"),
    ]:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        plt.semilogy(ep, vals, label=label)
    plt.xlabel("epoch")
    plt.ylabel("relative error")
    plt.title("Local Sobolev density diagnostics")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "local_density_training_errors.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7.4, 4.8))
    for key, label in [
        ("ambient_primitive_rel_l2", r"ambient $P_\theta$"),
        ("ambient_uux_rel_l2", r"ambient $J_xP_\theta$"),
        ("ambient_uuy_rel_l2", r"ambient $J_yP_\theta$"),
        ("small_uux_rel_l2", r"small $J_xP_\theta$"),
        ("small_uuy_rel_l2", r"small $J_yP_\theta$"),
    ]:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        plt.semilogy(ep, vals, label=label)
    plt.xlabel("epoch")
    plt.ylabel("relative error")
    plt.title("Ambient coefficient diagnostics, not training losses")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ambient_diagnostic_errors.png"), dpi=220)
    plt.close()


def plot_density_fit(model: nn.Module, outdir: str, u_max: float, device: torch.device, dtype: torch.dtype) -> None:
    model.eval()
    u = torch.linspace(-float(u_max), float(u_max), 2000, device=device, dtype=dtype)
    h, g, _ = density_h_g_g2(model, u, want_second=False)
    u_np = u.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    g_np = g.detach().cpu().numpy()
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(u_np, h_np, label=r"$h_\theta(u)$")
    plt.plot(u_np, u_np ** 3 / 6.0, "--", label=r"$u^3/6$")
    plt.xlabel("u"); plt.ylabel("density")
    plt.grid(alpha=0.35); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "density_h_fit.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(u_np, g_np, label=r"$h_\theta'(u)$")
    plt.plot(u_np, 0.5 * u_np ** 2, "--", label=r"$u^2/2$")
    plt.xlabel("u"); plt.ylabel("density derivative")
    plt.grid(alpha=0.35); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "density_g_fit.png"), dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    set_seed(args.seed)
    dtype = torch.float64 if args.use_double else torch.float32
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    meta = make_real_trig_basis_metadata(args.K)
    M = len(meta["k"])
    if args.K == 22 and M != 1517:
        raise RuntimeError(f"For K=22 expected M=1517, got M={M}")
    if args.n_grid < math.ceil(3.0 * args.K):
        raise ValueError("n_grid should be at least ceil(3K); use 96 for K=22.")

    print(f"[setup] K={args.K}, M={M}, n_grid={args.n_grid}, dtype={dtype}, device={device}")
    Phi = evaluate_basis_on_grid(meta, args.n_grid, device, dtype)
    nq = Phi.shape[0]
    with torch.no_grad():
        subset = min(M, 96)
        gram = Phi[:, :subset].T @ Phi[:, :subset] / nq
        gram_err = torch.linalg.norm(gram - torch.eye(subset, dtype=dtype, device=device)) / math.sqrt(subset)
        print(f"[setup] grid orthonormality check first {subset} modes: {gram_err.item():.3e}")
    Jx, Jy = assemble_derivative_matrices(meta, device, dtype)
    print(f"[setup] skew checks: ||Jx+Jx^T||={torch.linalg.norm(Jx+Jx.T).item():.3e}, ||Jy+Jy^T||={torch.linalg.norm(Jy+Jy.T).item():.3e}")

    sigma_base = torch.tensor((1.0 + torch.tensor(meta["r"], dtype=dtype)).numpy(), dtype=dtype, device=device)
    sigma_vec = float(args.amp) * sigma_base.pow(-float(args.alpha))

    model = DensityNet(width=args.width, depth=args.depth, act=args.act, init_last_scale=args.init_last_scale).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    config = vars(args).copy()
    config.update({
        "basis": "real trigonometric basis on Q=[-1,1]^2",
        "M": int(M),
        "density_type": "plain_mlp_local_sobolev",
        "training_style": "local Sobolev density matching: h ~= u^3/6, h' ~= u^2/2; ambient coefficient errors are diagnostics only",
        "density_net": {"width": int(args.width), "depth": int(args.depth), "act": str(args.act)},
        "n_grid": int(args.n_grid),
    })
    save_json(os.path.join(args.outdir, "config.json"), config)

    rows: List[Dict[str, float]] = []
    best_score = float("inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    t0 = time.time()
    print(f"[train] params={sum(p.numel() for p in model.parameters())}")
    train_values = sample_u_mixture(
        args.n_train, args.u_max, args.u_small_max, args.small_frac, device, dtype
    )
    test_values = sample_u_mixture(
        args.n_test, args.u_max, args.u_small_max, args.small_frac, device, dtype
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_acc = 0.0
        mh_acc = mg_acc = mgr_acc = ms_acc = mz_acc = 0.0
        n_steps = 0
        permutation = torch.randperm(args.n_train, device=device)
        for start in range(0, args.n_train, args.batch_size):
            u = train_values[permutation[start:start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = local_sobolev_loss(
                model,
                u,
                lambda_h=args.lambda_h,
                lambda_g=args.lambda_g,
                lambda_g_rel=args.lambda_g_rel,
                lambda_second=args.lambda_second,
                lambda_zero=args.lambda_zero,
                rel_floor_u=args.rel_floor_u,
            )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_acc += float(loss.detach().cpu())
            mh_acc += parts["loss_h"]; mg_acc += parts["loss_g"]; mgr_acc += parts["loss_g_rel"]
            ms_acc += parts["loss_second"]; mz_acc += parts["loss_zero"]
            n_steps += 1
        scheduler.step()

        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        if do_eval:
            point = evaluate_pointwise_samples(model, test_values, args.rel_floor_u)
            amb = evaluate_ambient_diagnostics(model, Phi, Jx, Jy, sigma_vec, device, dtype, args.ambient_eval_batches, args.ambient_eval_batch_size)
            small = evaluate_small_amplitude_diagnostics(model, Phi, Jx, Jy, sigma_vec, device, dtype, args.small_eval_batches, args.ambient_eval_batch_size, args.small_target_min, args.small_target_max)
        else:
            # Do not copy previous evaluation metrics into non-evaluation rows.
            # These epochs still train normally; only expensive diagnostics are skipped.
            point = {}
            amb = {}
            small = {}

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": loss_acc / max(n_steps, 1),
            "train_loss_h": mh_acc / max(n_steps, 1),
            "train_loss_g": mg_acc / max(n_steps, 1),
            "train_loss_g_rel": mgr_acc / max(n_steps, 1),
            "train_loss_second": ms_acc / max(n_steps, 1),
            "train_loss_zero": mz_acc / max(n_steps, 1),
            "seconds": float(time.time() - t0),
        }
        row.update({k: float(v) for k, v in point.items() if isinstance(v, (int, float, np.floating))})
        row.update({k: float(v) for k, v in amb.items() if isinstance(v, (int, float, np.floating))})
        row.update({k: float(v) for k, v in small.items() if isinstance(v, (int, float, np.floating))})
        rows.append(row)

        if do_eval:
            score = row.get("point_g_rel_l2", float("inf")) + 0.25 * row.get("point_g_rel_floor_rms", float("inf")) + 0.05 * row.get("point_h_rel_l2", float("inf"))
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({
                    "model_state": best_state,
                    "K": int(args.K),
                    "M": int(M),
                    "n_grid": int(args.n_grid),
                    "config": config,
                    "best_epoch": int(best_epoch),
                    "best_score": float(best_score),
                    "metrics": {k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
                }, os.path.join(args.outdir, "model_state.pt"))

            print(
                f"[epoch {epoch:04d} eval] "
                f"loss={row['train_loss']:.3e} "
                f"g={row.get('point_g_rel_l2', float('nan')):.3e} "
                f"grel={row.get('point_g_rel_floor_rms', float('nan')):.3e} "
                f"h={row.get('point_h_rel_l2', float('nan')):.3e} "
                f"Pdiag={row.get('ambient_primitive_rel_l2', float('nan')):.3e} "
                f"uuxdiag={row.get('ambient_uux_rel_l2', float('nan')):.3e} "
                f"uuydiag={row.get('ambient_uuy_rel_l2', float('nan')):.3e} "
                f"small_uux={row.get('small_uux_rel_l2', float('nan')):.3e} "
                f"small_uuy={row.get('small_uuy_rel_l2', float('nan')):.3e} "
                f"best_ep={best_epoch}"
            )
        else:
            print(
                f"[epoch {epoch:04d} train] "
                f"loss={row['train_loss']:.3e} "
                f"loss_g={row['train_loss_g']:.3e} "
                f"loss_grel={row['train_loss_g_rel']:.3e} "
                f"loss_h={row['train_loss_h']:.3e} "
                f"loss_second={row['train_loss_second']:.3e} "
                f"loss_zero={row['train_loss_zero']:.3e} "
                f"(diagnostics skipped; next eval every {args.eval_every} epochs) "
                f"best_ep={best_epoch}"
            )

        if args.early_stop_patience > 0 and epoch - best_epoch >= args.early_stop_patience:
            print(f"[early stop] no improvement for {args.early_stop_patience} epochs")
            break

    save_history_csv(os.path.join(args.outdir, "history.csv"), rows)
    if rows:
        np.savez(os.path.join(args.outdir, "history.npz"), **{k: np.array([r.get(k, np.nan) for r in rows]) for k in rows[0].keys()})
    plot_history(args.outdir, rows)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    plot_density_fit(model, args.outdir, args.u_max, device, dtype)
    best_row = rows[best_epoch - 1] if 0 < best_epoch <= len(rows) else (rows[-1] if rows else {})
    save_json(os.path.join(args.outdir, "final_metrics.json"), {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_metrics_from_training_history": best_row,
        "config": config,
    })
    print("[done] saved checkpoint:", os.path.join(args.outdir, "model_state.pt"))
    print(json.dumps({"best_epoch": int(best_epoch), "best_score": float(best_score), "best_metrics": best_row}, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Burgers local density h(u) by Sobolev matching; K22 ambient diagnostics only.")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--n-grid", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--act", type=str, required=True, choices=["gelu", "silu", "tanh"])
    p.add_argument("--init-last-scale", type=float, default=1.0e-3)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-train", type=int, required=True)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=1.0e-8)
    p.add_argument("--step-size", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--u-max", type=float, default=1.0)
    p.add_argument("--u-small-max", type=float, default=0.35)
    p.add_argument("--small-frac", type=float, default=0.70)
    p.add_argument("--rel-floor-u", type=float, default=0.05)
    p.add_argument("--lambda-h", type=float, default=0.05)
    p.add_argument("--lambda-g", type=float, default=1.0)
    p.add_argument("--lambda-g-rel", type=float, default=1.0)
    p.add_argument("--lambda-second", type=float, default=0.05)
    p.add_argument("--lambda-zero", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--ambient-eval-batches", type=int, default=16)
    p.add_argument("--ambient-eval-batch-size", type=int, default=16)
    p.add_argument("--small-eval-batches", type=int, default=8)
    p.add_argument("--small-target-min", type=float, default=0.04)
    p.add_argument("--small-target-max", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--use-double", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--outdir", type=str, required=True)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
