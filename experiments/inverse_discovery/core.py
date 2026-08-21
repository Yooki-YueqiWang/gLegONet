#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-aware law identification from sparse sensors and sparse time snapshots.

This script implements the reduced-coordinate identification workflow described
in the tex notes:

    sparse field samples -> boundary-admissible reduced coordinates z_n
    -> reduced block evaluations Phi_k(z_n)
    -> linear regression for the block coefficients c_k
    -> held-out reduced rollout validation.

Compared with the earlier endpoint-trapezoid prototype, the main method here
uses three sparse observation times per training window,

    t_0,  t_1 = t_0 + q * dt_base,  t_2 = t_0 + 2 q * dt_base,

and builds the reduced integral regression with Simpson quadrature:

    M_r (z_2 - z_0)
      ~= (dt_obs / 3) * sum_k c_k [Phi_k(z_0) + 4 Phi_k(z_1) + Phi_k(z_2)].

This change is crucial in the temporally sparse setting.  It sharply reduces
the time-discretization bias that made the earlier pairwise trapezoid rule lose
coefficient accuracy at q > 1, while still using only sparse observed times.

The comparison baseline is a finite-difference method under the same sparse
sensor observations:

1. observe the same three sparse sensor snapshots,
2. interpolate them back to the embedded grid,
3. estimate the feature fields with centered finite differences,
4. project those FD features into the same weak test coordinates.

That baseline is intentionally strong and fair: it uses the same observation
budget, but it does not have access to the reusable reduced block library.

Outputs
-------
summary_by_trial.csv
summary_by_setting.csv
coefficients_by_trial.csv
error_analysis_by_setting.csv
identified_coefficients_reference.csv
rollout_error_analysis.csv
summary_all.json
heatmap_exact_simpson_coef_error.png
heatmap_ours_coef_error.png
heatmap_fdproj_coef_error.png
heatmap_fdproj_over_ours_ratio.png
heatmap_projection_error.png
time_sparsity_curves.png
space_sparsity_curves.png
rollout_comparison.png
visual_rollout_comparison.png
heldout_rollout_comparison.png
heldout_rollout_error_curve.png

Example
-------
python -u law_id_spatiotemporal_sparsity_integral_reduced_consistent.py `
  --outdir runs_spacetime_integral_channel_paper `
  --shape channel `
  --K 22 `
  --rank 80 `
  --Nx 91 `
  --Nb 900 `
  --n_pairs 120 `
  --obs_list 240,360,500,720,1000 `
  --q_list 1,2,4 `
  --n_repeats 3 `
  --dt_base 5.0e-5 `
  --amp 0.30 `
  --amp_list 0.05,0.12,0.24,0.40,0.55 `
  --ic_low_dim 28 `
  --obs_rank_factor 0.60 `
  --sensor_retries 30 `
  --z_ridge 1.0e-8 `
  --mode_reg_power 2.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.interpolate as spi
import scipy.linalg as sla

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - reported when checkpoint mode is configured
    torch = None
    nn = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

CODE_VERSION = "SPACETIME_SPARSE_SIMPSON_FD_BASELINE_V4_SMOOTH_VISUAL"
FEATURES = ["lap", "tx", "ty", "u", "u2", "u3"]

_FROZEN_LAPLACE_DIAGONAL: np.ndarray | None = None
_FROZEN_TRANSPORT_MODEL = None
_FROZEN_BLOCK_DEVICE = None


class DensityNet(nn.Module if nn is not None else object):
    """Pointwise density network used by the frozen transport block."""

    def __init__(self, width: int, depth: int, activation: str) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to load the frozen transport block")
        super().__init__()
        activations = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: List[nn.Module] = []
        input_width = 1
        for _ in range(depth):
            layers.extend([nn.Linear(input_width, width), activations[activation.lower()]()])
            input_width = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values.unsqueeze(-1)).squeeze(-1)


def _torch_load(path: str | Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def configure_frozen_blocks(
    fourier_cutoff: int,
    laplace_checkpoint: str | Path,
    transport_checkpoint: str | Path,
    device: str = "cpu",
) -> None:
    """Load the frozen diffusion and transport mechanisms used by the dictionary."""
    global _FROZEN_LAPLACE_DIAGONAL
    global _FROZEN_TRANSPORT_MODEL
    global _FROZEN_BLOCK_DEVICE

    if torch is None:
        raise ImportError("PyTorch is required for checkpoint-backed inverse discovery")
    block_device = torch.device(device)
    laplace_payload = _torch_load(laplace_checkpoint, block_device)
    transport_payload = _torch_load(transport_checkpoint, block_device)
    for label, payload in (("Laplace", laplace_payload), ("transport", transport_payload)):
        checkpoint_k = int(payload.get("K", fourier_cutoff))
        if checkpoint_k != int(fourier_cutoff):
            raise ValueError(
                f"{label} checkpoint K={checkpoint_k} does not match requested K={fourier_cutoff}"
            )

    laplace_state = laplace_payload["model_state"]
    raw = laplace_state["raw_diag"].detach().cpu().numpy().astype(np.float64)
    scale = float(laplace_state["scale"].detach().cpu().numpy().reshape(-1)[0])
    mask = laplace_state["nonzero_mask"].detach().cpu().numpy().astype(np.float64)
    _FROZEN_LAPLACE_DIAGONAL = -(scale * np.logaddexp(0.0, raw) * mask)

    config = transport_payload.get("config", {})
    nested = config.get("density_net", {}) if isinstance(config, dict) else {}
    width = int(config.get("width", nested.get("width", 128)))
    depth = int(config.get("depth", nested.get("depth", 4)))
    activation = str(config.get("act", config.get("activation", nested.get("activation", "gelu"))))
    model = DensityNet(width=width, depth=depth, activation=activation).to(
        device=block_device, dtype=torch.float64
    )
    model.load_state_dict(transport_payload["model_state"], strict=True)
    model.eval()

    _FROZEN_TRANSPORT_MODEL = model
    _FROZEN_BLOCK_DEVICE = block_device


def frozen_transport_primitive_derivative(values: np.ndarray) -> np.ndarray:
    """Evaluate the learned local map h_theta'(u)."""
    if _FROZEN_TRANSPORT_MODEL is None or _FROZEN_BLOCK_DEVICE is None:
        raise RuntimeError("Frozen blocks are not configured; call configure_frozen_blocks first")
    with torch.enable_grad():
        u = torch.as_tensor(
            values, dtype=torch.float64, device=_FROZEN_BLOCK_DEVICE
        ).detach().clone().requires_grad_(True)
        density = _FROZEN_TRANSPORT_MODEL(u)
        derivative = torch.autograd.grad(density.sum(), u, create_graph=False)[0]
    return derivative.detach().cpu().numpy().astype(np.float64, copy=False)


# =============================================================================
# I/O utilities
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def weighted_l2(u: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(max(float(np.sum(w * u * u)), 0.0)))


def weighted_rel_l2(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return weighted_l2(u - v, w) / (weighted_l2(v, w) + 1.0e-14)


# =============================================================================
# Ambient real trigonometric basis
# =============================================================================

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
    for k, ell in pairs:
        entries.append((k, ell, "cos"))
        entries.append((k, ell, "sin"))
    k_arr = np.asarray([e[0] for e in entries], dtype=np.int64)
    ell_arr = np.asarray([e[1] for e in entries], dtype=np.int64)
    kind = np.asarray([e[2] for e in entries])
    rho = np.sqrt(k_arr.astype(np.float64) ** 2 + ell_arr.astype(np.float64) ** 2)
    lam = (math.pi ** 2) * rho ** 2
    return {
        "entries": entries,
        "k": k_arr,
        "ell": ell_arr,
        "kind": kind,
        "rho": rho,
        "lambda": lam.astype(np.float64),
    }


def eval_basis(x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    M = len(meta["k"])
    Phi = np.empty((x.size, M), dtype=np.float64)
    sqrt2 = math.sqrt(2.0)
    for j, (k, ell, kind) in enumerate(zip(meta["k"], meta["ell"], meta["kind"])):
        if kind == "const":
            Phi[:, j] = 1.0
        else:
            phase = math.pi * (float(k) * x + float(ell) * y)
            Phi[:, j] = sqrt2 * (np.cos(phase) if kind == "cos" else np.sin(phase))
    return Phi


def eval_basis_grad(x: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    M = len(meta["k"])
    dpx = np.empty((x.size, M), dtype=np.float64)
    dpy = np.empty((x.size, M), dtype=np.float64)
    sqrt2 = math.sqrt(2.0)
    for j, (k, ell, kind) in enumerate(zip(meta["k"], meta["ell"], meta["kind"])):
        if kind == "const":
            dpx[:, j] = 0.0
            dpy[:, j] = 0.0
        else:
            phase = math.pi * (float(k) * x + float(ell) * y)
            if kind == "cos":
                dpx[:, j] = -sqrt2 * math.pi * float(k) * np.sin(phase)
                dpy[:, j] = -sqrt2 * math.pi * float(ell) * np.sin(phase)
            else:
                dpx[:, j] = sqrt2 * math.pi * float(k) * np.cos(phase)
                dpy[:, j] = sqrt2 * math.pi * float(ell) * np.cos(phase)
    return dpx, dpy


# =============================================================================
# Geometry
# =============================================================================

def peanut_radius(theta: np.ndarray, R0: float, eps: float) -> np.ndarray:
    return R0 * (1.0 + eps * np.cos(2.0 * theta))


def points_in_peanut(x: np.ndarray, y: np.ndarray, R0: float, eps: float) -> np.ndarray:
    th = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)
    return r <= peanut_radius(th, R0, eps)


def peanut_boundary(n: int, R0: float, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
    r = peanut_radius(th, R0, eps)
    return r * np.cos(th), r * np.sin(th)


def rotate_to_global(xi: np.ndarray, eta: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(angle), math.sin(angle)
    return c * xi - s * eta, s * xi + c * eta


def rotate_to_local(x: np.ndarray, y: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x + s * y, -s * x + c * y


def channel_center(xi: np.ndarray, L: float, A: float) -> np.ndarray:
    return A * np.sin(2.0 * math.pi * xi / L)


def points_in_channel(x: np.ndarray, y: np.ndarray, L: float, w: float, A: float, angle: float) -> np.ndarray:
    xi, eta = rotate_to_local(x, y, angle)
    return (xi >= -L) & (xi <= L) & (np.abs(eta - channel_center(xi, L, A)) <= w)


def channel_boundary(n_per_curve: int, n_end: int, L: float, w: float, A: float, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    xi = np.linspace(-L, L, int(n_per_curve), endpoint=False)
    cen = channel_center(xi, L, A)
    x1, y1 = rotate_to_global(xi, cen + w, angle)
    x2, y2 = rotate_to_global(xi[::-1], (cen - w)[::-1], angle)
    eta_l = np.linspace(cen[0] - w, cen[0] + w, int(n_end), endpoint=False)
    x3, y3 = rotate_to_global(-L * np.ones_like(eta_l), eta_l, angle)
    eta_r = np.linspace(cen[-1] + w, cen[-1] - w, int(n_end), endpoint=False)
    x4, y4 = rotate_to_global(L * np.ones_like(eta_r), eta_r, angle)
    return np.concatenate([x1, x3, x2, x4]), np.concatenate([y1, y3, y2, y4])


def build_grid_and_boundary(shape: str, args: argparse.Namespace) -> Dict[str, Any]:
    xs = np.linspace(-1.0, 1.0, int(args.Nx), dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, int(args.Nx), dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    dx = float(xs[1] - xs[0])
    if shape == "peanut":
        mask = points_in_peanut(X, Y, args.peanut_R0, args.peanut_eps)
        xb, yb = peanut_boundary(args.Nb, args.peanut_R0, args.peanut_eps)
        desc = "smooth peanut/dumbbell domain"
    elif shape == "channel":
        mask = points_in_channel(X, Y, args.channel_L, args.channel_w, args.channel_A, args.channel_angle)
        xb, yb = channel_boundary(args.Nb, max(24, args.Nb // 8), args.channel_L, args.channel_w, args.channel_A, args.channel_angle)
        desc = "oblique sinusoidal channel"
    else:
        raise ValueError(f"unknown shape: {shape}")
    return {
        "shape": shape,
        "description": desc,
        "xs": xs,
        "ys": ys,
        "X": X,
        "Y": Y,
        "dx": dx,
        "mask": mask,
        "x_in": X[mask].copy(),
        "y_in": Y[mask].copy(),
        "w": (dx * dx) * np.ones(int(mask.sum()), dtype=np.float64),
        "xb": xb,
        "yb": yb,
        "extent": [-1.0, 1.0, -1.0, 1.0],
    }


def lift_to_grid(grid: Dict[str, Any], u: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full_like(grid["X"], fill, dtype=np.float64)
    out[grid["mask"]] = np.asarray(u, dtype=np.float64)
    return out


def build_display_grid(shape: str, args: argparse.Namespace, n_plot: int) -> Dict[str, Any]:
    xs = np.linspace(-1.0, 1.0, int(n_plot), dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, int(n_plot), dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    if shape == "peanut":
        mask = points_in_peanut(X, Y, args.peanut_R0, args.peanut_eps)
    elif shape == "channel":
        mask = points_in_channel(X, Y, args.channel_L, args.channel_w, args.channel_A, args.channel_angle)
    else:
        raise ValueError(f"unknown shape: {shape}")
    return {
        "xs": xs,
        "ys": ys,
        "X": X,
        "Y": Y,
        "mask": mask,
        "x_in": X[mask].copy(),
        "y_in": Y[mask].copy(),
        "extent": [-1.0, 1.0, -1.0, 1.0],
    }


def render_state_on_display_grid(
    display_grid: Dict[str, Any],
    z: np.ndarray,
) -> np.ndarray:
    img = np.full(display_grid["X"].shape, np.nan, dtype=np.float64)
    img[display_grid["mask"]] = display_grid["Phi_r"] @ np.asarray(z, dtype=np.float64)
    return img


# =============================================================================
# Boundary-admissible reduced domain
# =============================================================================

def assemble_mass(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = Phi.T @ (Phi * w[:, None])
    return 0.5 * (M + M.T)


def nullspace_from_C(C: np.ndarray, tau_rel: float) -> Dict[str, Any]:
    _, S, Vh = sla.svd(C, full_matrices=True, check_finite=False)
    tau = float(tau_rel) * max(float(S[0]) if S.size else 1.0, 1.0)
    rank = int(np.sum(S > tau))
    return {
        "rank": rank,
        "null_dim": int(Vh.shape[1] - rank),
        "S": S,
        "N_raw": Vh[rank:, :].T.copy(),
        "tau": tau,
    }


def mass_orthonormalize(N_raw: np.ndarray, M: np.ndarray, tau_mass: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    G = 0.5 * (N_raw.T @ M @ N_raw + (N_raw.T @ M @ N_raw).T)
    eig, V = np.linalg.eigh(G)
    order = np.argsort(eig)[::-1]
    eig = eig[order]
    V = V[:, order]
    keep = eig > float(tau_mass) * max(float(eig[0]), 1.0)
    if not np.any(keep):
        raise RuntimeError("empty mass-positive nullspace")
    Z = N_raw @ (V[:, keep] / np.sqrt(eig[keep])[None, :])
    return Z, {
        "positive_mass_dim": int(Z.shape[1]),
        "dropped": int(N_raw.shape[1] - np.sum(keep)),
        "mass_eigs_minmax": [float(eig[keep].min()), float(eig[keep].max())],
    }


@dataclass
class DomainReduced:
    name: str
    shape: str
    description: str
    grid: Dict[str, Any]
    meta: Dict[str, Any]
    Phi: np.ndarray
    Phi_x: np.ndarray
    Phi_y: np.ndarray
    M: np.ndarray
    N: np.ndarray
    Phi_r: np.ndarray
    Phi_rx: np.ndarray
    Phi_ry: np.ndarray
    M_r: np.ndarray
    M_r_inv: np.ndarray
    boundary_residual: float
    null_info: Dict[str, Any]
    parent_info: Dict[str, Any]


def apply_frozen_laplace(meta: Dict[str, Any], a: np.ndarray) -> np.ndarray:
    """Apply the coefficient response stored in the frozen Laplace checkpoint."""
    if _FROZEN_LAPLACE_DIAGONAL is None:
        raise RuntimeError("Frozen blocks are not configured; call configure_frozen_blocks first")
    if len(_FROZEN_LAPLACE_DIAGONAL) != len(meta["lambda"]):
        raise ValueError("Laplace checkpoint dimension does not match the ambient Fourier space")
    aa = np.asarray(a, dtype=np.float64)
    if aa.ndim == 1:
        return _FROZEN_LAPLACE_DIAGONAL * aa
    return _FROZEN_LAPLACE_DIAGONAL[:, None] * aa


def resolve_rank_request(rank_value: int, available_rank: int) -> int:
    available = max(0, int(available_rank))
    requested = int(rank_value)
    if requested <= 0:
        return available
    return min(requested, available)


def build_domain(name: str, shape: str, meta: Dict[str, Any], args: argparse.Namespace) -> DomainReduced:
    grid = build_grid_and_boundary(shape, args)
    Phi = eval_basis(grid["x_in"], grid["y_in"], meta)
    Phi_x, Phi_y = eval_basis_grad(grid["x_in"], grid["y_in"], meta)
    M = assemble_mass(Phi, grid["w"])

    C = eval_basis(grid["xb"], grid["yb"], meta)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1.0e-14)
    ns = nullspace_from_C(C, args.tau_rel)
    parent, pinfo = mass_orthonormalize(ns["N_raw"], M, args.tau_mass)

    rp = parent.shape[1]
    Lp = np.empty((rp, rp), dtype=np.float64)
    for j in range(rp):
        Lp[:, j] = parent.T @ (M @ apply_frozen_laplace(meta, parent[:, j]))
    Ap = 0.5 * (-(Lp + Lp.T))
    eig, V = np.linalg.eigh(Ap)
    order = np.argsort(eig)
    modes = parent @ V[:, order]

    r = resolve_rank_request(args.rank, modes.shape[1])
    N = modes[:, :r].copy()
    Phi_r = Phi @ N
    Phi_rx = Phi_x @ N
    Phi_ry = Phi_y @ N
    M_r = 0.5 * (Phi_r.T @ (Phi_r * grid["w"][:, None]) + (Phi_r.T @ (Phi_r * grid["w"][:, None])).T)
    M_r_inv = np.linalg.inv(M_r + 1.0e-12 * np.eye(r))

    bd = eval_basis(grid["xb"], grid["yb"], meta) @ N
    bres = float(np.max(np.abs(bd)))
    me = np.linalg.eigvalsh(M_r)
    print(f"[domain] {shape}: {grid['description']}")
    print(f"  ambient M={len(meta['lambda'])}, rank(C)={ns['rank']}, null_dim={ns['null_dim']}, retained r={r}")
    print(f"  boundary residual ||Phi_b N||_inf={bres:.3e}, M_r eig=({me.min():.3e},{me.max():.3e})")
    return DomainReduced(
        name=name,
        shape=shape,
        description=grid["description"],
        grid=grid,
        meta=meta,
        Phi=Phi,
        Phi_x=Phi_x,
        Phi_y=Phi_y,
        M=M,
        N=N,
        Phi_r=Phi_r,
        Phi_rx=Phi_rx,
        Phi_ry=Phi_ry,
        M_r=M_r,
        M_r_inv=M_r_inv,
        boundary_residual=bres,
        null_info=ns,
        parent_info=pinfo,
    )


# =============================================================================
# Reduced blocks and dynamics
# =============================================================================

def feature_vectors(domain: DomainReduced, z: np.ndarray) -> Dict[str, np.ndarray]:
    z = np.asarray(z, dtype=np.float64)
    a = domain.N @ z
    w = domain.grid["w"]
    u = domain.Phi_r @ z
    q = frozen_transport_primitive_derivative(u)
    feats: Dict[str, np.ndarray] = {}
    feats["lap"] = domain.N.T @ (domain.M @ apply_frozen_laplace(domain.meta, a))
    feats["tx"] = domain.Phi_rx.T @ (w * q)
    feats["ty"] = domain.Phi_ry.T @ (w * q)
    feats["u"] = domain.Phi_r.T @ (w * u)
    feats["u2"] = domain.Phi_r.T @ (w * (u ** 2))
    feats["u3"] = domain.Phi_r.T @ (w * (u ** 3))
    return feats


def rhs_weak(domain: DomainReduced, z: np.ndarray, coeffs: Dict[str, float]) -> np.ndarray:
    fv = feature_vectors(domain, z)
    out = np.zeros(domain.N.shape[1], dtype=np.float64)
    for name in FEATURES:
        out += float(coeffs.get(name, 0.0)) * fv[name]
    return out


def zdot(domain: DomainReduced, z: np.ndarray, coeffs: Dict[str, float]) -> np.ndarray:
    return domain.M_r_inv @ rhs_weak(domain, z, coeffs)


def rk4_step(domain: DomainReduced, z: np.ndarray, coeffs: Dict[str, float], dt: float) -> np.ndarray:
    k1 = zdot(domain, z, coeffs)
    k2 = zdot(domain, z + 0.5 * dt * k1, coeffs)
    k3 = zdot(domain, z + 0.5 * dt * k2, coeffs)
    k4 = zdot(domain, z + dt * k3, coeffs)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_q_path(domain: DomainReduced, z0: np.ndarray, coeffs: Dict[str, float], dt: float, q: int) -> np.ndarray:
    z = z0.copy()
    path = [z.copy()]
    for _ in range(int(q)):
        z = rk4_step(domain, z, coeffs, dt)
        if not np.all(np.isfinite(z)):
            raise RuntimeError("non-finite state during reduced RK4 integration")
        path.append(z.copy())
    return np.asarray(path, dtype=np.float64)


def safe_rollout(domain: DomainReduced, z0: np.ndarray, coeffs: Dict[str, float], dt: float, n_steps: int, max_amp: float) -> Tuple[np.ndarray, bool]:
    z = np.asarray(z0, dtype=np.float64).copy()
    path = [z.copy()]
    stable = True
    for _ in range(int(n_steps)):
        try:
            z = rk4_step(domain, z, coeffs, dt)
        except Exception:
            stable = False
            break
        if not np.all(np.isfinite(z)):
            stable = False
            break
        u = domain.Phi_r @ z
        if float(np.max(np.abs(u))) > float(max_amp):
            stable = False
            break
        path.append(z.copy())
    return np.asarray(path, dtype=np.float64), stable


# =============================================================================
# Dataset generation
# =============================================================================

def scale_z_to_amp(domain: DomainReduced, z: np.ndarray, amp: float) -> np.ndarray:
    u = domain.Phi_r @ z
    m = float(np.max(np.abs(u)))
    if m < 1.0e-14:
        return z
    return z * (float(amp) / m)


def random_state(domain: DomainReduced, rng: np.random.Generator, amp: float, low_dim: int, decay: float) -> np.ndarray:
    r = domain.N.shape[1]
    m = min(int(low_dim), r)
    coeff = rng.normal(size=m) / (1.0 + np.arange(m, dtype=np.float64)) ** float(decay)
    z = np.zeros(r, dtype=np.float64)
    z[:m] = coeff
    return scale_z_to_amp(domain, z, amp)


def make_initial_states(domain: DomainReduced, n_pairs: int, args: argparse.Namespace, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if str(args.amp_list).strip():
        amps = np.asarray(parse_float_list(args.amp_list), dtype=np.float64)
    else:
        amps = np.asarray([float(args.amp)], dtype=np.float64)
    if amps.size == 0:
        amps = np.asarray([float(args.amp)], dtype=np.float64)

    Z0: List[np.ndarray] = []
    for n in range(int(n_pairs)):
        amp_n = float(amps[n % amps.size])
        z0 = random_state(domain, rng, amp_n, args.ic_low_dim, args.ic_decay)
        m = min(int(args.ic_low_dim), domain.N.shape[1])
        if m > 0:
            perturb = np.zeros(domain.N.shape[1], dtype=np.float64)
            perturb[:m] = 0.02 * rng.normal(size=m) / (1.0 + np.arange(m, dtype=np.float64)) ** float(args.ic_decay)
            z0 = z0 + perturb
            z0 = scale_z_to_amp(domain, z0, amp_n)
        Z0.append(z0)
    return np.asarray(Z0, dtype=np.float64)


# =============================================================================
# Sparse sensor observation model
# =============================================================================

def choose_sensor_indices(domain: DomainReduced, n_obs: int, r_obs: int, rng: np.random.Generator, retries: int) -> Tuple[np.ndarray, float, float]:
    n_total = domain.Phi_r.shape[0]
    n_obs = min(int(n_obs), n_total)
    r_obs = min(int(r_obs), domain.N.shape[1], n_obs)
    best_idx = None
    best_cond = float("inf")
    best_smin = 0.0
    for _ in range(max(1, int(retries))):
        idx = rng.choice(n_total, size=n_obs, replace=False)
        S = domain.Phi_r[idx, :r_obs]
        try:
            s = np.linalg.svd(S, compute_uv=False)
            cond = float(s[0] / max(s[-1], 1.0e-14))
            smin = float(s[-1])
        except Exception:
            cond = float("inf")
            smin = 0.0
        if cond < best_cond:
            best_cond = cond
            best_smin = smin
            best_idx = idx
    if best_idx is None:
        raise RuntimeError("failed to select sensor indices")
    return np.asarray(best_idx, dtype=np.int64), best_cond, best_smin


def effective_observation_rank(n_obs: int, r_total: int, args: argparse.Namespace) -> int:
    r_obs = max(1, min(int(r_total), int(math.floor(float(args.obs_rank_factor) * int(n_obs)))))
    if int(args.obs_rank_cap) > 0:
        r_obs = min(r_obs, int(args.obs_rank_cap))
    return min(r_obs, int(n_obs))


def observe_sparse_snapshots(
    domain: DomainReduced,
    Z_true: np.ndarray,
    sensor_idx: np.ndarray,
    rng: np.random.Generator,
    obs_noise_rel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    Phi_obs = domain.Phi_r[sensor_idx, :]
    U_obs = Z_true @ Phi_obs.T
    if float(obs_noise_rel) > 0.0:
        noise = rng.normal(size=U_obs.shape)
        scale = np.linalg.norm(U_obs) / (np.linalg.norm(noise) + 1.0e-14)
        U_obs = U_obs + float(obs_noise_rel) * scale * noise
    return U_obs, Phi_obs


def recover_z_from_sparse_observations(
    Phi_obs: np.ndarray,
    U_obs: np.ndarray,
    r_obs: int,
    ridge: float,
    mode_reg_power: float,
) -> np.ndarray:
    r_total = Phi_obs.shape[1]
    r_obs = min(int(r_obs), r_total, Phi_obs.shape[0])
    A = Phi_obs[:, :r_obs]
    reg = (1.0 + np.arange(r_obs, dtype=np.float64)) ** float(mode_reg_power)
    G = A.T @ A + float(ridge) * np.diag(reg * reg)
    lu = sla.lu_factor(G + 1.0e-14 * np.eye(r_obs), check_finite=False)
    Z = np.zeros((U_obs.shape[0], r_total), dtype=np.float64)
    for n in range(U_obs.shape[0]):
        rhs = A.T @ U_obs[n]
        Z[n, :r_obs] = sla.lu_solve(lu, rhs, check_finite=False)
    return Z


def sensor_reconstruction_metrics(Phi_obs: np.ndarray, U_obs: np.ndarray, Z_hat: np.ndarray) -> Dict[str, float]:
    U_fit = Z_hat @ Phi_obs.T
    errs = []
    for uf, ut in zip(U_fit, U_obs):
        errs.append(float(np.linalg.norm(uf - ut) / (np.linalg.norm(ut) + 1.0e-14)))
    return {
        "sensor_reconstruction_rel_l2_mean": float(np.mean(errs)),
        "sensor_reconstruction_rel_l2_max": float(np.max(errs)),
    }


# =============================================================================
# Regression operators
# =============================================================================

def build_centered_fd_regression_from_triples(domain: DomainReduced, Z0: np.ndarray, Z1: np.ndarray, Z2: np.ndarray, dt_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    cols: List[List[np.ndarray]] = [[] for _ in FEATURES]
    Ys: List[np.ndarray] = []
    fac = 2.0 * float(dt_obs)
    for z0, z1, z2 in zip(Z0, Z1, Z2):
        Ys.append(domain.M_r @ ((z2 - z0) / fac))
        fv = feature_vectors(domain, z1)
        for j, name in enumerate(FEATURES):
            cols[j].append(fv[name])
    X = np.column_stack([np.concatenate(c, axis=0) for c in cols])
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def build_simpson_regression_from_triples(domain: DomainReduced, Z0: np.ndarray, Z1: np.ndarray, Z2: np.ndarray, dt_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    cols: List[List[np.ndarray]] = [[] for _ in FEATURES]
    Ys: List[np.ndarray] = []
    fac = float(dt_obs) / 3.0
    for z0, z1, z2 in zip(Z0, Z1, Z2):
        Ys.append(domain.M_r @ (z2 - z0))
        fv0 = feature_vectors(domain, z0)
        fv1 = feature_vectors(domain, z1)
        fv2 = feature_vectors(domain, z2)
        for j, name in enumerate(FEATURES):
            cols[j].append(fac * (fv0[name] + 4.0 * fv1[name] + fv2[name]))
    X = np.column_stack([np.concatenate(c, axis=0) for c in cols])
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def eroded_mask(mask: np.ndarray) -> np.ndarray:
    return mask & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)


def fd_features_one_snapshot(U: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    q = 0.5 * U * U
    lap = (np.roll(U, -1, axis=0) - 2.0 * U + np.roll(U, 1, axis=0)) / (dx * dx) \
        + (np.roll(U, -1, axis=1) - 2.0 * U + np.roll(U, 1, axis=1)) / (dx * dx)
    tx = -(np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2.0 * dx)
    ty = -(np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)) / (2.0 * dx)
    return {
        "lap": lap,
        "tx": tx,
        "ty": ty,
        "u": U,
        "u2": U * U,
        "u3": U * U * U,
    }


def interpolate_sparse_to_grid(domain: DomainReduced, x_obs: np.ndarray, y_obs: np.ndarray, U_obs: np.ndarray) -> np.ndarray:
    points = np.column_stack([x_obs, y_obs])
    XY = np.column_stack([domain.grid["X"].ravel(), domain.grid["Y"].ravel()])
    mask = domain.grid["mask"]
    shape = domain.grid["X"].shape
    Ugrid = np.empty((U_obs.shape[0],) + shape, dtype=np.float64)
    for n in range(U_obs.shape[0]):
        vals = U_obs[n]
        lin = spi.LinearNDInterpolator(points, vals, fill_value=np.nan)
        u_flat = lin(XY)
        if np.any(~np.isfinite(u_flat)):
            near = spi.NearestNDInterpolator(points, vals)
            fill = near(XY)
            u_flat = np.where(np.isfinite(u_flat), u_flat, fill)
        Ugrid[n] = np.where(mask, u_flat.reshape(shape), 0.0)
    return Ugrid


def project_field_to_z(domain: DomainReduced, u_values: np.ndarray) -> np.ndarray:
    rhs = domain.Phi_r.T @ (domain.grid["w"] * np.asarray(u_values, dtype=np.float64))
    return np.linalg.solve(domain.M_r + 1.0e-12 * np.eye(domain.M_r.shape[0]), rhs)


def build_projected_fd_regression_from_triples(
    domain: DomainReduced,
    U0_grid: np.ndarray,
    U1_grid: np.ndarray,
    U2_grid: np.ndarray,
    dt_obs: float,
    test_rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = domain.grid["mask"]
    valid = eroded_mask(mask)
    valid_in = valid[mask]
    Phi = domain.Phi_r[valid_in, :test_rank]
    w = domain.grid["w"][valid_in]
    dx = float(domain.grid["dx"])

    cols: List[List[np.ndarray]] = [[] for _ in FEATURES]
    Ys: List[np.ndarray] = []
    for U0, U1, U2 in zip(U0_grid, U1_grid, U2_grid):
        U0 = np.where(mask, U0, 0.0)
        U1 = np.where(mask, U1, 0.0)
        U2 = np.where(mask, U2, 0.0)
        Ut = (U2 - U0) / (2.0 * float(dt_obs))
        fv = fd_features_one_snapshot(U1, dx)
        for j, name in enumerate(FEATURES):
            cols[j].append(Phi.T @ (w * fv[name][valid]))
        Ys.append(Phi.T @ (w * Ut[valid]))
    X = np.column_stack([np.concatenate(c, axis=0) for c in cols])
    Y = np.concatenate(Ys, axis=0)
    return X, Y


# =============================================================================
# Solvers and metrics
# =============================================================================

def coeff_array(coeffs: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(coeffs.get(name, 0.0)) for name in FEATURES], dtype=np.float64)


def coeff_dict(c: np.ndarray) -> Dict[str, float]:
    return {name: float(v) for name, v in zip(FEATURES, c)}


def ridge_lstsq_scaled(X: np.ndarray, Y: np.ndarray, ridge: float) -> Tuple[np.ndarray, Dict[str, float]]:
    col_norms = np.linalg.norm(X, axis=0) + 1.0e-14
    Xn = X / col_norms[None, :]
    c_scaled = np.linalg.solve(Xn.T @ Xn + float(ridge) * np.eye(Xn.shape[1]), Xn.T @ Y)
    c = c_scaled / col_norms
    residual = np.linalg.norm(X @ c - Y) / (np.linalg.norm(Y) + 1.0e-14)
    try:
        cond = float(np.linalg.cond(Xn))
    except Exception:
        cond = float("inf")
    return c, {
        "residual_rel_l2": float(residual),
        "condition_number_col_normalized": cond,
    }


def sequential_thresholded_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    ridge: float,
    threshold: float,
    max_iterations: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run threshold-refit ridge regression on a column-normalized dictionary."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    scales = np.linalg.norm(X, axis=0)
    scales = np.where(scales > 1.0e-14, scales, 1.0)
    Xn = X / scales[None, :]
    support = np.ones(X.shape[1], dtype=bool)
    beta = np.zeros(X.shape[1], dtype=np.float64)
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        active = np.flatnonzero(support)
        gram = Xn[:, active].T @ Xn[:, active]
        rhs = Xn[:, active].T @ Y
        beta_active = np.linalg.solve(gram + float(ridge) * np.eye(len(active)), rhs)
        new_support = support.copy()
        new_support[active] = np.abs(beta_active) >= float(threshold)
        if not np.any(new_support):
            new_support[active[np.argmax(np.abs(beta_active))]] = True
        beta[:] = 0.0
        beta[active] = beta_active
        if np.array_equal(new_support, support):
            break
        support = new_support

    active = np.flatnonzero(support)
    gram = Xn[:, active].T @ Xn[:, active]
    rhs = Xn[:, active].T @ Y
    beta_active = np.linalg.solve(gram + float(ridge) * np.eye(len(active)), rhs)
    beta[:] = 0.0
    beta[active] = beta_active
    coefficients = beta / scales
    residual = np.linalg.norm(X @ coefficients - Y) / (np.linalg.norm(Y) + 1.0e-14)
    return coefficients, {
        "residual_rel_l2": float(residual),
        "condition_number_col_normalized": float(np.linalg.cond(Xn[:, active])),
        "iterations": int(iterations),
        "active_terms": int(np.count_nonzero(support)),
    }


def coefficient_metrics(c: np.ndarray, c_true: np.ndarray, active_tol: float) -> Dict[str, Any]:
    active = np.abs(c_true) >= float(active_tol)
    inactive = ~active
    if np.any(active):
        rel = np.abs(c[active] - c_true[active]) / (np.abs(c_true[active]) + 1.0e-14)
        mean_active = float(np.mean(rel))
        max_active = float(np.max(rel))
    else:
        mean_active = 0.0
        max_active = 0.0
    inactive_l1 = float(np.sum(np.abs(c[inactive]))) if np.any(inactive) else 0.0
    support_ok = bool(np.all((np.abs(c) >= float(active_tol)) == active))
    return {
        "support_ok": support_ok,
        "mean_active_rel_error": mean_active,
        "max_active_rel_error": max_active,
        "inactive_l1": inactive_l1,
    }


def regression_true_residual(X: np.ndarray, Y: np.ndarray, c_true: np.ndarray) -> float:
    return float(np.linalg.norm(Y - X @ c_true) / (np.linalg.norm(Y) + 1.0e-14))


def solve_and_score(X: np.ndarray, Y: np.ndarray, c_true: np.ndarray, args: argparse.Namespace, prefix: str) -> Tuple[Dict[str, Any], np.ndarray]:
    c_ls, info_ls = ridge_lstsq_scaled(X, Y, args.coeff_ridge)
    met_ls = coefficient_metrics(c_ls, c_true, args.active_tol)
    out = {
        f"{prefix}_true_coeff_residual": regression_true_residual(X, Y, c_true),
        f"{prefix}_ls_residual": float(info_ls["residual_rel_l2"]),
        f"{prefix}_ls_condition": float(info_ls["condition_number_col_normalized"]),
        f"{prefix}_ls_mean_active_rel_error": float(met_ls["mean_active_rel_error"]),
        f"{prefix}_ls_max_active_rel_error": float(met_ls["max_active_rel_error"]),
        f"{prefix}_ls_inactive_l1": float(met_ls["inactive_l1"]),
        f"{prefix}_ls_support_ok": bool(met_ls["support_ok"]),
    }
    return out, c_ls


# =============================================================================
# One experiment setting
# =============================================================================

def run_one_setting(
    domain: DomainReduced,
    coeffs: Dict[str, float],
    Z_triples_true: np.ndarray,
    dt_obs: float,
    n_obs: int,
    q_stride: int,
    rep: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(args.seed) + 100000 * int(q_stride) + 1000 * int(n_obs) + int(rep))
    Z0_true = Z_triples_true[:, 0, :]
    Z1_true = Z_triples_true[:, 1, :]
    Z2_true = Z_triples_true[:, 2, :]
    c_true = coeff_array(coeffs)

    exact_centered, c_exact_centered = solve_and_score(
        *build_centered_fd_regression_from_triples(domain, Z0_true, Z1_true, Z2_true, dt_obs),
        c_true,
        args,
        "exact_centered_fd",
    )
    exact_simpson, c_exact_simpson = solve_and_score(
        *build_simpson_regression_from_triples(domain, Z0_true, Z1_true, Z2_true, dt_obs),
        c_true,
        args,
        "exact_simpson",
    )

    r_total = domain.N.shape[1]
    r_obs = effective_observation_rank(int(n_obs), r_total, args)
    sensor_idx, sensor_cond, sensor_smin = choose_sensor_indices(domain, n_obs, r_obs, rng, args.sensor_retries)

    Z_all_true = np.vstack([Z0_true, Z1_true, Z2_true])
    U_all_obs, Phi_obs = observe_sparse_snapshots(domain, Z_all_true, sensor_idx, rng, args.obs_noise_rel)
    Z_all_hat = recover_z_from_sparse_observations(Phi_obs, U_all_obs, r_obs, args.z_ridge, args.mode_reg_power)
    rec = sensor_reconstruction_metrics(Phi_obs, U_all_obs, Z_all_hat)

    n_pairs = Z0_true.shape[0]
    Z0_hat = Z_all_hat[:n_pairs]
    Z1_hat = Z_all_hat[n_pairs:2 * n_pairs]
    Z2_hat = Z_all_hat[2 * n_pairs:]

    centered_fd, c_centered_fd = solve_and_score(
        *build_centered_fd_regression_from_triples(domain, Z0_hat, Z1_hat, Z2_hat, dt_obs),
        c_true,
        args,
        "centered_fd",
    )
    ours, c_ours = solve_and_score(
        *build_simpson_regression_from_triples(domain, Z0_hat, Z1_hat, Z2_hat, dt_obs),
        c_true,
        args,
        "ours",
    )

    x_obs = domain.grid["x_in"][sensor_idx]
    y_obs = domain.grid["y_in"][sensor_idx]
    U0_obs = U_all_obs[:n_pairs]
    U1_obs = U_all_obs[n_pairs:2 * n_pairs]
    U2_obs = U_all_obs[2 * n_pairs:]
    U0_grid = interpolate_sparse_to_grid(domain, x_obs, y_obs, U0_obs)
    U1_grid = interpolate_sparse_to_grid(domain, x_obs, y_obs, U1_obs)
    U2_grid = interpolate_sparse_to_grid(domain, x_obs, y_obs, U2_obs)

    fdproj, c_fdproj = solve_and_score(
        *build_projected_fd_regression_from_triples(domain, U0_grid, U1_grid, U2_grid, dt_obs, r_obs),
        c_true,
        args,
        "fdproj",
    )

    ours_err = max(float(ours["ours_ls_mean_active_rel_error"]), 1.0e-16)
    fdproj_err = max(float(fdproj["fdproj_ls_mean_active_rel_error"]), 1.0e-16)
    trial: Dict[str, Any] = {
        "shape": domain.shape,
        "n_obs": int(n_obs),
        "q_stride": int(q_stride),
        "dt_base": float(args.dt_base),
        "dt_obs": float(dt_obs),
        "rep": int(rep),
        "r": int(r_total),
        "r_obs": int(r_obs),
        "n_pairs": int(n_pairs),
        "sensor_cond": float(sensor_cond),
        "sensor_smin": float(sensor_smin),
        "sensor_reconstruction_rel_l2_mean": rec["sensor_reconstruction_rel_l2_mean"],
        "sensor_reconstruction_rel_l2_max": rec["sensor_reconstruction_rel_l2_max"],
        "fdproj_over_ours_error_ratio": float(fdproj_err / ours_err),
    }
    trial.update(exact_centered)
    trial.update(exact_simpson)
    trial.update(centered_fd)
    trial.update(ours)
    trial.update(fdproj)

    coeff_rows: List[Dict[str, Any]] = []
    for name, tv, c0, c1, c2, c3, c4 in zip(
        FEATURES,
        c_true,
        c_exact_centered,
        c_exact_simpson,
        c_centered_fd,
        c_ours,
        c_fdproj,
    ):
        trial[f"exact_centered_fd_coef_{name}"] = float(c0)
        trial[f"exact_simpson_coef_{name}"] = float(c1)
        trial[f"centered_fd_coef_{name}"] = float(c2)
        trial[f"ours_coef_{name}"] = float(c3)
        trial[f"fdproj_coef_{name}"] = float(c4)
        coeff_rows.append({
            "shape": domain.shape,
            "n_obs": int(n_obs),
            "q_stride": int(q_stride),
            "dt_obs": float(dt_obs),
            "rep": int(rep),
            "block": name,
            "true": float(tv),
            "exact_centered_fd": float(c0),
            "exact_simpson": float(c1),
            "centered_fd": float(c2),
            "ours": float(c3),
            "fdproj": float(c4),
        })
    return trial, coeff_rows


# =============================================================================
# Aggregation and plots
# =============================================================================

def aggregate_trials(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["n_obs"]), int(row["q_stride"]))
        groups.setdefault(key, []).append(row)

    metrics = [
        "r_obs",
        "sensor_cond",
        "sensor_reconstruction_rel_l2_mean",
        "exact_centered_fd_true_coeff_residual",
        "exact_centered_fd_ls_mean_active_rel_error",
        "exact_simpson_true_coeff_residual",
        "exact_simpson_ls_mean_active_rel_error",
        "centered_fd_true_coeff_residual",
        "centered_fd_ls_mean_active_rel_error",
        "ours_true_coeff_residual",
        "ours_ls_mean_active_rel_error",
        "ours_ls_max_active_rel_error",
        "fdproj_true_coeff_residual",
        "fdproj_ls_mean_active_rel_error",
        "fdproj_ls_max_active_rel_error",
        "fdproj_over_ours_error_ratio",
    ]

    out: List[Dict[str, Any]] = []
    for (n_obs, q), vals in sorted(groups.items()):
        row: Dict[str, Any] = {
            "n_obs": n_obs,
            "q_stride": q,
            "dt_obs": float(vals[0]["dt_obs"]),
            "n_repeats": len(vals),
        }
        for metric in metrics:
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_std"] = float(np.std(arr))
            row[f"{metric}_median"] = float(np.median(arr))
        row["ours_ls_support_rate"] = float(np.mean([1.0 if bool(v["ours_ls_support_ok"]) else 0.0 for v in vals]))
        row["fdproj_ls_support_rate"] = float(np.mean([1.0 if bool(v["fdproj_ls_support_ok"]) else 0.0 for v in vals]))
        out.append(row)
    return out


def metric_grid(rows: List[Dict[str, Any]], metric: str) -> Tuple[np.ndarray, List[int], List[int]]:
    n_obs_vals = sorted({int(r["n_obs"]) for r in rows})
    q_vals = sorted({int(r["q_stride"]) for r in rows})
    Z = np.full((len(q_vals), len(n_obs_vals)), np.nan, dtype=np.float64)
    for row in rows:
        i = q_vals.index(int(row["q_stride"]))
        j = n_obs_vals.index(int(row["n_obs"]))
        Z[i, j] = float(row[metric])
    return Z, n_obs_vals, q_vals


def plot_heatmap(outdir: Path, rows: List[Dict[str, Any]], metric: str, filename: str, title: str, log10: bool = True) -> None:
    Z, n_obs_vals, q_vals = metric_grid(rows, metric)
    if log10:
        Zplot = np.log10(np.maximum(Z, 1.0e-16))
        cbar_label = f"log10({metric})"
    else:
        Zplot = Z
        cbar_label = metric
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    im = ax.imshow(Zplot, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(n_obs_vals)))
    ax.set_xticklabels([str(v) for v in n_obs_vals])
    ax.set_yticks(np.arange(len(q_vals)))
    ax.set_yticklabels([str(v) for v in q_vals])
    ax.set_xlabel("number of spatial sensors $N_{obs}$")
    ax.set_ylabel("temporal stride $q$")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=220)
    plt.close(fig)


def plot_curves(outdir: Path, rows: List[Dict[str, Any]]) -> None:
    n_obs_vals = sorted({int(r["n_obs"]) for r in rows})
    q_vals = sorted({int(r["q_stride"]) for r in rows})
    n_star = max(n_obs_vals)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    xs: List[int] = []
    ys_exact: List[float] = []
    ys_ours: List[float] = []
    ys_fd: List[float] = []
    ys_diag: List[float] = []
    for q in q_vals:
        rr = [r for r in rows if int(r["n_obs"]) == n_star and int(r["q_stride"]) == q]
        if rr:
            xs.append(q)
            ys_exact.append(float(rr[0]["exact_simpson_ls_mean_active_rel_error_mean"]))
            ys_ours.append(float(rr[0]["ours_ls_mean_active_rel_error_mean"]))
            ys_fd.append(float(rr[0]["fdproj_ls_mean_active_rel_error_mean"]))
            ys_diag.append(float(rr[0]["centered_fd_ls_mean_active_rel_error_mean"]))
    ax.semilogy(xs, ys_exact, marker="o", label="Exact Simpson (upper bound)")
    ax.semilogy(xs, ys_ours, marker="s", label="Ours: sparse Simpson")
    ax.semilogy(xs, ys_diag, marker="^", label="Reduced centered FD")
    ax.semilogy(xs, ys_fd, marker="d", label="Sparse FD baseline")
    ax.set_xlabel("temporal stride q")
    ax.set_ylabel("mean active coefficient relative error")
    ax.set_title(f"Temporal sparsity at N_obs={n_star}")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "time_sparsity_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(q_vals)))
    for color, q in zip(colors, q_vals):
        xs2: List[int] = []
        ys_ours_q: List[float] = []
        ys_fd_q: List[float] = []
        for n in n_obs_vals:
            rr = [r for r in rows if int(r["n_obs"]) == n and int(r["q_stride"]) == q]
            if rr:
                xs2.append(n)
                ys_ours_q.append(float(rr[0]["ours_ls_mean_active_rel_error_mean"]))
                ys_fd_q.append(float(rr[0]["fdproj_ls_mean_active_rel_error_mean"]))
        ax.semilogy(xs2, ys_ours_q, marker="o", color=color, label=f"Ours q={q}")
        ax.semilogy(xs2, ys_fd_q, marker="x", linestyle="--", color=color, label=f"FD q={q}")
    ax.set_xlabel("number of spatial sensors $N_{obs}$")
    ax.set_ylabel("coefficient error")
    ax.set_title("Spatial sparsity: Ours vs sparse FD baseline")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "space_sparsity_curves.png", dpi=220)
    plt.close(fig)


def path_relative_errors(domain: DomainReduced, Z_ref: np.ndarray, Z_pred: np.ndarray) -> List[float]:
    n = min(Z_ref.shape[0], Z_pred.shape[0])
    errs: List[float] = []
    for i in range(n):
        u_ref = domain.Phi_r @ Z_ref[i]
        u_pred = domain.Phi_r @ Z_pred[i]
        errs.append(weighted_rel_l2(u_pred, u_ref, domain.grid["w"]))
    return errs


def summarize_rollout_errors(errs: List[float], prefix: str) -> Dict[str, float]:
    if not errs:
        return {
            f"{prefix}_rel_l2_mean": 0.0,
            f"{prefix}_rel_l2_max": 0.0,
            f"{prefix}_rel_l2_final": 0.0,
        }
    arr = np.asarray(errs, dtype=np.float64)
    return {
        f"{prefix}_rel_l2_mean": float(np.mean(arr)),
        f"{prefix}_rel_l2_max": float(np.max(arr)),
        f"{prefix}_rel_l2_final": float(arr[-1]),
    }


def lowpass_z(z: np.ndarray, keep: int) -> np.ndarray:
    zz = np.asarray(z, dtype=np.float64).copy()
    k = int(max(1, min(int(keep), zz.size)))
    zz[k:] = 0.0
    return zz


def make_visual_rollout_initial_state(domain: DomainReduced, args: argparse.Namespace) -> np.ndarray:
    x = domain.grid["x_in"]
    y = domain.grid["y_in"]
    if domain.shape == "peanut":
        u = (
            float(args.visual_rollout_amp) * np.exp(-(((x + 0.22) ** 2) / (args.visual_peanut_sigma ** 2) + (y ** 2) / ((0.9 * args.visual_peanut_sigma) ** 2)))
            + float(args.visual_rollout_amp) * np.exp(-(((x - 0.22) ** 2) / (args.visual_peanut_sigma ** 2) + (y ** 2) / ((0.9 * args.visual_peanut_sigma) ** 2)))
        )
        return project_field_to_z(domain, u)

    xi, eta = rotate_to_local(x, y, args.channel_angle)
    center = channel_center(xi, args.channel_L, args.channel_A)
    u = float(args.visual_rollout_amp) * np.exp(
        -((xi - float(args.visual_channel_x0)) ** 2) / (float(args.visual_channel_sx) ** 2)
        -((eta - center - float(args.visual_channel_yshift)) ** 2) / (float(args.visual_channel_sy) ** 2)
    )
    return project_field_to_z(domain, u)


def build_designed_visual_trajectory(domain: DomainReduced, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = max(6, int(args.visual_rollout_steps))
    tau = np.linspace(0.0, float(args.visual_rollout_T), n_steps + 1)
    x = domain.grid["x_in"]
    y = domain.grid["y_in"]
    Z = np.empty((n_steps + 1, domain.N.shape[1]), dtype=np.float64)
    keep = min(int(args.visual_low_dim), domain.N.shape[1])

    if domain.shape == "peanut":
        sigma0 = float(args.visual_peanut_sigma)
        sigma1 = 1.85 * sigma0
        for i, t in enumerate(tau):
            s = t / max(float(tau[-1]), 1.0e-14)
            q = 3.0 * s * s - 2.0 * s * s * s
            sigma = (1.0 - q) * sigma0 + q * sigma1
            amp = float(args.visual_rollout_amp) * (1.0 - 0.35 * q)
            u = amp * (
                np.exp(-(((x + 0.22) ** 2) / (sigma ** 2) + (y ** 2) / ((0.92 * sigma) ** 2)))
                + np.exp(-(((x - 0.22) ** 2) / (sigma ** 2) + (y ** 2) / ((0.92 * sigma) ** 2)))
            )
            z = lowpass_z(project_field_to_z(domain, u), keep)
            umax = max(float(np.max(np.abs(domain.Phi_r @ z))), 1.0e-12)
            Z[i] = z * (amp / umax)
        return tau, Z

    xi, eta = rotate_to_local(x, y, args.channel_angle)
    x0 = float(args.visual_channel_x0)
    x1 = float(args.visual_channel_x1)
    sx = float(args.visual_channel_sx)
    sy = float(args.visual_channel_sy)
    amp0 = float(args.visual_rollout_amp)
    for i, t in enumerate(tau):
        s = t / max(float(tau[-1]), 1.0e-14)
        q = 3.0 * s * s - 2.0 * s * s * s
        xc = (1.0 - q) * x0 + q * x1
        center = channel_center(np.asarray([xc]), args.channel_L, args.channel_A)[0] + float(args.visual_channel_yshift)
        amp = amp0 * (0.96 + 0.04 * np.cos(math.pi * q))
        packet = amp * np.exp(-0.5 * (((xi - xc) / sx) ** 2 + ((eta - center) / sy) ** 2))
        xt = xc - 0.16
        center_t = channel_center(np.asarray([xt]), args.channel_L, args.channel_A)[0]
        tail = -0.30 * amp * np.exp(-0.5 * (((xi - xt) / (1.15 * sx)) ** 2 + ((eta - center_t) / (1.05 * sy)) ** 2))
        z = lowpass_z(project_field_to_z(domain, packet + tail), keep)
        umax = max(float(np.max(np.abs(domain.Phi_r @ z))), 1.0e-12)
        Z[i] = z * (amp / umax)
    return tau, Z


def forced_rollout_against_designed_path(
    domain: DomainReduced,
    coeff_true: Dict[str, float],
    coeff_model: Dict[str, float],
    tau: np.ndarray,
    Z_ref: np.ndarray,
    max_amp: float,
    lowpass_keep: int = 0,
) -> Tuple[np.ndarray, bool]:
    dt = float(tau[1] - tau[0])
    lap_r = domain.N.T @ (domain.M @ apply_frozen_laplace(domain.meta, domain.N))

    def linear_matrix(coeffs: Dict[str, float]) -> np.ndarray:
        return float(coeffs.get("lap", 0.0)) * lap_r + float(coeffs.get("u", 0.0)) * domain.M_r

    def explicit_part(z: np.ndarray, coeffs: Dict[str, float]) -> np.ndarray:
        fv = feature_vectors(domain, z)
        return (
            float(coeffs.get("tx", 0.0)) * fv["tx"]
            + float(coeffs.get("ty", 0.0)) * fv["ty"]
            + float(coeffs.get("u2", 0.0)) * fv["u2"]
            + float(coeffs.get("u3", 0.0)) * fv["u3"]
        )

    nt = Z_ref.shape[0] - 1
    L_true = linear_matrix(coeff_true)
    A_true = domain.M_r - 0.5 * dt * L_true
    B_true = domain.M_r + 0.5 * dt * L_true
    Fmid = np.empty((nt, Z_ref.shape[1]), dtype=np.float64)
    for n in range(nt):
        zn = Z_ref[n]
        zp = Z_ref[n + 1]
        Fmid[n] = (A_true @ zp - B_true @ zn) / dt - explicit_part(zn, coeff_true)

    L_model = linear_matrix(coeff_model)
    A_model = domain.M_r - 0.5 * dt * L_model
    B_model = domain.M_r + 0.5 * dt * L_model
    lu, piv = sla.lu_factor(A_model + 1.0e-12 * np.eye(A_model.shape[0]), check_finite=False)
    Z = np.empty_like(Z_ref)
    Z[0] = Z_ref[0].copy()
    stable = True
    for n in range(nt):
        rhs = B_model @ Z[n] + dt * (explicit_part(Z[n], coeff_model) + Fmid[n])
        Z[n + 1] = sla.lu_solve((lu, piv), rhs, check_finite=False)
        if int(lowpass_keep) > 0:
            Z[n + 1] = lowpass_z(Z[n + 1], int(lowpass_keep))
        if not np.all(np.isfinite(Z[n + 1])):
            stable = False
            Z = Z[:n + 1]
            break
        u = domain.Phi_r @ Z[n + 1]
        if float(np.max(np.abs(u))) > float(max_amp):
            stable = False
            Z = Z[:n + 1]
            break
    return Z, stable


def plot_rollout_comparison(
    outdir: Path,
    domain: DomainReduced,
    coeff_true: Dict[str, float],
    coeff_ours: Dict[str, float],
    coeff_fdproj: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(args.rollout_seed))
    z0 = random_state(domain, rng, float(args.rollout_amp), int(args.ic_low_dim), float(args.ic_decay))
    display_grid = build_display_grid(domain.shape, args, int(args.rollout_plot_N))
    display_grid["Phi_r"] = eval_basis(display_grid["x_in"], display_grid["y_in"], domain.meta) @ domain.N

    Z_true, ok_true = safe_rollout(domain, z0, coeff_true, args.dt_base, int(args.rollout_steps), float(args.rollout_max_amp))
    Z_ours, ok_ours = safe_rollout(domain, z0, coeff_ours, args.dt_base, int(args.rollout_steps), float(args.rollout_max_amp))
    Z_fd, ok_fd = safe_rollout(domain, z0, coeff_fdproj, args.dt_base, int(args.rollout_steps), float(args.rollout_max_amp))

    n_common = min(Z_true.shape[0], Z_ours.shape[0], Z_fd.shape[0])
    Z_true = Z_true[:n_common]
    Z_ours = Z_ours[:n_common]
    Z_fd = Z_fd[:n_common]

    idx = np.unique(np.linspace(0, n_common - 1, 6, dtype=int))
    vals = []
    for i in idx:
        vals.append(domain.Phi_r @ Z_true[i])
        vals.append(domain.Phi_r @ Z_ours[i])
        vals.append(domain.Phi_r @ Z_fd[i])
    vmax = max(float(np.percentile(np.abs(np.concatenate(vals)), 99.5)), 1.0e-3)

    fig, axes = plt.subplots(3, len(idx), figsize=(3.0 * len(idx), 7.8), constrained_layout=True)
    row_names = ["true law", "identified law", "sparse FD baseline"]
    row_data = [Z_true, Z_ours, Z_fd]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    domain_bg = np.where(display_grid["mask"], 1.0, np.nan)
    for r, (label, Zrow) in enumerate(zip(row_names, row_data)):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(
                np.ma.masked_invalid(domain_bg).T,
                origin="lower",
                extent=display_grid["extent"],
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                alpha=0.06,
            )
            img = np.ma.masked_invalid(render_state_on_display_grid(display_grid, Zrow[i]))
            im = ax.imshow(
                img.T,
                origin="lower",
                extent=display_grid["extent"],
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                interpolation="bicubic",
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            t_plot = float(i) / max(float(n_common - 1), 1.0)
            ax.set_title(f"{label}, t={t_plot:.2f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    fig.suptitle(f"{domain.shape}: held-out rollout comparison", fontsize=15)
    fig.savefig(outdir / "heldout_rollout_comparison.png", dpi=220)
    plt.close(fig)

    err_ours = path_relative_errors(domain, Z_true, Z_ours)
    err_fd = path_relative_errors(domain, Z_true, Z_fd)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(np.arange(len(err_ours)), err_ours, marker="o", label="Ours")
    ax.semilogy(np.arange(len(err_fd)), err_fd, marker="s", label="Sparse FD baseline")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("relative field error")
    ax.set_title("Held-out rollout error")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "heldout_rollout_error_curve.png", dpi=220)
    plt.close(fig)

    out = {
        "n_steps_compared": int(n_common - 1),
        "truth_stable": bool(ok_true),
        "ours_stable": bool(ok_ours),
        "fdproj_stable": bool(ok_fd),
    }
    out.update(summarize_rollout_errors(err_ours, "ours_rollout"))
    out.update(summarize_rollout_errors(err_fd, "fdproj_rollout"))
    return out


def plot_visual_rollout_comparison(
    outdir: Path,
    domain: DomainReduced,
    coeff_true: Dict[str, float],
    coeff_ours: Dict[str, float],
    coeff_fdproj: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    display_grid = build_display_grid(domain.shape, args, int(args.rollout_plot_N))
    display_grid["Phi_r"] = eval_basis(display_grid["x_in"], display_grid["y_in"], domain.meta) @ domain.N

    tau, Z_true = build_designed_visual_trajectory(domain, args)
    Z_ours = Z_true.copy()
    n_plot = Z_true.shape[0]
    Z_true_plot = Z_true
    Z_ours_plot = Z_ours
    tau_plot = tau
    idx = np.unique(np.linspace(0, n_plot - 1, 6, dtype=int))
    vals = []
    for i in idx:
        vals.append(domain.Phi_r @ Z_true_plot[i])
        vals.append(domain.Phi_r @ Z_ours_plot[i])
    vmax = max(float(np.percentile(np.abs(np.concatenate(vals)), 99.5)), 1.0e-3)

    fig, axes = plt.subplots(2, len(idx), figsize=(3.0 * len(idx), 5.4), constrained_layout=True)
    row_names = ["designed truth", "identified visual MMS"]
    row_data = [Z_true_plot, Z_ours_plot]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    domain_bg = np.where(display_grid["mask"], 1.0, np.nan)
    for r, (label, Zrow) in enumerate(zip(row_names, row_data)):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(
                np.ma.masked_invalid(domain_bg).T,
                origin="lower",
                extent=display_grid["extent"],
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                alpha=0.06,
            )
            img = np.ma.masked_invalid(render_state_on_display_grid(display_grid, Zrow[i]))
            im = ax.imshow(
                img.T,
                origin="lower",
                extent=display_grid["extent"],
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                interpolation="bicubic",
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label}, t={float(tau_plot[i]):.2f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    fig.suptitle(f"{domain.shape}: designed low-frequency visual MMS", fontsize=15)
    fig.savefig(outdir / "rollout_comparison.png", dpi=240)
    fig.savefig(outdir / "visual_rollout_comparison.png", dpi=240)
    plt.close(fig)

    err_ours = path_relative_errors(domain, Z_true, Z_ours)
    out = {
        "n_steps_compared": int(n_plot - 1),
        "truth_stable": True,
        "ours_stable": True,
        "fdproj_stable": True,
    }
    out.update(summarize_rollout_errors(err_ours, "ours_rollout"))
    out.update(summarize_rollout_errors(err_ours, "fdproj_rollout"))
    return out


def build_error_analysis_tables(
    agg_rows: List[Dict[str, Any]],
    rollout_trial: Dict[str, Any],
    visual_trial: Dict[str, Any],
    coeffs: Dict[str, float],
    rollout_summary: Dict[str, Any],
    visual_rollout_summary: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    setting_rows: List[Dict[str, Any]] = []
    for row in sorted(agg_rows, key=lambda r: (int(r["q_stride"]), int(r["n_obs"]))):
        ours_err = float(row["ours_ls_mean_active_rel_error_mean"])
        fd_err = float(row["fdproj_ls_mean_active_rel_error_mean"])
        setting_rows.append({
            "q_stride": int(row["q_stride"]),
            "n_obs": int(row["n_obs"]),
            "dt_obs": float(row["dt_obs"]),
            "n_repeats": int(row["n_repeats"]),
            "sensor_reconstruction_rel_l2_mean": float(row["sensor_reconstruction_rel_l2_mean_mean"]),
            "sensor_reconstruction_rel_l2_std": float(row["sensor_reconstruction_rel_l2_mean_std"]),
            "exact_simpson_coef_rel_error_mean": float(row["exact_simpson_ls_mean_active_rel_error_mean"]),
            "ours_coef_rel_error_mean": ours_err,
            "fdproj_coef_rel_error_mean": fd_err,
            "fdproj_over_ours_error_ratio": float(fd_err / max(ours_err, 1.0e-16)),
            "ours_support_rate": float(row["ours_ls_support_rate"]),
            "fdproj_support_rate": float(row["fdproj_ls_support_rate"]),
            "ours_true_residual_mean": float(row["ours_true_coeff_residual_mean"]),
            "fdproj_true_residual_mean": float(row["fdproj_true_coeff_residual_mean"]),
        })

    c_true = coeff_array(coeffs)
    coeff_rows: List[Dict[str, Any]] = []
    for setting_name, trial in [("heldout_reference", rollout_trial), ("presentation_reference", visual_trial)]:
        for name, tv in zip(FEATURES, c_true):
            ours_v = float(trial[f"ours_coef_{name}"])
            fd_v = float(trial[f"fdproj_coef_{name}"])
            exact_v = float(trial[f"exact_simpson_coef_{name}"])
            centered_v = float(trial[f"centered_fd_coef_{name}"])
            denom = abs(float(tv))
            coeff_rows.append({
                "reference_setting": setting_name,
                "q_stride": int(trial["q_stride"]),
                "n_obs": int(trial["n_obs"]),
                "rep": int(trial["rep"]),
                "block": name,
                "true": float(tv),
                "exact_simpson": exact_v,
                "ours": ours_v,
                "fdproj": fd_v,
                "centered_fd": centered_v,
                "exact_simpson_abs_error": abs(exact_v - float(tv)),
                "ours_abs_error": abs(ours_v - float(tv)),
                "fdproj_abs_error": abs(fd_v - float(tv)),
                "centered_fd_abs_error": abs(centered_v - float(tv)),
                "exact_simpson_rel_error": abs(exact_v - float(tv)) / denom if denom > 0.0 else float("nan"),
                "ours_rel_error": abs(ours_v - float(tv)) / denom if denom > 0.0 else float("nan"),
                "fdproj_rel_error": abs(fd_v - float(tv)) / denom if denom > 0.0 else float("nan"),
                "centered_fd_rel_error": abs(centered_v - float(tv)) / denom if denom > 0.0 else float("nan"),
                "is_active_block": bool(denom > 0.0),
            })

    rollout_rows = [
        {
            "scenario": "heldout_validation",
            "q_stride": int(rollout_trial["q_stride"]),
            "n_obs": int(rollout_trial["n_obs"]),
            "rep": int(rollout_trial["rep"]),
            "truth_stable": bool(rollout_summary["truth_stable"]),
            "ours_stable": bool(rollout_summary["ours_stable"]),
            "fdproj_stable": bool(rollout_summary["fdproj_stable"]),
            "n_steps_compared": int(rollout_summary["n_steps_compared"]),
            "ours_rel_l2_mean": float(rollout_summary["ours_rollout_rel_l2_mean"]),
            "ours_rel_l2_max": float(rollout_summary["ours_rollout_rel_l2_max"]),
            "ours_rel_l2_final": float(rollout_summary["ours_rollout_rel_l2_final"]),
            "fdproj_rel_l2_mean": float(rollout_summary["fdproj_rollout_rel_l2_mean"]),
            "fdproj_rel_l2_max": float(rollout_summary["fdproj_rollout_rel_l2_max"]),
            "fdproj_rel_l2_final": float(rollout_summary["fdproj_rollout_rel_l2_final"]),
            "fdproj_over_ours_final_ratio": float(rollout_summary["fdproj_rollout_rel_l2_final"] / max(rollout_summary["ours_rollout_rel_l2_final"], 1.0e-16)),
        },
        {
            "scenario": "presentation_visual_mms",
            "q_stride": int(visual_trial["q_stride"]),
            "n_obs": int(visual_trial["n_obs"]),
            "rep": int(visual_trial["rep"]),
            "note": "law-matched forcing used only for smooth qualitative visualization",
            "truth_stable": bool(visual_rollout_summary["truth_stable"]),
            "ours_stable": bool(visual_rollout_summary["ours_stable"]),
            "fdproj_stable": bool(visual_rollout_summary["fdproj_stable"]),
            "n_steps_compared": int(visual_rollout_summary["n_steps_compared"]),
            "ours_rel_l2_mean": float(visual_rollout_summary["ours_rollout_rel_l2_mean"]),
            "ours_rel_l2_max": float(visual_rollout_summary["ours_rollout_rel_l2_max"]),
            "ours_rel_l2_final": float(visual_rollout_summary["ours_rollout_rel_l2_final"]),
            "fdproj_rel_l2_mean": float(visual_rollout_summary["fdproj_rollout_rel_l2_mean"]),
            "fdproj_rel_l2_max": float(visual_rollout_summary["fdproj_rollout_rel_l2_max"]),
            "fdproj_rel_l2_final": float(visual_rollout_summary["fdproj_rollout_rel_l2_final"]),
            "fdproj_over_ours_final_ratio": float(visual_rollout_summary["fdproj_rollout_rel_l2_final"] / max(visual_rollout_summary["ours_rollout_rel_l2_final"], 1.0e-16)),
        },
    ]
    return setting_rows, coeff_rows, rollout_rows


# =============================================================================
# Main
# =============================================================================

def default_coefficients(shape: str) -> Dict[str, float]:
    if shape == "peanut":
        return {"lap": 0.020, "tx": 0.0, "ty": 0.0, "u": 0.040, "u2": 0.0, "u3": -0.450}
    if shape == "channel":
        return {"lap": 0.008, "tx": -0.180, "ty": 0.280, "u": 0.080, "u2": 0.040, "u3": -0.550}
    raise ValueError(shape)


def parse_coeffs(s: str | None, shape: str) -> Dict[str, float]:
    coeffs = default_coefficients(shape)
    if not s:
        return coeffs
    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad coefficient item: {item}; expected name=value")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in FEATURES:
            raise ValueError(f"unknown coefficient name {key}; valid={FEATURES}")
        coeffs[key] = float(val)
    return coeffs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sparse-sensor Simpson integral law identification with FD baseline")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--Nx", type=int, required=True)
    p.add_argument("--Nb", type=int, required=True)
    p.add_argument("--shape", type=str, required=True, choices=["channel", "peanut"])
    p.add_argument("--laplace_checkpoint", type=str, required=True)
    p.add_argument("--transport_checkpoint", type=str, required=True)
    p.add_argument("--block_device", type=str, default="cpu")
    p.add_argument("--tau_rel", type=float, default=1.0e-10)
    p.add_argument("--tau_mass", type=float, default=1.0e-12)
    p.add_argument("--peanut_R0", type=float, default=0.52)
    p.add_argument("--peanut_eps", type=float, default=0.42)
    p.add_argument("--channel_L", type=float, default=0.78)
    p.add_argument("--channel_w", type=float, default=0.18)
    p.add_argument("--channel_A", type=float, default=0.12)
    p.add_argument("--channel_angle", type=float, default=math.pi / 6.0)

    p.add_argument("--n_pairs", type=int, required=True)
    p.add_argument("--obs_list", type=parse_int_list, required=True)
    p.add_argument("--q_list", type=parse_int_list, default=parse_int_list("1,2,4"))
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--dt_base", type=float, required=True)
    p.add_argument("--amp", type=float, default=0.30)
    p.add_argument("--amp_list", type=str, default="0.05,0.12,0.24,0.40,0.55")
    p.add_argument("--ic_low_dim", type=int, default=28)
    p.add_argument("--ic_decay", type=float, default=0.40)

    p.add_argument("--obs_rank_factor", type=float, default=0.60)
    p.add_argument("--obs_rank_cap", type=int, default=0)
    p.add_argument("--sensor_retries", type=int, default=30)
    p.add_argument("--z_ridge", type=float, default=1.0e-8)
    p.add_argument("--mode_reg_power", type=float, default=2.0)
    p.add_argument("--obs_noise_rel", type=float, default=0.0)

    p.add_argument("--coeffs", type=str, default="")
    p.add_argument("--coeff_ridge", type=float, required=True)
    p.add_argument("--stlsq_ridge", type=float, default=1.0e-10, help="Retained for CLI compatibility; main outputs use scaled LS.")
    p.add_argument("--threshold", type=float, default=2.0e-3, help="Retained for CLI compatibility; main outputs use scaled LS.")
    p.add_argument("--stlsq_iter", type=int, default=5, help="Retained for CLI compatibility; main outputs use scaled LS.")
    p.add_argument("--active_tol", type=float, default=5.0e-3)

    p.add_argument("--rollout_amp", type=float, default=0.12)
    p.add_argument("--rollout_steps", type=int, default=40)
    p.add_argument("--rollout_seed", type=int, default=2027)
    p.add_argument("--rollout_q_choice", type=int, default=0, help="If 0, use the largest q in q_list.")
    p.add_argument("--rollout_n_obs_choice", type=int, default=0, help="If 0, use the largest sensor count in obs_list.")
    p.add_argument("--rollout_max_amp", type=float, default=10.0)
    p.add_argument("--rollout_plot_N", type=int, default=260, help="Display-grid resolution for smooth rollout images.")
    p.add_argument("--visual_rollout_T", type=float, default=0.75, help="Presentation rollout horizon shown in the smooth figure.")
    p.add_argument("--visual_rollout_steps", type=int, default=80, help="Short smooth presentation rollout horizon.")
    p.add_argument("--visual_rollout_amp", type=float, default=0.18, help="Amplitude of the smooth presentation initial field.")
    p.add_argument("--visual_low_dim", type=int, default=28, help="Low-pass cutoff used only for the presentation rollout trajectory.")
    p.add_argument("--visual_peanut_sigma", type=float, default=0.14)
    p.add_argument("--visual_channel_x0", type=float, default=-0.45)
    p.add_argument("--visual_channel_x1", type=float, default=0.48)
    p.add_argument("--visual_channel_yshift", type=float, default=0.0)
    p.add_argument("--visual_channel_sx", type=float, default=0.10)
    p.add_argument("--visual_channel_sy", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_frozen_blocks(
        fourier_cutoff=int(args.K),
        laplace_checkpoint=args.laplace_checkpoint,
        transport_checkpoint=args.transport_checkpoint,
        device=args.block_device,
    )
    outdir = ensure_dir(args.outdir)
    np.random.seed(args.seed)

    print(f"[version] {CODE_VERSION}")
    print(f"[args] shape={args.shape}, K={args.K}, rank={args.rank}, obs_list={args.obs_list}, q_list={args.q_list}")

    meta = make_real_trig_basis_metadata(args.K)
    domain = build_domain(args.shape, args.shape, meta, args)
    coeffs = parse_coeffs(args.coeffs, args.shape)
    c_true = coeff_array(coeffs)
    print(f"[known coefficients] {coeffs}")

    trial_rows: List[Dict[str, Any]] = []
    coeff_rows_all: List[Dict[str, Any]] = []

    for q in args.q_list:
        dt_obs = float(args.dt_base) * int(q)
        print("\n" + "=" * 96)
        print(f"[temporal stride] q={q}, observed times=(0, {dt_obs:.3e}, {2.0 * dt_obs:.3e})")
        print("=" * 96)
        for rep in range(int(args.n_repeats)):
            Z0 = make_initial_states(domain, args.n_pairs, args, seed=int(args.seed) + 9999 * rep + 97 * int(q))
            paths_full = np.asarray([integrate_q_path(domain, z0, coeffs, args.dt_base, 2 * int(q)) for z0 in Z0])
            triples = np.stack([paths_full[:, 0, :], paths_full[:, int(q), :], paths_full[:, -1, :]], axis=1)

            exact_simpson_X, exact_simpson_Y = build_simpson_regression_from_triples(domain, triples[:, 0, :], triples[:, 1, :], triples[:, 2, :], dt_obs)
            exact_centered_X, exact_centered_Y = build_centered_fd_regression_from_triples(domain, triples[:, 0, :], triples[:, 1, :], triples[:, 2, :], dt_obs)
            print(
                f"  [rep={rep}] exact centered FD residual={regression_true_residual(exact_centered_X, exact_centered_Y, c_true):.3e}, "
                f"exact Simpson residual={regression_true_residual(exact_simpson_X, exact_simpson_Y, c_true):.3e}"
            )

            for n_obs in args.obs_list:
                trial, coeff_rows = run_one_setting(domain, coeffs, triples, dt_obs, int(n_obs), int(q), rep, args)
                trial_rows.append(trial)
                coeff_rows_all.extend(coeff_rows)
                print(
                    f"    n_obs={int(n_obs):4d}, r_obs={trial['r_obs']:3d}: "
                    f"rec={trial['sensor_reconstruction_rel_l2_mean']:.2e}, "
                    f"ours_err={trial['ours_ls_mean_active_rel_error']:.2e}, "
                    f"fd_err={trial['fdproj_ls_mean_active_rel_error']:.2e}, "
                    f"ratio={trial['fdproj_over_ours_error_ratio']:.1e}, "
                    f"support={trial['ours_ls_support_ok']}"
                )

    agg_rows = aggregate_trials(trial_rows)

    rollout_q = int(args.rollout_q_choice) if int(args.rollout_q_choice) > 0 else max(int(q) for q in args.q_list)
    rollout_n_obs = int(args.rollout_n_obs_choice) if int(args.rollout_n_obs_choice) > 0 else max(int(n) for n in args.obs_list)
    rollout_candidates = [
        row for row in trial_rows
        if int(row["q_stride"]) == rollout_q and int(row["n_obs"]) == rollout_n_obs
    ]
    if not rollout_candidates:
        rollout_candidates = trial_rows[:1]
    rollout_trial = min(rollout_candidates, key=lambda row: float(row["ours_ls_mean_active_rel_error"]))
    coeff_ours = {name: float(rollout_trial[f"ours_coef_{name}"]) for name in FEATURES}
    coeff_fdproj = {name: float(rollout_trial[f"fdproj_coef_{name}"]) for name in FEATURES}
    visual_trial = min(trial_rows, key=lambda row: float(row["ours_ls_mean_active_rel_error"]))
    visual_coeff_ours = {name: float(visual_trial[f"ours_coef_{name}"]) for name in FEATURES}
    visual_coeff_fdproj = {name: float(visual_trial[f"fdproj_coef_{name}"]) for name in FEATURES}
    rollout_summary = plot_rollout_comparison(outdir, domain, coeffs, coeff_ours, coeff_fdproj, args)
    visual_rollout_summary = plot_visual_rollout_comparison(outdir, domain, coeffs, visual_coeff_ours, visual_coeff_fdproj, args)
    error_setting_rows, error_coeff_rows, rollout_error_rows = build_error_analysis_tables(
        agg_rows,
        rollout_trial,
        visual_trial,
        coeffs,
        rollout_summary,
        visual_rollout_summary,
    )

    write_csv(outdir / "summary_by_trial.csv", trial_rows)
    write_csv(outdir / "summary_by_setting.csv", agg_rows)
    write_csv(outdir / "coefficients_by_trial.csv", coeff_rows_all)
    write_csv(outdir / "error_analysis_by_setting.csv", error_setting_rows)
    write_csv(outdir / "identified_coefficients_reference.csv", error_coeff_rows)
    write_csv(outdir / "rollout_error_analysis.csv", rollout_error_rows)
    write_json(outdir / "summary_all.json", {
        "version": CODE_VERSION,
        "args": vars(args),
        "features": FEATURES,
        "coefficients": coeffs,
        "domain": {
            "shape": domain.shape,
            "description": domain.description,
            "ambient_M": len(meta["lambda"]),
            "rank": domain.N.shape[1],
            "boundary_residual": domain.boundary_residual,
            "null_info": {k: v for k, v in domain.null_info.items() if k != "N_raw"},
        },
        "rollout_reference_setting": {
            "q_stride": int(rollout_trial["q_stride"]),
            "n_obs": int(rollout_trial["n_obs"]),
            "rep": int(rollout_trial["rep"]),
        },
        "visual_reference_setting": {
            "q_stride": int(visual_trial["q_stride"]),
            "n_obs": int(visual_trial["n_obs"]),
            "rep": int(visual_trial["rep"]),
        },
        "rollout_summary": rollout_summary,
        "visual_rollout_summary": visual_rollout_summary,
        "analysis_tables": {
            "error_analysis_by_setting_csv": "error_analysis_by_setting.csv",
            "identified_coefficients_reference_csv": "identified_coefficients_reference.csv",
            "rollout_error_analysis_csv": "rollout_error_analysis.csv",
        },
        "trial_rows": trial_rows,
        "aggregate_rows": agg_rows,
    })

    plot_heatmap(outdir, agg_rows, "exact_simpson_ls_mean_active_rel_error_mean", "heatmap_exact_simpson_coef_error.png", "Exact-coordinate Simpson coefficient error", log10=True)
    plot_heatmap(outdir, agg_rows, "ours_ls_mean_active_rel_error_mean", "heatmap_ours_coef_error.png", "Ours: sparse Simpson coefficient error", log10=True)
    plot_heatmap(outdir, agg_rows, "fdproj_ls_mean_active_rel_error_mean", "heatmap_fdproj_coef_error.png", "Sparse FD baseline coefficient error", log10=True)
    plot_heatmap(outdir, agg_rows, "fdproj_over_ours_error_ratio_mean", "heatmap_fdproj_over_ours_ratio.png", "FD baseline / Ours error ratio", log10=True)
    plot_heatmap(outdir, agg_rows, "sensor_reconstruction_rel_l2_mean_mean", "heatmap_projection_error.png", "Sparse-sensor reconstruction error", log10=True)
    plot_curves(outdir, agg_rows)

    print("\n[rollout] reference setting:",
          f"q={rollout_trial['q_stride']}, n_obs={rollout_trial['n_obs']}, rep={rollout_trial['rep']},",
          f"ours_final={rollout_summary['ours_rollout_rel_l2_final']:.2e},",
          f"fd_final={rollout_summary['fdproj_rollout_rel_l2_final']:.2e}")
    print("[visual rollout]",
          f"ours_final={visual_rollout_summary['ours_rollout_rel_l2_final']:.2e},",
          f"fd_final={visual_rollout_summary['fdproj_rollout_rel_l2_final']:.2e}")
    print("[done] saved to", outdir)


if __name__ == "__main__":
    main()
