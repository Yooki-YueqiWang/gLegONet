#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manufactured benchmark for logistic reaction-diffusion on a bunny-head domain.

This version uses the same Section-3 affine Robin construction on a smooth
bunny-head boundary, but the reported rollout error is measured against the
prescribed analytic manufactured solution, not against its projection onto the
boundary-adapted reduced space.  The initial condition is still represented by
the Robin-compatible reduced projection of the analytic field, so the initial
reported error is the genuine reduced-space projection error to the analytic
solution.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib.path import Path as MplPath
import scipy.linalg as sla
import torch

from block_checkpoints import load_laplace_operator

DTYPE = torch.float64


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def weighted_rel_l2(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    num = math.sqrt(float(np.sum(w * (u - v) ** 2)))
    den = math.sqrt(float(np.sum(w * v * v))) + 1e-14
    return float(num / den)



def _rms_from_sq_count(sq_sum: float, count: int) -> float:
    return float(np.sqrt(float(sq_sum) / max(int(count), 1)))


def _write_table_metrics_json(outdir: str, summary: Dict[str, Any], rel_arr: np.ndarray,
                              bc_arr: np.ndarray, solution_sq_sum: float,
                              solution_count: int, extra: Dict[str, Any] | None = None) -> None:
    bc_rms = float(np.sqrt(float(np.mean(np.asarray(bc_arr, dtype=np.float64) ** 2))))
    sol_rms = _rms_from_sq_count(solution_sq_sum, solution_count)
    metrics = {
        "method": "Ours",
        "train_time_sec": None,
        "inference_time_sec": summary.get("total_time_sec", summary.get("inference_time_sec", None)),
        "final_rel_l2": float(rel_arr[-1]),
        "mean_rel_l2": float(np.mean(rel_arr)),
        "max_rel_l2": float(np.max(rel_arr)),
        "E_boundary_rms": bc_rms,
        "solution_rms": sol_rms,
        "E_boundary_sc": float(bc_rms / (sol_rms + 1e-14)),
    }
    metrics.update({k: v for k, v in summary.items() if k not in metrics})
    if extra:
        metrics.update(extra)
    path = os.path.join(outdir, "table_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics), f, indent=2)
    print(f"[table] wrote {path}")


def _write_scalar_rollout_csv(out_path: str, x: np.ndarray, y: np.ndarray,
                              field_records: List[Dict[str, Any]],
                              times: np.ndarray, rel_arr: np.ndarray) -> None:
    header = [
        "row_type", "time_index", "time", "point_index", "x", "y",
        "reference_u", "pred_u", "pointwise_error", "relative_l2_error",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for k, t in enumerate(times):
            f.write(f"history,{k},{float(t):.17g},,,,,,,{float(rel_arr[k]):.17g}\n")
        for rec in field_records:
            k = int(rec["k"])
            t = float(rec["t"])
            ref = np.asarray(rec["ref"], dtype=np.float64).reshape(-1)
            pred = np.asarray(rec["pred"], dtype=np.float64).reshape(-1)
            for q in range(ref.size):
                err = pred[q] - ref[q]
                f.write(
                    f"field,{k},{t:.17g},{q},{float(x[q]):.17g},{float(y[q]):.17g},"
                    f"{float(ref[q]):.17g},{float(pred[q]):.17g},{float(err):.17g},{float(rec['rel']):.17g}\n"
                )
    print(f"[csv] wrote {out_path}")

def build_trig_basis_meta(K: int, radial_truncation: bool = True) -> List[Tuple[str, int, int]]:
    idx: List[Tuple[int, int]] = []
    for k in range(K + 1):
        for l in range(K + 1):
            if (not radial_truncation) or (k * k + l * l <= K * K):
                idx.append((k, l))
    meta: List[Tuple[str, int, int]] = []
    for k, l in idx:
        meta.append(("coscos", k, l))
        if l > 0:
            meta.append(("cossin", k, l))
        if k > 0:
            meta.append(("sincos", k, l))
        if k > 0 and l > 0:
            meta.append(("sinsin", k, l))
    return meta


def eval_trig_basis_from_meta(
    x: np.ndarray, y: np.ndarray, meta: List[Tuple[str, int, int]], L: float = 1.0
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Phi = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    ax = np.pi * x / L
    ay = np.pi * y / L
    for j, (kind, k, l) in enumerate(meta):
        if kind == "coscos":
            Phi[:, j] = np.cos(k * ax) * np.cos(l * ay)
        elif kind == "cossin":
            Phi[:, j] = np.cos(k * ax) * np.sin(l * ay)
        elif kind == "sincos":
            Phi[:, j] = np.sin(k * ax) * np.cos(l * ay)
        elif kind == "sinsin":
            Phi[:, j] = np.sin(k * ax) * np.sin(l * ay)
        else:
            raise ValueError(kind)
    return Phi


def eval_trig_basis_grad_from_meta(
    x: np.ndarray, y: np.ndarray, meta: List[Tuple[str, int, int]], L: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dphix = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    dphiy = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    ax = np.pi * x / L
    ay = np.pi * y / L
    sf = np.pi / L
    for j, (kind, k, l) in enumerate(meta):
        kf = float(k)
        lf = float(l)
        if kind == "coscos":
            dphix[:, j] = -kf * sf * np.sin(kf * ax) * np.cos(lf * ay)
            dphiy[:, j] = -lf * sf * np.cos(kf * ax) * np.sin(lf * ay)
        elif kind == "cossin":
            dphix[:, j] = -kf * sf * np.sin(kf * ax) * np.sin(lf * ay)
            dphiy[:, j] = +lf * sf * np.cos(kf * ax) * np.cos(lf * ay)
        elif kind == "sincos":
            dphix[:, j] = +kf * sf * np.cos(kf * ax) * np.cos(lf * ay)
            dphiy[:, j] = -lf * sf * np.sin(kf * ax) * np.sin(lf * ay)
        elif kind == "sinsin":
            dphix[:, j] = +kf * sf * np.cos(kf * ax) * np.sin(lf * ay)
            dphiy[:, j] = +lf * sf * np.sin(kf * ax) * np.cos(lf * ay)
        else:
            raise ValueError(kind)
    return dphix, dphiy


def laplace_diag_from_meta(meta: List[Tuple[str, int, int]], L: float = 1.0) -> np.ndarray:
    sf2 = (np.pi / L) ** 2
    return np.array([-sf2 * float(k * k + l * l) for _, k, l in meta], dtype=np.float64)


def build_grid_from_mask(mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], Nxy: int) -> Dict[str, Any]:
    x = np.linspace(-1.0, 1.0, Nxy, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, Nxy, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = mask_fn(X, Y)
    h = x[1] - x[0]
    return {
        "X": X,
        "Y": Y,
        "mask": mask,
        "x_in": X[mask].copy(),
        "y_in": Y[mask].copy(),
        "weight": (h * h) * np.ones(int(mask.sum()), dtype=np.float64),
        "extent": (-1.0, 1.0, -1.0, 1.0),
    }


def tangent_space_from_boundary(C: np.ndarray, tau_rel: float) -> Dict[str, Any]:
    U, S, Vh = np.linalg.svd(C, full_matrices=True)
    tau = tau_rel * max(float(np.max(S)), 1.0)
    r = int(np.sum(S > tau))
    N = Vh[r:, :].T.copy()
    return {"U": U, "S": S, "Vh": Vh, "rank": r, "tau": tau, "N": N, "null_dim": int(N.shape[1])}


def particular_solution_from_svd(sec: Dict[str, Any], d: np.ndarray) -> np.ndarray:
    r = sec["rank"]
    if r == 0:
        return np.zeros(sec["Vh"].shape[1], dtype=np.float64)
    U = sec["U"][:, :r]
    S = sec["S"][:r]
    V = sec["Vh"][:r, :].T
    return V @ ((U.T @ d) / S)


def build_m_minimal_lift_operator(C: np.ndarray, M: np.ndarray, lam: float) -> np.ndarray:
    """Return L such that a_bc=L d is the M-minimal Robin lift.

    The lift solves
        min_a 0.5 a^T (M+lam I) a  subject to C a=d.
    It is assembled as L=A^{-1} C^T (C A^{-1} C^T)^{-1}, where
    A=M+lam I.  This avoids repeated large KKT solves during time stepping.
    """
    Mdim = M.shape[0]
    A = 0.5 * (M + M.T) + float(lam) * np.eye(Mdim)
    try:
        cf = sla.cho_factor(A, lower=True, check_finite=False)
        X = sla.cho_solve(cf, C.T, check_finite=False)  # A^{-1} C^T
    except Exception:
        X = sla.solve(A, C.T, assume_a="sym", check_finite=False)
    H = C @ X
    H = 0.5 * (H + H.T) + 1.0e-12 * np.eye(H.shape[0])
    try:
        hf = sla.cho_factor(H, lower=True, check_finite=False)
        Y = sla.cho_solve(hf, np.eye(H.shape[0]), check_finite=False)
    except Exception:
        Y = sla.pinvh(H, check_finite=False)
    return X @ Y


def build_svd_m_minimal_lift_operator(sec: Dict[str, Any], M: np.ndarray) -> np.ndarray:
    """Assemble the exact M-minimal Robin lift from the SVD null-space split."""
    r = int(sec["rank"])
    ncoef = int(sec["Vh"].shape[1])
    nbc = int(sec["U"].shape[0])
    if r == 0:
        return np.zeros((ncoef, nbc), dtype=np.float64)
    U = sec["U"][:, :r]
    S = sec["S"][:r]
    V = sec["Vh"][:r, :].T
    P = (V / S[None, :]) @ U.T
    N_raw = sec["N"]
    if N_raw.size == 0:
        return P
    G = 0.5 * (N_raw.T @ M @ N_raw + (N_raw.T @ M @ N_raw).T)
    G = G + 1.0e-12 * max(float(np.trace(G)) / max(G.shape[0], 1), 1.0) * np.eye(G.shape[0])
    MP = N_raw.T @ M @ P
    try:
        cf = sla.cho_factor(G, lower=True, check_finite=False)
        correction = sla.cho_solve(cf, MP, check_finite=False)
    except Exception:
        correction = sla.solve(G, MP, assume_a="sym", check_finite=False)
    return P - N_raw @ correction


def m_orthonormalize(N_raw: np.ndarray, M: np.ndarray, max_rank: int | None) -> Tuple[np.ndarray, np.ndarray]:
    G = 0.5 * (N_raw.T @ M @ N_raw + (N_raw.T @ M @ N_raw).T)
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    keep = evals > 1e-11 * max(float(evals[0]), 1.0)
    if max_rank is not None and max_rank > 0:
        ids = np.where(keep)[0][:max_rank]
        keep2 = np.zeros_like(keep, dtype=bool)
        keep2[ids] = True
        keep = keep2
    T = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    return N_raw @ T, evals[keep]


def weighted_lstsq_fit(Phi_r: np.ndarray, values: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    A = Phi_r.T @ (Phi_r * w[:, None]) + lam * np.eye(Phi_r.shape[1])
    b = Phi_r.T @ (w * values)
    return np.linalg.solve(0.5 * (A + A.T), b)


def bc_violation(a: np.ndarray, C: np.ndarray, d: np.ndarray | None = None) -> float:
    rhs = 0.0 if d is None else d
    return float(np.linalg.norm(C @ a - rhs) / (np.linalg.norm(a) + np.linalg.norm(rhs) + 1e-14))


def rk4_step(z: np.ndarray, t: float, dt: float, rhs_fn) -> np.ndarray:
    k1 = rhs_fn(z, t)
    k2 = rhs_fn(z + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_fn(z + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_fn(z + dt * k3, t + dt)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lift_to_grid(grid: Dict[str, Any], values: np.ndarray) -> np.ndarray:
    Z = np.full_like(grid["X"], np.nan, dtype=np.float64)
    Z[grid["mask"]] = values
    return Z


def plot_error_curve(times: np.ndarray, rel: np.ndarray, bc: np.ndarray, outpath: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogy(times, rel, label="field")
    ax.semilogy(times, bc, label="boundary")
    ax.set_xlabel("time")
    ax.set_ylabel("rel err")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_curve(
    times: np.ndarray,
    curves: List[Tuple[str, np.ndarray]],
    outpath: str,
    title: str,
    ylabel: str,
    semilogy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, vals in curves:
        if semilogy:
            ax.semilogy(times, vals, label=label)
        else:
            ax.plot(times, vals, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_change_from_initial(
    grid: Dict[str, Any],
    snaps: Dict[int, Dict[str, Any]],
    outpath: str,
    boundaries: List[Tuple[np.ndarray, np.ndarray]],
    title: str,
) -> None:
    ids = sorted(snaps.keys())
    if not ids:
        return
    u0 = snaps[ids[0]]["pred"]
    diffs = [snaps[k]["pred"] - u0 for k in ids]
    emax = max(float(np.max(np.abs(d))) for d in diffs) + 1e-14
    fig, axes = plt.subplots(1, len(ids), figsize=(3.8 * len(ids), 3.4), constrained_layout=True)
    if len(ids) == 1:
        axes = [axes]
    for j, k in enumerate(ids):
        Z = lift_to_grid(grid, diffs[j])
        im = axes[j].imshow(
            Z.T,
            origin="lower",
            extent=grid["extent"],
            cmap="coolwarm",
            vmin=-emax,
            vmax=emax,
            aspect="equal",
        )
        for bx, by in boundaries:
            axes[j].plot(bx, by, "k-", lw=1.0)
        axes[j].set_xlim(-1, 1)
        axes[j].set_ylim(-1, 1)
        axes[j].set_title(f"$u(t)-u(0)$, t={snaps[k]['t']:.3f}")
        fig.colorbar(im, ax=axes[j], fraction=0.046)
    fig.suptitle(title)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_scalar_snapshots(
    grid: Dict[str, Any],
    snaps: Dict[int, Dict[str, Any]],
    outpath: str,
    boundaries: List[Tuple[np.ndarray, np.ndarray]],
    title: str,
) -> None:
    ids = sorted(snaps.keys())
    fig, axes = plt.subplots(3, len(ids), figsize=(4.0 * len(ids), 9.0), constrained_layout=True)
    if len(ids) == 1:
        axes = axes.reshape(3, 1)
    all_vals: List[np.ndarray] = []
    all_errs: List[np.ndarray] = []
    for k in ids:
        all_vals.append(snaps[k]["pred"])
        all_vals.append(snaps[k]["ref"])
        all_errs.append(snaps[k]["pred"] - snaps[k]["ref"])
    vmin = float(min(np.min(v) for v in all_vals))
    vmax = float(max(np.max(v) for v in all_vals))
    emax = float(max(np.max(np.abs(e)) for e in all_errs)) + 1e-14
    for j, k in enumerate(ids):
        pred = lift_to_grid(grid, snaps[k]["pred"])
        ref = lift_to_grid(grid, snaps[k]["ref"])
        err = lift_to_grid(grid, snaps[k]["pred"] - snaps[k]["ref"])
        im0 = axes[0, j].imshow(pred.T, origin="lower", extent=grid["extent"], vmin=vmin, vmax=vmax, aspect="equal")
        axes[0, j].set_title(f"predicted solution, t={snaps[k]['t']:.3f}")
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046)
        im1 = axes[1, j].imshow(ref.T, origin="lower", extent=grid["extent"], vmin=vmin, vmax=vmax, aspect="equal")
        axes[1, j].set_title(f"reference solution, t={snaps[k]['t']:.3f}")
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046)
        im2 = axes[2, j].imshow(
            err.T,
            origin="lower",
            extent=grid["extent"],
            cmap="coolwarm",
            vmin=-emax,
            vmax=emax,
            aspect="equal",
        )
        axes[2, j].set_title("pointwise error")
        fig.colorbar(im2, ax=axes[2, j], fraction=0.046)
        for ax in axes[:, j]:
            for bx, by in boundaries:
                ax.plot(bx, by, "k-", lw=1.0)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_domain(grid: Dict[str, Any], outpath: str, boundaries: List[Tuple[np.ndarray, np.ndarray, str]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid["mask"].T.astype(float), origin="lower", extent=grid["extent"], cmap="Greys", alpha=0.35)
    for bx, by, label in boundaries:
        ax.plot(bx, by, lw=1.6, label=label)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Bunny-head domain
# -----------------------------------------------------------------------------


def _angle_diff(theta: np.ndarray, center: float) -> np.ndarray:
    """Periodic signed difference theta-center in [-pi, pi]."""
    return np.arctan2(np.sin(theta - center), np.cos(theta - center))


def _gaussian_bump(theta: np.ndarray, center: float, width: float) -> Tuple[np.ndarray, np.ndarray]:
    d = _angle_diff(theta, center)
    g = np.exp(-0.5 * (d / width) ** 2)
    dg = -(d / (width * width)) * g
    return g, dg


def _bunny_radius_and_derivative(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth star-shaped bunny-head radius and derivative.

    The previous version used two very thin ears.  That made the Robin trace
    difficult to represent with K=22 and produced a poor affine lift.  This
    version keeps the bunny-head visual appearance but uses wider, smoother
    ears and a less pinched cleft, which is better aligned with the fixed
    ambient trigonometric space.
    """
    theta = np.asarray(theta, dtype=np.float64)
    r = 0.475 * np.ones_like(theta)
    dr = np.zeros_like(theta)

    # Two broad ears.  The centers are intentionally separated but not too
    # narrow; this keeps the curve smooth enough for K=22.
    for center, amp, width in [
        (0.5 * np.pi - 0.43, 0.305, 0.255),
        (0.5 * np.pi + 0.43, 0.305, 0.255),
    ]:
        g, dg = _gaussian_bump(theta, center, width)
        r += amp * g
        dr += amp * dg

    # Soft cleft between ears.
    g, dg = _gaussian_bump(theta, 0.5 * np.pi, 0.165)
    r += -0.070 * g
    dr += -0.070 * dg

    # Mild cheeks and chin, all deliberately low frequency.
    for center, amp, width in [
        (np.pi - 0.28, 0.040, 0.36),
        (0.28, 0.040, 0.36),
        (1.5 * np.pi, -0.025, 0.42),
    ]:
        g, dg = _gaussian_bump(theta, center, width)
        r += amp * g
        dr += amp * dg

    r = np.maximum(r, 0.22)
    return r, dr

def boundary_bunny_head(nb: int):
    theta = np.linspace(0.0, 2.0 * np.pi, nb, endpoint=False, dtype=np.float64)
    r, dr = _bunny_radius_and_derivative(theta)

    cx, cy = 0.0, -0.10
    ct = np.cos(theta)
    st = np.sin(theta)
    x = cx + r * ct
    y = cy + r * st

    dx = dr * ct - r * st
    dy = dr * st + r * ct

    # For this counter-clockwise parameterization, (dy,-dx) is the outward normal.
    nx = dy
    ny = -dx
    nn = np.sqrt(nx * nx + ny * ny) + 1e-15
    nx /= nn
    ny /= nn
    return x, y, nx, ny, theta / (2.0 * np.pi)


def mask_fn(X, Y):
    xb, yb, _, _, _ = boundary_bunny_head(2400)
    path = MplPath(np.column_stack([xb, yb]), closed=True)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return path.contains_points(pts, radius=1e-12).reshape(X.shape)


def exact_u_torch(x, y, t, Kc: float, front_speed: float, sharpness: float, ripple_amp: float):
    """Two expanding colonies plus a sweeping invasion front."""
    phase = front_speed * t

    c1x = -0.34 + 0.05 * torch.sin(0.45 * phase)
    c1y = -0.10 + 0.07 * torch.cos(0.60 * phase)
    r1 = 0.18 + 0.18 * (1.0 - torch.exp(-1.6 * t))
    d1 = torch.sqrt(((x - c1x) / 0.42) ** 2 + ((y - c1y) / 0.34) ** 2 + 1e-12)
    colony1 = torch.sigmoid(1.15 * sharpness * (r1 - d1))

    c2x = 0.06 + 0.06 * torch.cos(0.55 * phase + 0.3)
    c2y = 0.24 + 0.05 * torch.sin(0.90 * phase)
    r2 = 0.12 + 0.16 * (1.0 - torch.exp(-2.0 * t))
    d2 = torch.sqrt(((x - c2x) / 0.36) ** 2 + ((y - c2y) / 0.28) ** 2 + 1e-12)
    colony2 = torch.sigmoid(1.05 * sharpness * (r2 - d2))

    front_center = -0.32 + 0.78 * (1.0 - torch.exp(-1.10 * t))
    front = torch.sigmoid(sharpness * (front_center - (x + 0.10 * y)))

    wake = 0.12 * torch.exp(-0.9 * t) * torch.cos(2.6 * np.pi * y - 0.45 * phase)
    ripples = ripple_amp * torch.exp(-1.6 * t) * (
        torch.sin(4.0 * np.pi * x + 0.9 * phase) * torch.cos(3.0 * np.pi * y - 0.6 * phase)
        + 0.55 * torch.cos(2.5 * np.pi * (x - y) + 0.4 * phase)
    )

    cavity = torch.sigmoid(1.10 * sharpness * (0.18 - torch.sqrt(((x - 0.28) / 0.34) ** 2 + ((y + 0.22) / 0.28) ** 2 + 1e-12)))
    raw = -1.10 + 1.80 * front + 1.15 * colony1 + 0.90 * colony2 - 0.55 * cavity + 0.45 * wake + 0.35 * ripples
    return Kc * torch.sigmoid(2.2 * raw)


def exact_force_and_boundary(
    x_np,
    y_np,
    t_scalar,
    *,
    D,
    r_growth,
    Kc,
    kappa,
    front_speed,
    sharpness,
    ripple_amp,
    nx_np=None,
    ny_np=None,
    device=None,
):
    if device is None:
        device = torch.device("cpu")
    x = torch.tensor(x_np, dtype=DTYPE, device=device, requires_grad=True)
    y = torch.tensor(y_np, dtype=DTYPE, device=device, requires_grad=True)
    t = torch.full_like(x, float(t_scalar), requires_grad=True)
    u = exact_u_torch(x, y, t, Kc, front_speed, sharpness, ripple_amp)
    ones = torch.ones_like(u)
    ux = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    uy = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    lap = uxx + uyy
    f = ut - D * lap - r_growth * u * (1.0 - u / Kc)
    g = None
    if nx_np is not None and ny_np is not None:
        nx = torch.tensor(nx_np, dtype=DTYPE, device=device)
        ny = torch.tensor(ny_np, dtype=DTYPE, device=device)
        g = nx * ux + ny * uy + kappa * u
        return u.detach().cpu().numpy(), f.detach().cpu().numpy(), g.detach().cpu().numpy()
    return u.detach().cpu().numpy(), f.detach().cpu().numpy(), None


def snapshot_ids_for(nsteps: int) -> set[int]:
    ids = {0, nsteps // 3, (2 * nsteps) // 3, nsteps}
    return {int(k) for k in ids}


@dataclass
class Args:
    outdir: str
    K: int
    Nx_eval: int
    Nb: int
    Nb_lift: int
    Nb_dense: int
    lift_solver: str
    reduced_rank: int
    tau_rel: float
    T: float
    dt: float
    D: float
    r_growth: float
    Kc: float
    kappa: float
    front_speed: float
    sharpness: float
    ripple_amp: float
    fit_lam: float
    lift_lam: float
    abc_dt_eps: float
    seed: int
    device: str
    log_every: int
    field_time_stride: int
    laplace_checkpoint: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Dynamic bunny-head logistic reaction-diffusion with affine Robin Section 3."
    )
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--Nx_eval", type=int, required=True)
    p.add_argument("--Nb", type=int, required=True)
    p.add_argument("--Nb_lift", type=int, required=True, help="Boundary sample count used to build the affine Robin lift/null-space.")
    p.add_argument("--Nb_dense", type=int, required=True)
    p.add_argument("--lift_solver", type=str, default="normal_eq", choices=("normal_eq", "svd_exact"),
                   help="Affine Robin lift assembly: fast normal-equation solve or more accurate SVD/null-space solve.")
    p.add_argument("--reduced_rank", type=int, required=True)
    p.add_argument("--tau_rel", type=float, default=1e-10)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--D", type=float, required=True)
    p.add_argument("--r_growth", type=float, required=True)
    p.add_argument("--Kc", type=float, required=True)
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--front_speed", type=float, required=True)
    p.add_argument("--sharpness", type=float, required=True)
    p.add_argument("--ripple_amp", type=float, required=True)
    p.add_argument("--fit_lam", type=float, default=1e-9)
    p.add_argument("--lift_lam", type=float, default=1e-10)
    p.add_argument("--abc_dt_eps", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--field_time_stride", type=int, default=1, help="Save full-field CSV rows every this many time steps; use <=0 to save history only.")
    p.add_argument("--laplace_checkpoint", type=str, required=True)
    return Args(**vars(p.parse_args()))


def main():
    args = parse_args()
    wall_time_start = time.perf_counter()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    meta = build_trig_basis_meta(args.K, radial_truncation=True)
    laplace_operator, laplace_diagnostics = load_laplace_operator(meta, args.K, args.laplace_checkpoint)
    grid = build_grid_from_mask(mask_fn, args.Nx_eval)
    x_in, y_in, wq = grid["x_in"], grid["y_in"], grid["weight"]
    Phi = eval_trig_basis_from_meta(x_in, y_in, meta)
    M = Phi.T @ (Phi * wq[:, None])

    xb, yb, nxb, nyb, _ = boundary_bunny_head(args.Nb)
    xd, yd, nxd, nyd, _ = boundary_bunny_head(args.Nb_dense)
    Phib = eval_trig_basis_from_meta(xb, yb, meta)
    dpxb, dpyb = eval_trig_basis_grad_from_meta(xb, yb, meta)
    C = nxb[:, None] * dpxb + nyb[:, None] * dpyb + args.kappa * Phib
    Phid = eval_trig_basis_from_meta(xd, yd, meta)
    dpxd, dpyd = eval_trig_basis_grad_from_meta(xd, yd, meta)
    C_dense = nxd[:, None] * dpxd + nyd[:, None] * dpyd + args.kappa * Phid
    # Build the affine Robin lift and null space on a moderately enriched
    # boundary sample.  This keeps the dense-grid Robin residual much smaller
    # than the original coarse construction while avoiding the large projection
    # floor caused by enforcing the constraint on every dense evaluation point.
    xbc, ybc, nxbc, nybc, _ = boundary_bunny_head(args.Nb_lift)
    Phic = eval_trig_basis_from_meta(xbc, ybc, meta)
    dpxc, dpyc = eval_trig_basis_grad_from_meta(xbc, ybc, meta)
    C_constraint = nxbc[:, None] * dpxc + nybc[:, None] * dpyc + args.kappa * Phic
    sec = tangent_space_from_boundary(C_constraint, args.tau_rel)
    dense_sec = tangent_space_from_boundary(C_dense, args.tau_rel)

    # Stable affine Robin lift.  For the bunny geometry the raw SVD particular
    # solution can have a large interior extension.  The M-minimal lift keeps
    # the nonhomogeneous Robin extension small in L2(Omega), after which the
    # nullspace coordinates represent the physical state.
    if args.lift_solver == "svd_exact":
        lift_op = build_svd_m_minimal_lift_operator(sec, M)
    else:
        lift_op = build_m_minimal_lift_operator(C_constraint, M, args.lift_lam)

    # Keep the well-behaved coarse affine lift, then polish both the
    # nonhomogeneous trace and the homogeneous reduced basis on the dense
    # boundary operator.  This preserves the original analytic MMS while
    # driving the reported dense Robin residual much closer to machine precision.
    dense_polish_op = build_svd_m_minimal_lift_operator(dense_sec, M)
    N_seed, _ = m_orthonormalize(sec["N"], M, args.reduced_rank)
    N_raw = N_seed - dense_polish_op @ (C_dense @ N_seed)
    N, _ = m_orthonormalize(N_raw, M, args.reduced_rank)
    PhiN = Phi @ N

    print(f"[Section3] M={len(meta)} rank(C)={sec['rank']} null={sec['null_dim']} reduced={N.shape[1]} | affine Robin")
    print(
        f"[lift] using {args.lift_solver} Robin lift with lift_lam={args.lift_lam:.1e} "
        f"+ dense polish rank={dense_sec['rank']}"
    )
    plot_domain(grid, os.path.join(args.outdir, "domain.png"), [(xd, yd, "Robin boundary")])

    def d_build(t):
        return exact_force_and_boundary(
            xbc,
            ybc,
            t,
            D=args.D,
            r_growth=args.r_growth,
            Kc=args.Kc,
            kappa=args.kappa,
            front_speed=args.front_speed,
            sharpness=args.sharpness,
            ripple_amp=args.ripple_amp,
            nx_np=nxbc,
            ny_np=nybc,
            device=device,
        )[2]

    def d_dense(t):
        return exact_force_and_boundary(
            xd,
            yd,
            t,
            D=args.D,
            r_growth=args.r_growth,
            Kc=args.Kc,
            kappa=args.kappa,
            front_speed=args.front_speed,
            sharpness=args.sharpness,
            ripple_amp=args.ripple_amp,
            nx_np=nxd,
            ny_np=nyd,
            device=device,
        )[2]

    def abc(t):
        a_coarse = lift_op @ d_build(t)
        return a_coarse + dense_polish_op @ (d_dense(t) - C_dense @ a_coarse)

    def abc_dot(t):
        h = args.abc_dt_eps
        if t - h < 0:
            return (abc(t + h) - abc(t)) / h
        return (abc(t + h) - abc(t - h)) / (2.0 * h)

    def analytic_values_and_source_in_domain(t: float) -> Tuple[np.ndarray, np.ndarray]:
        u_val, f_val, _ = exact_force_and_boundary(
            x_in,
            y_in,
            float(t),
            D=args.D,
            r_growth=args.r_growth,
            Kc=args.Kc,
            kappa=args.kappa,
            front_speed=args.front_speed,
            sharpness=args.sharpness,
            ripple_amp=args.ripple_amp,
            device=device,
        )
        return u_val, f_val

    def analytic_values_in_domain(t: float) -> np.ndarray:
        return analytic_values_and_source_in_domain(float(t))[0]

    def analytic_source_in_domain(t: float) -> np.ndarray:
        return analytic_values_and_source_in_domain(float(t))[1]

    def project_exact_to_reduced(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Project the analytic field at time t into a_bc(t)+range(N).

        Returns z_ref, a_ref, u_projected, u_analytic, and the projection error
        relative to the original analytic field.  The rollout starts from this
        projected state, but all reported field errors use u_analytic as the
        reference solution.
        """
        u_true = analytic_values_in_domain(float(t))
        at = abc(float(t))
        z_ref = weighted_lstsq_fit(PhiN, u_true - Phi @ at, wq, args.fit_lam)
        a_ref = at + N @ z_ref
        u_proj = Phi @ a_ref
        proj_rel = weighted_rel_l2(u_proj, u_true, wq)
        return z_ref, a_ref, u_proj, u_true, proj_rel

    def z_ref_at(t: float) -> np.ndarray:
        return project_exact_to_reduced(float(t))[0]

    def z_ref_dot(t: float) -> np.ndarray:
        h = args.abc_dt_eps
        if t - h < 0.0:
            return (z_ref_at(t + h) - z_ref_at(t)) / h
        if t + h > args.T:
            return (z_ref_at(t) - z_ref_at(t - h)) / h
        return (z_ref_at(t + h) - z_ref_at(t - h)) / (2.0 * h)

    z, a_init_ref, u_init_ref, u_init_analytic, init_proj_rel = project_exact_to_reduced(0.0)
    init_bc = bc_violation(a_init_ref, C_dense, d_dense(0.0))
    print(
        f"[init reduced projection] rel_to_analytic={init_proj_rel:.3e} "
        f"rel_to_projected=0.000e+00 robin_bc={init_bc:.3e} "
        f"max|u0|={np.max(np.abs(u_init_ref)):.3e}"
    )

    nsteps = int(round(args.T / args.dt))
    times = np.linspace(0.0, args.T, nsteps + 1)
    snap_ids = snapshot_ids_for(nsteps)
    snaps: Dict[int, Dict[str, Any]] = {}
    rel_hist: List[float] = []
    rel_projected_hist: List[float] = []
    proj_hist: List[float] = []
    bc_hist: List[float] = []
    change_hist: List[float] = []
    max_hist: List[float] = []
    field_records: List[Dict[str, Any]] = []
    solution_sq_sum = 0.0
    solution_count = 0

    def reduced_rhs(zv: np.ndarray, t: float) -> np.ndarray:
        """Reduced Galerkin RHS for the analytic MMS PDE.

        The source term f is computed from the analytic manufactured solution,
        so the continuum reference is the analytic field.  This is different
        from the previous projected-MMS version, where an extra coefficient
        forcing was constructed to make the projected trajectory exact.
        """
        at = abc(t)
        adot = abc_dot(t)
        a = at + N @ zv
        u = Phi @ a
        lap = Phi @ (laplace_operator @ a)
        f_val = analytic_source_in_domain(float(t))
        rhs_values = args.D * lap + args.r_growth * u * (1.0 - u / args.Kc) + f_val - Phi @ adot
        return weighted_lstsq_fit(PhiN, rhs_values, wq, args.fit_lam)

    def rhs_fn(zv, t):
        return reduced_rhs(zv, float(t))

    for k, t in enumerate(times):
        t_float = float(t)
        at = abc(t_float)
        a = at + N @ z
        pred = Phi @ a
        z_ref, a_ref, projected_ref, analytic_ref, proj_rel = project_exact_to_reduced(t_float)
        rel = weighted_rel_l2(pred, analytic_ref, wq)
        rel_projected = weighted_rel_l2(pred, projected_ref, wq)
        bc = bc_violation(a, C_dense, d_dense(t_float))
        change = weighted_rel_l2(pred, u_init_analytic, wq)
        max_u = float(np.max(np.abs(pred)))
        solution_sq_sum += float(np.sum(analytic_ref * analytic_ref))
        solution_count += int(analytic_ref.size)
        if args.field_time_stride > 0 and (k % args.field_time_stride == 0 or k == nsteps):
            field_records.append({"k": int(k), "t": t_float, "ref": analytic_ref.copy(), "pred": pred.copy(), "rel": float(rel)})
        rel_hist.append(rel)
        rel_projected_hist.append(rel_projected)
        proj_hist.append(proj_rel)
        bc_hist.append(bc)
        change_hist.append(change)
        max_hist.append(max_u)
        if k in snap_ids:
            snaps[k] = {"pred": pred.copy(), "ref": analytic_ref.copy(), "t": t_float}
        if k < nsteps:
            z = rk4_step(z, t_float, args.dt, rhs_fn)
        if k % max(args.log_every, 1) == 0 or k == nsteps:
            print(
                f"[rollout] step={k:04d}/{nsteps:04d} t={t:.3f} "
                f"rel_exact={rel:.3e} rel_projected={rel_projected:.3e} "
                f"projection_floor={proj_rel:.3e} robin_bc={bc:.3e} "
                f"change={change:.3e} max|u|={max_u:.3e}"
            )

    rel_arr = np.array(rel_hist)
    rel_projected_arr = np.array(rel_projected_hist)
    proj_arr = np.array(proj_hist)
    bc_arr = np.array(bc_hist)
    change_arr = np.array(change_hist)
    max_arr = np.array(max_hist)

    plot_error_curve(
        times,
        rel_arr,
        bc_arr,
        os.path.join(args.outdir, "error_curves.png"),
        "MMS-III bunny-head logistic RD, error to analytic solution",
    )
    plot_curve(
        times,
        [("rollout error to projected trajectory", rel_projected_arr), ("projection floor to analytic field", proj_arr)],
        os.path.join(args.outdir, "projected_diagnostics.png"),
        "Projected-space diagnostics",
        "relative error",
        semilogy=True,
    )

    plot_scalar_snapshots(
        grid,
        snaps,
        os.path.join(args.outdir, "snapshots_section3_vs_exact.png"),
        [(xd, yd)],
        "MMS-III bunny-head logistic RD: predicted vs analytic exact",
    )
    plot_change_from_initial(
        grid,
        snaps,
        os.path.join(args.outdir, "change_from_initial.png"),
        [(xd, yd)],
        "MMS-III bunny-head logistic RD: change from analytic initial field",
    )
    plot_curve(
        times,
        [("relative change from initial", change_arr)],
        os.path.join(args.outdir, "change_history.png"),
        "Change from initial condition",
        "relative change",
    )
    plot_curve(
        times,
        [("max |u|", max_arr)],
        os.path.join(args.outdir, "max_history.png"),
        "Maximum amplitude history",
        "max |u|",
    )

    total_wall_time_sec = time.perf_counter() - wall_time_start
    summary = {
        "args": to_jsonable(vars(args)),
        "inference_time_sec": float(total_wall_time_sec),
        "total_time_sec": float(total_wall_time_sec),
        "basis_dim": len(meta),
        "reduced_rank": int(N.shape[1]),
        "error_reference": "analytic_manufactured_solution",
        "final_rel_l2_to_analytic": float(rel_arr[-1]),
        "mean_rel_l2_to_analytic": float(np.mean(rel_arr)),
        "max_rel_l2_to_analytic": float(np.max(rel_arr)),
        "initial_rel_l2_to_analytic": float(rel_arr[0]),
        "final_rel_l2_to_projected_trajectory": float(rel_projected_arr[-1]),
        "mean_rel_l2_to_projected_trajectory": float(np.mean(rel_projected_arr)),
        "max_rel_l2_to_projected_trajectory": float(np.max(rel_projected_arr)),
        "initial_projection_error_to_analytic": float(proj_arr[0]),
        "final_projection_error_to_analytic": float(proj_arr[-1]),
        "max_projection_error_to_analytic": float(np.max(proj_arr)),
        "final_robin_bc": float(bc_arr[-1]),
        "laplace_checkpoint_diagnostics": laplace_diagnostics,
        "final_change_from_initial_analytic": float(change_arr[-1]),
        "max_change_from_initial_analytic": float(np.max(change_arr)),
        "max_abs_u": float(np.max(max_arr)),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    _write_table_metrics_json(
        args.outdir, summary, rel_arr, bc_arr, solution_sq_sum, solution_count,
        extra={
            "final_rel_l2_projected": float(rel_projected_arr[-1]),
            "mean_rel_l2_projected": float(np.mean(rel_projected_arr)),
            "max_rel_l2_projected": float(np.max(rel_projected_arr)),
            "final_projection_error_to_analytic": float(proj_arr[-1]),
            "max_projection_error_to_analytic": float(np.max(proj_arr)),
        },
    )
    _write_scalar_rollout_csv(
        os.path.join(args.outdir, "rollout_fields_and_relerr.csv"),
        x_in, y_in, field_records, times, rel_arr,
    )
    print(
        f"[summary] final_rel_exact={rel_arr[-1]:.3e} max_rel_exact={rel_arr.max():.3e} "
        f"final_rel_projected={rel_projected_arr[-1]:.3e} "
        f"max_projection_floor={proj_arr.max():.3e} final_bc={bc_arr[-1]:.3e} "
        f"final_change={change_arr[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
