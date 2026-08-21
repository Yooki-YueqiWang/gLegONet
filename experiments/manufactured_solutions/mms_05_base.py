#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section-3 manufactured-solution benchmark for scalar 2D Burgers on a new
pinwheel-shell irregular domain.

The operator blocks are still assembled on the square trigonometric basis and
then restricted to an irregular interior domain through Section-3 boundary
constraints, matching the pattern used in the earlier MMS scripts.
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
import torch

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
            dphiy[:, j] = lf * sf * np.cos(kf * ax) * np.cos(lf * ay)
        elif kind == "sincos":
            dphix[:, j] = kf * sf * np.cos(kf * ax) * np.cos(lf * ay)
            dphiy[:, j] = -lf * sf * np.sin(kf * ax) * np.sin(lf * ay)
        elif kind == "sinsin":
            dphix[:, j] = kf * sf * np.cos(kf * ax) * np.sin(lf * ay)
            dphiy[:, j] = lf * sf * np.sin(kf * ax) * np.cos(lf * ay)
        else:
            raise ValueError(kind)
    return dphix, dphiy


def laplace_diag_from_meta(meta: List[Tuple[str, int, int]], L: float = 1.0) -> np.ndarray:
    sf2 = (np.pi / L) ** 2
    return np.array([-sf2 * float(k * k + l * l) for _, k, l in meta], dtype=np.float64)


def numerical_grad_q(
    q_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qx = (q_fn(x + eps, y) - q_fn(x - eps, y)) / (2.0 * eps)
    qy = (q_fn(x, y + eps) - q_fn(x, y - eps)) / (2.0 * eps)
    return qx, qy


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
    boundary_xy: Tuple[np.ndarray, np.ndarray],
    title: str,
) -> None:
    ids = sorted(snaps.keys())
    if not ids:
        return
    bx, by = boundary_xy
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
    boundary_xy: Tuple[np.ndarray, np.ndarray],
    title: str,
) -> None:
    ids = sorted(snaps.keys())
    fig, axes = plt.subplots(3, len(ids), figsize=(4.0 * len(ids), 9.0), constrained_layout=True)
    if len(ids) == 1:
        axes = axes.reshape(3, 1)
    bx, by = boundary_xy
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


def re_z3_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**3 - 3.0 * x * y**2


def im_z5_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 5.0 * x**4 * y - 10.0 * x**2 * y**3 + y**5


def re_z6_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**6 - 15.0 * x**4 * y**2 + 15.0 * x**2 * y**4 - y**6


def re_z3_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x**3 - 3.0 * x * y**2


def im_z5_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 5.0 * x**4 * y - 10.0 * x**2 * y**3 + y**5


def re_z6_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x**6 - 15.0 * x**4 * y**2 + 15.0 * x**2 * y**4 - y**6


def q_np(x, y):
    r2 = x * x + y * y
    return 0.60**8 - r2**4 + 0.30 * re_z6_np(x, y) - 0.10 * im_z5_np(x, y) + 0.12 * r2 * re_z3_np(x, y)


def q_torch(x, y):
    r2 = x * x + y * y
    return 0.60**8 - r2**4 + 0.30 * re_z6_torch(x, y) - 0.10 * im_z5_torch(x, y) + 0.12 * r2 * re_z3_torch(x, y)


def mask_fn(X, Y):
    return q_np(X, Y) >= 0.0


def boundary_pinwheel_shell(nb: int):
    th = np.linspace(0.0, 2.0 * np.pi, nb, endpoint=False)
    R = np.empty_like(th)
    for j, theta in enumerate(th):
        lo = 0.0
        hi = 1.2
        q_hi = q_np(hi * np.cos(theta), hi * np.sin(theta))
        while q_hi > 0.0 and hi < 2.5:
            hi *= 1.15
            q_hi = q_np(hi * np.cos(theta), hi * np.sin(theta))
        if q_hi > 0.0:
            raise RuntimeError("Failed to bracket boundary root for pinwheel-shell domain.")
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            q_mid = q_np(mid * np.cos(theta), mid * np.sin(theta))
            if q_mid > 0.0:
                lo = mid
            else:
                hi = mid
        R[j] = 0.5 * (lo + hi)
    x = R * np.cos(th)
    y = R * np.sin(th)
    qx, qy = numerical_grad_q(q_np, x, y)
    nx = -qx
    ny = -qy
    nn = np.sqrt(nx * nx + ny * ny) + 1e-15
    return x, y, nx / nn, ny / nn


def exact_u_torch(x, y, t, amp: float, omega: float):
    """Manufactured Burgers field with sweeping fronts and transported wakes."""
    q = q_torch(x, y)
    geom = q / 0.45
    phase = omega * t

    ang1 = 0.35 + 0.30 * torch.sin(0.55 * t)
    ang2 = -0.95 + 0.22 * torch.cos(0.70 * t)
    s1 = x * torch.cos(ang1) + y * torch.sin(ang1)
    s2 = x * torch.cos(ang2) - y * torch.sin(ang2)

    sharp1 = 4.0 + 0.9 * torch.sin(0.60 * t) ** 2
    sharp2 = 3.4 + 0.8 * torch.cos(0.75 * t) ** 2
    shift1 = -0.28 + 0.52 * t + 0.06 * torch.sin(0.40 * phase)
    shift2 = 0.12 * torch.cos(0.35 * phase) - 0.08

    front_train = torch.tanh(sharp1 * (s1 - shift1)) - 0.85 * torch.tanh((sharp1 - 0.45) * (s1 - shift1 - 0.24))
    cross_front = torch.tanh(sharp2 * (s2 - shift2))

    xc1 = 0.26 * torch.cos(0.52 * phase + 0.3)
    yc1 = 0.20 * torch.sin(0.60 * phase - 0.4)
    xc2 = -0.18 + 0.16 * torch.sin(0.48 * phase + 0.9)
    yc2 = 0.24 * torch.cos(0.44 * phase + 0.1)
    blob1 = torch.exp(-(((x - xc1) / 0.23) ** 2 + ((y - yc1) / 0.17) ** 2))
    blob2 = torch.exp(-(((x - xc2) / 0.18) ** 2 + ((y - yc2) / 0.22) ** 2))

    wake = torch.sin(2.0 * np.pi * (0.72 * x + 0.30 * y) - 0.75 * phase) * torch.exp(-((s1 + 0.04 * torch.cos(t)) / 0.44) ** 2)
    envelope = 1.00 + 0.24 * torch.sin(0.35 * phase) + 0.18 * torch.cos(0.80 * t)
    profile = 0.72 * front_train + 0.35 * cross_front + 0.52 * blob1 - 0.42 * blob2 + 0.18 * wake
    return amp * envelope * geom * profile


def exact_and_force(x_np, y_np, t_scalar, *, nu, amp, omega, device):
    x = torch.tensor(x_np, dtype=DTYPE, device=device, requires_grad=True)
    y = torch.tensor(y_np, dtype=DTYPE, device=device, requires_grad=True)
    t = torch.full_like(x, float(t_scalar), requires_grad=True)
    u = exact_u_torch(x, y, t, amp, omega)
    ones = torch.ones_like(u)
    ux = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    uy = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    lap = uxx + uyy
    f = ut - nu * lap + u * ux + u * uy
    return u.detach().cpu().numpy(), f.detach().cpu().numpy()


def snapshot_ids_for(nsteps: int) -> set[int]:
    ids = {0, nsteps // 3, (2 * nsteps) // 3, nsteps}
    return {int(k) for k in ids}


@dataclass
class Args:
    outdir: str
    K: int
    Nx_eval: int
    Nb: int
    Nb_dense: int
    reduced_rank: int
    tau_rel: float
    T: float
    dt: float
    nu: float
    amp: float
    omega: float
    fit_lam: float
    seed: int
    device: str
    log_every: int
    field_time_stride: int
    laplace_checkpoint: str
    transport_checkpoint: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="MMS-V scalar 2D Burgers on a pinwheel-shell irregular domain with visibly dynamic rollout."
    )
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--Nx_eval", type=int, required=True)
    p.add_argument("--Nb", type=int, required=True)
    p.add_argument("--Nb_dense", type=int, required=True)
    p.add_argument("--reduced_rank", type=int, required=True)
    p.add_argument("--tau_rel", type=float, default=1e-10)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--nu", type=float, required=True)
    p.add_argument("--amp", type=float, required=True)
    p.add_argument("--omega", type=float, required=True)
    p.add_argument("--fit_lam", type=float, default=1e-11)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--field_time_stride", type=int, default=1, help="Save full-field CSV rows every this many time steps; use <=0 to save history only.")
    p.add_argument("--laplace_checkpoint", type=str, required=True)
    p.add_argument("--transport_checkpoint", type=str, required=True)
    return Args(**vars(p.parse_args()))


def main():
    args = parse_args()
    wall_time_start = time.perf_counter()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    meta = build_trig_basis_meta(args.K, radial_truncation=True)
    lap_diag = laplace_diag_from_meta(meta)
    grid = build_grid_from_mask(mask_fn, args.Nx_eval)
    x_in, y_in, wq = grid["x_in"], grid["y_in"], grid["weight"]
    Phi = eval_trig_basis_from_meta(x_in, y_in, meta)
    dPhix, dPhiy = eval_trig_basis_grad_from_meta(x_in, y_in, meta)
    M = Phi.T @ (Phi * wq[:, None])

    xb, yb, _, _ = boundary_pinwheel_shell(args.Nb)
    xd, yd, _, _ = boundary_pinwheel_shell(args.Nb_dense)
    C = eval_trig_basis_from_meta(xb, yb, meta)
    C_dense = eval_trig_basis_from_meta(xd, yd, meta)
    sec = tangent_space_from_boundary(C, args.tau_rel)
    N, _ = m_orthonormalize(sec["N"], M, args.reduced_rank)
    PhiN = Phi @ N

    print(f"[Section3] M={len(meta)} rank(C)={sec['rank']} null={sec['null_dim']} reduced={N.shape[1]}")
    plot_domain(grid, os.path.join(args.outdir, "domain.png"), [(xd, yd, "Dirichlet boundary")])

    u_init, _ = exact_and_force(
        x_in, y_in, 0.0, nu=args.nu, amp=args.amp, omega=args.omega, device=device
    )
    z = weighted_lstsq_fit(PhiN, u_init, wq, args.fit_lam)

    nsteps = int(round(args.T / args.dt))
    times = np.linspace(0.0, args.T, nsteps + 1)
    snap_ids = snapshot_ids_for(nsteps)
    snaps: Dict[int, Dict[str, Any]] = {}
    rel_hist: List[float] = []
    bc_hist: List[float] = []
    change_hist: List[float] = []
    max_hist: List[float] = []
    field_records: List[Dict[str, Any]] = []
    solution_sq_sum = 0.0
    solution_count = 0

    def rhs_fn(zv: np.ndarray, t: float) -> np.ndarray:
        a = N @ zv
        u = Phi @ a
        ux = dPhix @ a
        uy = dPhiy @ a
        lap = Phi @ (lap_diag * a)
        _, f = exact_and_force(
            x_in, y_in, t, nu=args.nu, amp=args.amp, omega=args.omega, device=device
        )
        rhs = args.nu * lap - u * ux - u * uy + f
        return weighted_lstsq_fit(PhiN, rhs, wq, args.fit_lam)

    for k, t in enumerate(times):
        a = N @ z
        pred = Phi @ a
        ref, _ = exact_and_force(
            x_in, y_in, float(t), nu=args.nu, amp=args.amp, omega=args.omega, device=device
        )
        rel = weighted_rel_l2(pred, ref, wq)
        bc = bc_violation(a, C_dense)
        change = weighted_rel_l2(pred, u_init, wq)
        max_u = max(float(np.max(np.abs(pred))), float(np.max(np.abs(ref))))
        solution_sq_sum += float(np.sum(ref * ref))
        solution_count += int(ref.size)
        if args.field_time_stride > 0 and (k % args.field_time_stride == 0 or k == nsteps):
            field_records.append({"k": int(k), "t": float(t), "ref": ref.copy(), "pred": pred.copy(), "rel": float(rel)})
        rel_hist.append(rel)
        bc_hist.append(bc)
        change_hist.append(change)
        max_hist.append(max_u)
        if k in snap_ids:
            snaps[k] = {"pred": pred.copy(), "ref": ref.copy(), "t": float(t)}
        if k < nsteps:
            z = rk4_step(z, float(t), args.dt, rhs_fn)
        if k % max(args.log_every, 1) == 0 or k == nsteps:
            print(
                f"[rollout] step={k:04d}/{nsteps:04d} t={t:.3f} "
                f"rel={rel:.3e} bc={bc:.3e} change={change:.3e} max|u|={max_u:.3e}"
            )

    rel_arr = np.array(rel_hist)
    bc_arr = np.array(bc_hist)
    change_arr = np.array(change_hist)
    max_arr = np.array(max_hist)

    plot_error_curve(
        times,
        rel_arr,
        bc_arr,
        os.path.join(args.outdir, "error_curves.png"),
        "MMS-V scalar Burgers",
    )
    plot_scalar_snapshots(
        grid,
        snaps,
        os.path.join(args.outdir, "snapshots_section3_vs_exact.png"),
        (xd, yd),
        "MMS-V scalar Burgers",
    )
    plot_change_from_initial(
        grid,
        snaps,
        os.path.join(args.outdir, "change_from_initial.png"),
        (xd, yd),
        "MMS-V scalar Burgers",
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
        "final_rel_l2": float(rel_arr[-1]),
        "mean_rel_l2": float(np.mean(rel_arr)),
        "max_rel_l2": float(np.max(rel_arr)),
        "final_bc": float(bc_arr[-1]),
        "final_change_from_initial": float(change_arr[-1]),
        "max_change_from_initial": float(np.max(change_arr)),
        "max_abs_u": float(np.max(max_arr)),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    _write_table_metrics_json(args.outdir, summary, rel_arr, bc_arr, solution_sq_sum, solution_count)
    _write_scalar_rollout_csv(
        os.path.join(args.outdir, "rollout_fields_and_relerr.csv"),
        x_in, y_in, field_records, times, rel_arr,
    )
    print(
        f"[summary] final_rel={rel_arr[-1]:.3e} max_rel={np.max(rel_arr):.3e} "
        f"final_bc={bc_arr[-1]:.3e} final_change={change_arr[-1]:.3e}"
    )


if __name__ == "__main__":
    raise SystemExit("This is an internal support module; run mms_05_pinwheel.py instead.")
