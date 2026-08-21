#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common numerical tools for the inner-square advective vector Burgers experiments.

This module is self-contained and does not require the exploratory Section-3/4
base driver.  It implements the same mathematical test used in the wrapper code:

    u_t = nu Delta u - u u_x - v u_y,
    v_t = nu Delta v - u v_x - v v_y,
    u=v=0 on the boundary of Omega=[-h,h]^2.

Two reduced block constructions are provided.

1. Ambient-S3 block.  A real tensor Fourier basis is built on Q=[-L,L]^2.
   The boundary matrix C is assembled from traces on the inner square, and the
   reduced space is the numerical nullspace N subset ker(C).  Rollout is then
   performed in coefficient coordinates a=N z, using preassembled matrices
   E=Phi_Omega N, Ex=Phi_x N, Ey=Phi_y N, and P=(N^T M_Omega N)^(-1)E^T W.

2. Direct-inner block.  A sine basis is built directly on Omega, so the zero
   Dirichlet boundary condition is built into the basis without using ambient
   transfer.

The module also provides a strong-form upwind finite-difference reference and
common diagnostics: relative L2 errors, boundary residuals, runtime, memory, and
kinetic-energy budget residuals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def weighted_l2(values: np.ndarray, w: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return math.sqrt(max(float(np.sum(w * v * v)), 0.0))


def weighted_rel_l2(pred: np.ndarray, ref: np.ndarray, w: np.ndarray) -> float:
    return weighted_l2(np.asarray(pred) - np.asarray(ref), w) / (weighted_l2(ref, w) + 1.0e-14)


def combined_rel_l2(u: np.ndarray, v: np.ndarray, ur: np.ndarray, vr: np.ndarray, w: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    ur = np.asarray(ur, dtype=np.float64).reshape(-1)
    vr = np.asarray(vr, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    num = math.sqrt(max(float(np.sum(w * ((u - ur) ** 2 + (v - vr) ** 2))), 0.0))
    den = math.sqrt(max(float(np.sum(w * (ur ** 2 + vr ** 2))), 0.0)) + 1.0e-14
    return float(num / den)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def save_curve(path: str, x: np.ndarray, curves: Sequence[Tuple[str, np.ndarray]], ylabel: str,
               title: str = "", semilogy: bool = False) -> None:
    if plt is None:
        return
    ensure_dir(os.path.dirname(path) or ".")
    plt.figure(figsize=(7.0, 4.5))
    for label, y in curves:
        if semilogy:
            plt.semilogy(x, y, label=label, linewidth=2)
        else:
            plt.plot(x, y, label=label, linewidth=2)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


# -----------------------------------------------------------------------------
# Geometry, bases, and quadrature
# -----------------------------------------------------------------------------

def inner_square_points(n: int, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centered tensor quadrature on Omega=[-h,h]^2."""
    n = int(n)
    dx = 2.0 * h / n
    xv = -h + (np.arange(n, dtype=np.float64) + 0.5) * dx
    yv = -h + (np.arange(n, dtype=np.float64) + 0.5) * dx
    X, Y = np.meshgrid(xv, yv, indexing="ij")
    x = X.reshape(-1)
    y = Y.reshape(-1)
    w = np.full_like(x, dx * dx, dtype=np.float64)
    return x, y, w, xv, yv


def boundary_points(nb: int, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Boundary sample points on the four sides of the inner square."""
    nb = max(4, int(nb))
    m = max(2, nb // 4)
    s = np.linspace(-h, h, m, endpoint=True)
    x = np.concatenate([s, s, -h * np.ones_like(s), h * np.ones_like(s)])
    y = np.concatenate([-h * np.ones_like(s), h * np.ones_like(s), s, s])
    return x.astype(np.float64), y.astype(np.float64)


def basis_1d(x: np.ndarray, K: int, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D real Fourier basis [1, cos1, sin1, cos2, sin2, ...] and derivative."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    K = int(K)
    B = np.empty((x.size, 2 * K + 1), dtype=np.float64)
    dB = np.empty_like(B)
    lam = np.empty(2 * K + 1, dtype=np.float64)
    B[:, 0] = 1.0
    dB[:, 0] = 0.0
    lam[0] = 0.0
    for k in range(1, K + 1):
        kk = math.pi * k / L
        cidx = 2 * k - 1
        sidx = 2 * k
        B[:, cidx] = np.cos(kk * x)
        B[:, sidx] = np.sin(kk * x)
        dB[:, cidx] = -kk * np.sin(kk * x)
        dB[:, sidx] = kk * np.cos(kk * x)
        lam[cidx] = kk * kk
        lam[sidx] = kk * kk
    return B, dB, lam


def tensor_basis_eval(x: np.ndarray, y: np.ndarray, K: int, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate tensor Fourier basis and gradients. Returns Phi, Phix, Phiy, lambda_sq."""
    Bx, dBx, lamx = basis_1d(x, K, L)
    By, dBy, lamy = basis_1d(y, K, L)
    n = np.asarray(x).size
    m1 = 2 * int(K) + 1
    M = m1 * m1
    Phi = np.empty((n, M), dtype=np.float64)
    Phix = np.empty_like(Phi)
    Phiy = np.empty_like(Phi)
    lam = np.empty(M, dtype=np.float64)
    col = 0
    for i in range(m1):
        for j in range(m1):
            Phi[:, col] = Bx[:, i] * By[:, j]
            Phix[:, col] = dBx[:, i] * By[:, j]
            Phiy[:, col] = Bx[:, i] * dBy[:, j]
            lam[col] = lamx[i] + lamy[j]
            col += 1
    return Phi, Phix, Phiy, lam


def sine_basis_eval(x: np.ndarray, y: np.ndarray, h: float, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Direct sine basis on Omega, ordered by increasing Laplace eigenvalue."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    rank = int(rank)
    nmax = int(math.ceil(math.sqrt(rank))) + 3
    modes: List[Tuple[float, int, int]] = []
    for m in range(1, nmax + 1):
        for n in range(1, nmax + 1):
            lam = ((math.pi * m / (2.0 * h)) ** 2 + (math.pi * n / (2.0 * h)) ** 2)
            modes.append((lam, m, n))
    modes.sort(key=lambda t: t[0])
    modes = modes[:rank]
    M = len(modes)
    Phi = np.empty((x.size, M), dtype=np.float64)
    Phix = np.empty_like(Phi)
    Phiy = np.empty_like(Phi)
    lam = np.empty(M, dtype=np.float64)
    sx = (x + h) / (2.0 * h)
    sy = (y + h) / (2.0 * h)
    for idx, (lv, m, n) in enumerate(modes):
        ax = math.pi * m / (2.0 * h)
        ay = math.pi * n / (2.0 * h)
        Phi[:, idx] = np.sin(math.pi * m * sx) * np.sin(math.pi * n * sy)
        Phix[:, idx] = ax * np.cos(math.pi * m * sx) * np.sin(math.pi * n * sy)
        Phiy[:, idx] = ay * np.sin(math.pi * m * sx) * np.cos(math.pi * n * sy)
        lam[idx] = lv
    return Phi, Phix, Phiy, lam, [(m, n) for _, m, n in modes]


def mass_matrix(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    return Phi.T @ (w[:, None] * Phi)


def build_s3_nullspace(Phi_b: np.ndarray, Phi_o: np.ndarray, w: np.ndarray, rank: int,
                       tau_rel: float = 1.0e-10) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Boundary-admissible S3 basis N subset ker(C), M_Omega-orthonormalized."""
    C = np.asarray(Phi_b, dtype=np.float64)
    U, s, Vt = np.linalg.svd(C, full_matrices=True)
    if s.size:
        tol = float(tau_rel) * max(float(s[0]), 1.0)
        rC = int(np.sum(s > tol))
    else:
        rC = 0
    Z = Vt[rC:, :].T.copy()
    null_dim = int(Z.shape[1])
    if rank <= 0 or rank > null_dim:
        rank_eff = null_dim
    else:
        rank_eff = int(rank)
    Z = Z[:, :rank_eff]
    MO = mass_matrix(Phi_o, w)
    G = 0.5 * (Z.T @ (MO @ Z) + (Z.T @ (MO @ Z)).T)
    evals, evecs = np.linalg.eigh(G)
    keep = evals > max(1.0e-14, 1.0e-12 * float(np.max(evals)))
    evals = evals[keep]
    evecs = evecs[:, keep]
    N = Z @ (evecs / np.sqrt(evals)[None, :])
    if rank_eff > 0:
        N = N[:, :rank_eff]
    info = {
        "rank_C": rC,
        "null_dim": null_dim,
        "requested_rank": int(rank),
        "final_rank": int(N.shape[1]),
        "mass_eigs": [float(np.min(evals)) if evals.size else 0.0, float(np.max(evals)) if evals.size else 0.0],
        "svd_tol": float(tol if s.size else 0.0),
    }
    return N, info


def rescaled_sine_ic(x: np.ndarray, y: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    u = np.sin(np.pi * (x + h) / (2.0 * h)) * np.sin(np.pi * (y + h) / (2.0 * h))
    v = np.sin(np.pi * (x + h) / h) * np.sin(np.pi * (y + h) / (2.0 * h))
    return u, v


# -----------------------------------------------------------------------------
# Reduced Galerkin model
# -----------------------------------------------------------------------------

@dataclass
class ReducedModel:
    name: str
    E: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    P: np.ndarray
    Lr: np.ndarray
    Mr: np.ndarray
    w: np.ndarray
    C_eval: Optional[np.ndarray] = None
    N: Optional[np.ndarray] = None
    lambda_sq: Optional[np.ndarray] = None
    build_time: float = 0.0
    memory_mib: float = 0.0

    def lift(self, z: np.ndarray) -> np.ndarray:
        return self.E @ np.asarray(z, dtype=np.float64)

    def lift_grad(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        return self.Ex @ z, self.Ey @ z

    def project_values(self, f: np.ndarray) -> np.ndarray:
        return self.P @ np.asarray(f, dtype=np.float64).reshape(-1)

    def nonlinear_rhs(self, zu: np.ndarray, zv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        zu = np.asarray(zu, dtype=np.float64)
        zv = np.asarray(zv, dtype=np.float64)
        u = self.E @ zu
        v = self.E @ zv
        ux = self.Ex @ zu
        uy = self.Ey @ zu
        vx = self.Ex @ zv
        vy = self.Ey @ zv
        fu = -self.P @ (u * ux + v * uy)
        fv = -self.P @ (u * vx + v * vy)
        return fu, fv

    def rhs_full(self, zu: np.ndarray, zv: np.ndarray, nu: float) -> Tuple[np.ndarray, np.ndarray]:
        nlu, nlv = self.nonlinear_rhs(zu, zv)
        return nlu + float(nu) * (self.Lr @ zu), nlv + float(nu) * (self.Lr @ zv)

    def boundary_residual(self, z: np.ndarray) -> float:
        if self.C_eval is None:
            return 0.0
        vals = self.C_eval @ np.asarray(z, dtype=np.float64)
        den = np.linalg.norm(z) + 1.0e-14
        return float(np.linalg.norm(vals) / den)

    def kinetic_budget_rhs(self, zu: np.ndarray, zv: np.ndarray, nu: float) -> float:
        u = self.E @ zu
        v = self.E @ zv
        ux = self.Ex @ zu
        uy = self.Ey @ zu
        vx = self.Ex @ zv
        vy = self.Ey @ zv
        div = ux + vy
        Econv = 0.5 * float(np.sum(self.w * (u * u + v * v) * div))
        Ediff = -float(nu) * float(np.sum(self.w * (ux * ux + uy * uy + vx * vx + vy * vy)))
        return Econv + Ediff

    def kinetic_energy(self, zu: np.ndarray, zv: np.ndarray) -> float:
        u = self.E @ zu
        v = self.E @ zv
        return 0.5 * float(np.sum(self.w * (u * u + v * v)))


@dataclass
class LegONetRolloutOperators:
    E: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    projector: np.ndarray
    linear_step: np.ndarray
    w: np.ndarray


def build_ambient_s3_model(K: int, L: float, h: float, n_quad: int, nb_build: int, nb_dense: int,
                           rank: int, tau_rel: float = 1.0e-10) -> Tuple[ReducedModel, Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    x, y, w, xv, yv = inner_square_points(n_quad, h)
    Phi, Phix, Phiy, lam = tensor_basis_eval(x, y, K, L)
    xb, yb = boundary_points(nb_build, h)
    Cb, _, _, _ = tensor_basis_eval(xb, yb, K, L)
    xd, yd = boundary_points(nb_dense, h)
    Cd_full, _, _, _ = tensor_basis_eval(xd, yd, K, L)
    N, info = build_s3_nullspace(Cb, Phi, w, rank, tau_rel=tau_rel)
    E = Phi @ N
    Ex = Phix @ N
    Ey = Phiy @ N
    Mr = N.T @ (mass_matrix(Phi, w) @ N)
    P = np.linalg.solve(Mr, E.T * w[None, :])
    Lred_full = -(lam[:, None] * N)
    Lr = np.linalg.solve(Mr, N.T @ (mass_matrix(Phi, w) @ Lred_full))
    C_eval = Cd_full @ N
    mem = (E.nbytes + Ex.nbytes + Ey.nbytes + P.nbytes + N.nbytes + Mr.nbytes + Lr.nbytes) / (1024.0 ** 2)
    model = ReducedModel("S3 ambient transferred block", E, Ex, Ey, P, Lr, Mr, w,
                         C_eval=C_eval, N=N, lambda_sq=lam, build_time=time.time() - t0, memory_mib=mem)
    grid = {"x": x, "y": y, "w": w, "xv": xv, "yv": yv, "h": h, "n_quad": n_quad}
    return model, info, grid


def build_direct_inner_model(h: float, n_quad: int, rank: int, nb_dense: int = 1200) -> Tuple[ReducedModel, Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    x, y, w, xv, yv = inner_square_points(n_quad, h)
    E, Ex, Ey, lam, modes = sine_basis_eval(x, y, h, rank)
    Mr = E.T @ (w[:, None] * E)
    P = np.linalg.solve(Mr, E.T * w[None, :])
    Lr = -np.diag(lam)
    xb, yb = boundary_points(nb_dense, h)
    C_eval, _, _, _, _ = sine_basis_eval(xb, yb, h, rank)
    mem = (E.nbytes + Ex.nbytes + Ey.nbytes + P.nbytes + Mr.nbytes + Lr.nbytes) / (1024.0 ** 2)
    model = ReducedModel("Direct inner-square sine block", E, Ex, Ey, P, Lr, Mr, w,
                         C_eval=C_eval, N=None, lambda_sq=lam, build_time=time.time() - t0, memory_mib=mem)
    info = {"basis": "direct_inner_sine", "requested_rank": int(rank), "final_rank": int(E.shape[1]), "modes": modes[:10]}
    grid = {"x": x, "y": y, "w": w, "xv": xv, "yv": yv, "h": h, "n_quad": n_quad}
    return model, info, grid


def project_initial(model: ReducedModel, h: float, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    u0, v0 = rescaled_sine_ic(x, y, h)
    zu = model.project_values(u0)
    zv = model.project_values(v0)
    u = model.lift(zu)
    v = model.lift(zv)
    info = {
        "init_rel_u": weighted_rel_l2(u, u0, model.w),
        "init_rel_v": weighted_rel_l2(v, v0, model.w),
        "u0_l2": weighted_l2(u, model.w),
        "v0_l2": weighted_l2(v, model.w),
    }
    return zu, zv, info


def rk4_nonlinear(model: ReducedModel, zu: np.ndarray, zv: np.ndarray, tau: float, nsub: int) -> Tuple[np.ndarray, np.ndarray]:
    if abs(tau) < 1.0e-15:
        return zu.copy(), zv.copy()
    hstep = float(tau) / max(1, int(nsub))
    u = np.asarray(zu, dtype=np.float64).copy()
    v = np.asarray(zv, dtype=np.float64).copy()
    for _ in range(max(1, int(nsub))):
        k1u, k1v = model.nonlinear_rhs(u, v)
        k2u, k2v = model.nonlinear_rhs(u + 0.5 * hstep * k1u, v + 0.5 * hstep * k1v)
        k3u, k3v = model.nonlinear_rhs(u + 0.5 * hstep * k2u, v + 0.5 * hstep * k2v)
        k4u, k4v = model.nonlinear_rhs(u + hstep * k3u, v + hstep * k3v)
        u += (hstep / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v += (hstep / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return u, v


def build_legonet_rollout_operators(model: ReducedModel, nu: float, dt: float) -> LegONetRolloutOperators:
    """Precompute time-independent matrices used during the online LegONet rollout.

    This matches the rollout principle in LEGONET_ROLLOUT_METHOD.md:
    - use one precomputed Galerkin projector for return-to-coefficient steps;
    - use one precomputed Crank--Nicolson linear step matrix;
    - keep the online loop to matrix multiplies plus physical-space nonlinear work.
    """
    I = np.eye(model.Lr.shape[0], dtype=np.float64)
    A = I - 0.5 * float(dt) * float(nu) * model.Lr
    B = I + 0.5 * float(dt) * float(nu) * model.Lr
    linear_step = np.linalg.solve(A, B)
    return LegONetRolloutOperators(
        E=np.asarray(model.E, dtype=np.float64),
        Ex=np.asarray(model.Ex, dtype=np.float64),
        Ey=np.asarray(model.Ey, dtype=np.float64),
        projector=np.asarray(model.P, dtype=np.float64),
        linear_step=np.asarray(linear_step, dtype=np.float64),
        w=np.asarray(model.w, dtype=np.float64),
    )


def legonet_nonlinear_rhs_pair(ops: LegONetRolloutOperators, state: np.ndarray,
                               stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    if stats is not None:
        t0 = time.perf_counter()
    uv = ops.E @ state
    gx = ops.Ex @ state
    gy = ops.Ey @ state
    if stats is not None:
        stats["lift_grad_sec"] = float(stats.get("lift_grad_sec", 0.0) + (time.perf_counter() - t0))
        t1 = time.perf_counter()
    rhs_in = np.empty_like(uv)
    rhs_in[:, 0] = -(uv[:, 0] * gx[:, 0] + uv[:, 1] * gy[:, 0])
    rhs_in[:, 1] = -(uv[:, 0] * gx[:, 1] + uv[:, 1] * gy[:, 1])
    if stats is not None:
        stats["pointwise_sec"] = float(stats.get("pointwise_sec", 0.0) + (time.perf_counter() - t1))
        t2 = time.perf_counter()
    out = ops.projector @ rhs_in
    if stats is not None:
        stats["project_sec"] = float(stats.get("project_sec", 0.0) + (time.perf_counter() - t2))
        stats["rhs_calls"] = float(stats.get("rhs_calls", 0.0) + 1.0)
    return out


def legonet_rk4_nonlinear_pair(ops: LegONetRolloutOperators, state: np.ndarray, tau: float, nsub: int,
                               stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    if abs(tau) < 1.0e-15:
        return np.asarray(state, dtype=np.float64).copy()
    hstep = float(tau) / max(1, int(nsub))
    out = np.asarray(state, dtype=np.float64).copy()
    for _ in range(max(1, int(nsub))):
        k1 = legonet_nonlinear_rhs_pair(ops, out, stats=stats)
        k2 = legonet_nonlinear_rhs_pair(ops, out + 0.5 * hstep * k1, stats=stats)
        k3 = legonet_nonlinear_rhs_pair(ops, out + 0.5 * hstep * k2, stats=stats)
        k4 = legonet_nonlinear_rhs_pair(ops, out + hstep * k3, stats=stats)
        out += (hstep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out


def rollout_legonet_final_only(model: ReducedModel, grid: Dict[str, Any], h: float, nu: float, dt: float, T: float,
                               nonlinear_substeps: int, profile_steps: bool = False) -> Dict[str, Any]:
    x = np.asarray(grid["x"], dtype=np.float64)
    y = np.asarray(grid["y"], dtype=np.float64)
    zu, zv, init_info = project_initial(model, h, x, y)
    Z = np.column_stack((zu, zv))
    ops = build_legonet_rollout_operators(model, nu=nu, dt=dt)
    nsteps = int(round(T / dt))
    half_dt = 0.5 * float(dt)
    step_rows: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, nsteps + 1):
        if profile_steps:
            step_t0 = time.perf_counter()
            t_a = time.perf_counter()
            stats1: Dict[str, float] = {}
        else:
            stats1 = {}
        Z = legonet_rk4_nonlinear_pair(ops, Z, half_dt, nonlinear_substeps, stats=stats1 if profile_steps else None)
        if profile_steps:
            reaction1_sec = time.perf_counter() - t_a
            t_b = time.perf_counter()
        Z = ops.linear_step @ Z
        if profile_steps:
            diffusion_sec = time.perf_counter() - t_b
            t_c = time.perf_counter()
            stats2: Dict[str, float] = {}
        else:
            stats2 = {}
        Z = legonet_rk4_nonlinear_pair(ops, Z, half_dt, nonlinear_substeps, stats=stats2 if profile_steps else None)
        if profile_steps:
            reaction2_sec = time.perf_counter() - t_c
            step_total_sec = time.perf_counter() - step_t0
            step_rows.append(
                {
                    "step": float(step),
                    "time": float(step * dt),
                    "reaction1_sec": float(reaction1_sec),
                    "diffusion_sec": float(diffusion_sec),
                    "reaction2_sec": float(reaction2_sec),
                    "reaction1_rhs_calls": float(stats1.get("rhs_calls", 0.0)),
                    "reaction1_lift_grad_sec": float(stats1.get("lift_grad_sec", 0.0)),
                    "reaction1_pointwise_sec": float(stats1.get("pointwise_sec", 0.0)),
                    "reaction1_project_sec": float(stats1.get("project_sec", 0.0)),
                    "reaction2_rhs_calls": float(stats2.get("rhs_calls", 0.0)),
                    "reaction2_lift_grad_sec": float(stats2.get("lift_grad_sec", 0.0)),
                    "reaction2_pointwise_sec": float(stats2.get("pointwise_sec", 0.0)),
                    "reaction2_project_sec": float(stats2.get("project_sec", 0.0)),
                    "step_total_sec": float(step_total_sec),
                }
            )
    rollout_time = time.perf_counter() - t0
    zu = Z[:, 0]
    zv = Z[:, 1]
    result = {
        "zu": zu,
        "zv": zv,
        "u_final": model.lift(zu),
        "v_final": model.lift(zv),
        "rollout_time_sec": float(rollout_time),
        "init_info": init_info,
        "timing_method": "legonet_precomputed_projector_precomputed_cn_matrix",
    }
    if profile_steps:
        result["step_timing"] = step_rows
    return result


def strang_step(model: ReducedModel, zu: np.ndarray, zv: np.ndarray, dt: float, nu: float, nonlinear_substeps: int) -> Tuple[np.ndarray, np.ndarray]:
    zu, zv = rk4_nonlinear(model, zu, zv, 0.5 * dt, nonlinear_substeps)
    I = np.eye(model.Lr.shape[0], dtype=np.float64)
    A = I - 0.5 * dt * float(nu) * model.Lr
    B = I + 0.5 * dt * float(nu) * model.Lr
    zu = np.linalg.solve(A, B @ zu)
    zv = np.linalg.solve(A, B @ zv)
    zu, zv = rk4_nonlinear(model, zu, zv, 0.5 * dt, nonlinear_substeps)
    return zu, zv


# -----------------------------------------------------------------------------
# FV reference and interpolation
# -----------------------------------------------------------------------------

def fv_initial_from_values(h: float, ref_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    dx = 2.0 * h / ref_n
    xv = -h + (np.arange(ref_n) + 0.5) * dx
    yv = -h + (np.arange(ref_n) + 0.5) * dx
    X, Y = np.meshgrid(xv, yv, indexing="ij")
    u, v = rescaled_sine_ic(X.reshape(-1), Y.reshape(-1), h)
    return xv, yv, u.reshape(ref_n, ref_n), v.reshape(ref_n, ref_n), dx


def fv_rhs_advective(u: np.ndarray, v: np.ndarray, dx: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    ug = np.pad(np.asarray(u, dtype=np.float64), ((1, 1), (1, 1)), mode="constant")
    vg = np.pad(np.asarray(v, dtype=np.float64), ((1, 1), (1, 1)), mode="constant")
    uc = ug[1:-1, 1:-1]
    vc = vg[1:-1, 1:-1]
    ux_b = (ug[1:-1, 1:-1] - ug[:-2, 1:-1]) / dx
    ux_f = (ug[2:, 1:-1] - ug[1:-1, 1:-1]) / dx
    uy_b = (ug[1:-1, 1:-1] - ug[1:-1, :-2]) / dx
    uy_f = (ug[1:-1, 2:] - ug[1:-1, 1:-1]) / dx
    vx_b = (vg[1:-1, 1:-1] - vg[:-2, 1:-1]) / dx
    vx_f = (vg[2:, 1:-1] - vg[1:-1, 1:-1]) / dx
    vy_b = (vg[1:-1, 1:-1] - vg[1:-1, :-2]) / dx
    vy_f = (vg[1:-1, 2:] - vg[1:-1, 1:-1]) / dx
    ux = np.where(uc >= 0.0, ux_b, ux_f)
    uy = np.where(vc >= 0.0, uy_b, uy_f)
    vx = np.where(uc >= 0.0, vx_b, vx_f)
    vy = np.where(vc >= 0.0, vy_b, vy_f)
    lap_u = (ug[2:, 1:-1] + ug[:-2, 1:-1] + ug[1:-1, 2:] + ug[1:-1, :-2] - 4.0 * uc) / (dx * dx)
    lap_v = (vg[2:, 1:-1] + vg[:-2, 1:-1] + vg[1:-1, 2:] + vg[1:-1, :-2] - 4.0 * vc) / (dx * dx)
    return -(uc * ux + vc * uy) + float(nu) * lap_u, -(uc * vx + vc * vy) + float(nu) * lap_v


def fv_advance(u: np.ndarray, v: np.ndarray, dt: float, dx: float, nu: float, cfl: float, diff_cfl: float) -> Tuple[np.ndarray, np.ndarray, int]:
    max_speed = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1.0e-12)
    dt_conv = float(cfl) * dx / max_speed
    dt_diff = float(diff_cfl) * dx * dx / max(float(nu), 1.0e-14)
    dt_sub = min(dt_conv, dt_diff, dt)
    nsub = max(1, int(math.ceil(dt / dt_sub)))
    hstep = dt / nsub
    uu = np.asarray(u, dtype=np.float64).copy()
    vv = np.asarray(v, dtype=np.float64).copy()
    for _ in range(nsub):
        ku, kv = fv_rhs_advective(uu, vv, dx, nu)
        uu = uu + hstep * ku
        vv = vv + hstep * kv
    return uu, vv, nsub


def bilinear_interpolate(grid: np.ndarray, xv: np.ndarray, yv: np.ndarray, x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = xv[1] - xv[0]
    # cell-centered coordinates, clamp to valid range
    xi = (x - xv[0]) / dx
    yi = (y - yv[0]) / dx
    i0 = np.floor(xi).astype(int)
    j0 = np.floor(yi).astype(int)
    tx = xi - i0
    ty = yi - j0
    n = len(xv)
    i0 = np.clip(i0, 0, n - 2)
    j0 = np.clip(j0, 0, n - 2)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)
    out = ((1 - tx) * (1 - ty) * grid[i0, j0]
           + tx * (1 - ty) * grid[i0 + 1, j0]
           + (1 - tx) * ty * grid[i0, j0 + 1]
           + tx * ty * grid[i0 + 1, j0 + 1])
    return out


def fv_budget_rhs(u: np.ndarray, v: np.ndarray, dx: float, nu: float) -> float:
    ug = np.pad(np.asarray(u, dtype=np.float64), ((1, 1), (1, 1)), mode="constant")
    vg = np.pad(np.asarray(v, dtype=np.float64), ((1, 1), (1, 1)), mode="constant")
    uc = ug[1:-1, 1:-1]
    vc = vg[1:-1, 1:-1]
    ux = (ug[2:, 1:-1] - ug[:-2, 1:-1]) / (2.0 * dx)
    uy = (ug[1:-1, 2:] - ug[1:-1, :-2]) / (2.0 * dx)
    vx = (vg[2:, 1:-1] - vg[:-2, 1:-1]) / (2.0 * dx)
    vy = (vg[1:-1, 2:] - vg[1:-1, :-2]) / (2.0 * dx)
    div = ux + vy
    area_w = dx * dx
    return 0.5 * float(np.sum(area_w * (uc * uc + vc * vc) * div)) - float(nu) * float(np.sum(area_w * (ux * ux + uy * uy + vx * vx + vy * vy)))


def fv_energy(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    return 0.5 * float(np.sum(dx * dx * (u * u + v * v)))


def boundary_residual_grid(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    ub = np.concatenate([u[0, :], u[-1, :], u[:, 0], u[:, -1]])
    vb = np.concatenate([v[0, :], v[-1, :], v[:, 0], v[:, -1]])
    return float(np.linalg.norm(ub) / (np.linalg.norm(u) + 1.0e-14)), float(np.linalg.norm(vb) / (np.linalg.norm(v) + 1.0e-14))


# -----------------------------------------------------------------------------
# Rollout and diagnostics
# -----------------------------------------------------------------------------

def run_reduced_vs_fv(model: ReducedModel, grid: Dict[str, Any], h: float, nu: float, dt: float, T: float,
                      nonlinear_substeps: int, ref_n: int, ref_cfl: float, ref_diff_cfl: float,
                      log_every: int = 100, verbose: bool = True) -> Dict[str, Any]:
    x = grid["x"]
    y = grid["y"]
    w = grid["w"]
    zu, zv, init_info = project_initial(model, h, x, y)
    xv_ref, yv_ref, u_ref, v_ref, dx_ref = fv_initial_from_values(h, ref_n)
    nsteps = int(round(T / dt))
    t_hist: List[float] = [0.0]
    err_u: List[float] = []
    err_v: List[float] = []
    err_c: List[float] = []
    bc_u: List[float] = [model.boundary_residual(zu)]
    bc_v: List[float] = [model.boundary_residual(zv)]
    energy: List[float] = [model.kinetic_energy(zu, zv)]
    budget_rhs: List[float] = [model.kinetic_budget_rhs(zu, zv, nu)]
    budget_res: List[float] = [0.0]
    ref_energy: List[float] = [fv_energy(u_ref, v_ref, dx_ref)]
    ref_budget_rhs: List[float] = [fv_budget_rhs(u_ref, v_ref, dx_ref, nu)]
    ref_budget_res: List[float] = [0.0]
    ref_substeps: List[int] = [0]
    t_start = time.time()
    u_q = model.lift(zu)
    v_q = model.lift(zv)
    ur_q = bilinear_interpolate(u_ref, xv_ref, yv_ref, x, y, h)
    vr_q = bilinear_interpolate(v_ref, xv_ref, yv_ref, x, y, h)
    err_u.append(weighted_rel_l2(u_q, ur_q, w))
    err_v.append(weighted_rel_l2(v_q, vr_q, w))
    err_c.append(combined_rel_l2(u_q, v_q, ur_q, vr_q, w))
    for step in range(1, nsteps + 1):
        Eold = energy[-1]
        Bold = budget_rhs[-1]
        Rold = ref_energy[-1]
        RBold = ref_budget_rhs[-1]
        zu, zv = strang_step(model, zu, zv, dt, nu, nonlinear_substeps)
        u_ref, v_ref, ns = fv_advance(u_ref, v_ref, dt, dx_ref, nu, ref_cfl, ref_diff_cfl)
        u_q = model.lift(zu)
        v_q = model.lift(zv)
        ur_q = bilinear_interpolate(u_ref, xv_ref, yv_ref, x, y, h)
        vr_q = bilinear_interpolate(v_ref, xv_ref, yv_ref, x, y, h)
        En = model.kinetic_energy(zu, zv)
        Bn = model.kinetic_budget_rhs(zu, zv, nu)
        Rn = fv_energy(u_ref, v_ref, dx_ref)
        RBn = fv_budget_rhs(u_ref, v_ref, dx_ref, nu)
        t_hist.append(step * dt)
        err_u.append(weighted_rel_l2(u_q, ur_q, w))
        err_v.append(weighted_rel_l2(v_q, vr_q, w))
        err_c.append(combined_rel_l2(u_q, v_q, ur_q, vr_q, w))
        bc_u.append(model.boundary_residual(zu))
        bc_v.append(model.boundary_residual(zv))
        energy.append(En)
        budget_rhs.append(Bn)
        budget_res.append(abs((En - Eold) / dt - 0.5 * (Bold + Bn)))
        ref_energy.append(Rn)
        ref_budget_rhs.append(RBn)
        ref_budget_res.append(abs((Rn - Rold) / dt - 0.5 * (RBold + RBn)))
        ref_substeps.append(ns)
        if verbose and ((step % log_every == 0) or step == nsteps):
            print(f"[{model.name}] step={step:05d}/{nsteps:05d} t={step*dt:.3f} rel={err_c[-1]:.3e} bc=({bc_u[-1]:.2e},{bc_v[-1]:.2e})", flush=True)
    rollout_time = time.time() - t_start
    return {
        "method": model.name,
        "rank": int(model.E.shape[1]),
        "build_time": float(model.build_time),
        "rollout_time": float(rollout_time),
        "memory_mib": float(model.memory_mib),
        "init_info": init_info,
        "history": {
            "time": t_hist,
            "rel_u": err_u,
            "rel_v": err_v,
            "rel_combined": err_c,
            "bc_u": bc_u,
            "bc_v": bc_v,
            "energy": energy,
            "budget_rhs": budget_rhs,
            "budget_residual": budget_res,
            "reference_energy": ref_energy,
            "reference_budget_rhs": ref_budget_rhs,
            "reference_budget_residual": ref_budget_res,
            "ref_substeps": ref_substeps,
        },
        "final": {
            "rel_u": float(err_u[-1]),
            "rel_v": float(err_v[-1]),
            "rel_combined": float(err_c[-1]),
            "bc_u": float(bc_u[-1]),
            "bc_v": float(bc_v[-1]),
            "budget_residual": float(budget_res[-1]),
            "energy": float(energy[-1]),
        },
        "summary": {
            "max_rel_combined": float(np.max(err_c)),
            "mean_rel_combined": float(np.mean(err_c)),
            "max_budget_residual": float(np.max(budget_res)),
            "mean_budget_residual": float(np.mean(budget_res)),
            "max_bc_u": float(np.max(bc_u)),
            "max_bc_v": float(np.max(bc_v)),
        },
    }


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--box_halfwidth", type=float, required=True)
    parser.add_argument("--inner_halfwidth", type=float, required=True)
    parser.add_argument("--n_quad", type=int, required=True)
    parser.add_argument("--Nb_build", type=int, required=True)
    parser.add_argument("--Nb_dense", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--nonlinear_substeps", type=int, required=True)
    parser.add_argument("--ref_N", type=int, required=True)
    parser.add_argument("--ref_cfl", type=float, default=0.22)
    parser.add_argument("--ref_diff_cfl", type=float, default=0.12)
    parser.add_argument("--tau_rel", type=float, default=1.0e-10)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def save_standard_outputs(outdir: str, results: Dict[str, Any]) -> None:
    ensure_dir(outdir)
    save_json(os.path.join(outdir, "summary.json"), results)
    rows = []
    for name, res in results.get("methods", {}).items():
        row = {"method": name, "rank": res.get("rank"), "build_time": res.get("build_time"),
               "rollout_time": res.get("rollout_time"), "memory_mib": res.get("memory_mib")}
        row.update({f"final_{k}": v for k, v in res.get("final", {}).items()})
        row.update({f"summary_{k}": v for k, v in res.get("summary", {}).items()})
        rows.append(row)
    write_csv(os.path.join(outdir, "metrics.csv"), rows)
    if results.get("methods"):
        first = next(iter(results["methods"].values()))
        t = np.asarray(first["history"]["time"])
        save_curve(os.path.join(outdir, "relative_error.png"), t,
                   [(k, np.asarray(v["history"]["rel_combined"])) for k, v in results["methods"].items()],
                   "relative L2", "Combined relative error", semilogy=True)
        save_curve(os.path.join(outdir, "boundary_error.png"), t,
                   [(k + " bc_u", np.asarray(v["history"]["bc_u"])) for k, v in results["methods"].items()] +
                   [(k + " bc_v", np.asarray(v["history"]["bc_v"])) for k, v in results["methods"].items()],
                   "relative boundary residual", "Boundary residual", semilogy=True)
        save_curve(os.path.join(outdir, "kinetic_budget_residual.png"), t,
                   [(k, np.asarray(v["history"]["budget_residual"])) for k, v in results["methods"].items()],
                   "budget residual", "Kinetic-energy budget residual", semilogy=True)
