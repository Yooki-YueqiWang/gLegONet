#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style boundary-aware block identification:

1. reduced-space adequacy / projection diagnostics,
2. short-time dense observation identification,
3. long-time held-out rollout validation,
4. spatial observation studies,
5. boundary-adaptation ablations.

This script intentionally moves away from temporal sparsity as the main study.
The default regime is temporally well-resolved, short-time dense observations
with q_stride = 1, while spatial observation patterns and admissible-coordinate
choice are treated as the main experimental axes.

For `reference_mode=projected_strong`, the default backend is a high-consistency
dense-render-and-reprojection reference, rather than the older masked finite-
difference backend.  The legacy backend is still available for diagnostics, but
it is typically projection-limited on irregular boundaries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.interpolate as spi
import scipy.linalg as sla
import scipy.signal as spsig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import core

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

CODE_VERSION = "BOUNDARY_BLOCK_IDENTIFICATION_SHORTTIME_LONGROLLOUT_V3"
FEATURES = core.FEATURES


@dataclass
class ReferenceTrajectory:
    ts: np.ndarray
    U: np.ndarray
    U_grid: np.ndarray
    Z_ref: np.ndarray
    stable: bool
    source_mode: str


@dataclass
class ReferenceBundle:
    domain_ref: core.DomainReduced
    coeffs_true: Dict[str, float]
    train_trajs: List[ReferenceTrajectory]
    heldout_traj: ReferenceTrajectory
    obs_indices: np.ndarray
    id_indices: np.ndarray


@dataclass
class ProjectedReferenceCache:
    render_grid: Dict[str, Any]
    render_matrix: np.ndarray


def parse_string_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ensure_integer_ratio(a: float, b: float, label: str) -> int:
    q = float(a) / float(b)
    qi = int(round(q))
    if abs(q - qi) > 1.0e-12:
        raise ValueError(f"{label} must be an integer ratio, got {a} / {b} = {q}")
    return max(1, qi)


def mean_std_rows(rows: List[Dict[str, Any]], group_keys: Sequence[str], metric_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_repeats"] = len(vals)
        for metric in metric_keys:
            arr = np.asarray([float(v[metric]) for v in vals if metric in v and np.isfinite(float(v[metric]))], dtype=np.float64)
            if arr.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_median"] = float("nan")
            else:
                row[f"{metric}_mean"] = float(np.mean(arr))
                row[f"{metric}_std"] = float(np.std(arr))
                row[f"{metric}_median"] = float(np.median(arr))
        out.append(row)
    return out


def default_coefficients(shape: str) -> Dict[str, float]:
    if shape == "peanut":
        return {"lap": 0.040, "tx": 0.0, "ty": 0.0, "u": 0.060, "u2": 0.0, "u3": -0.350}
    if shape == "channel":
        return {"lap": -0.020, "tx": -0.060, "ty": 0.090, "u": -0.030, "u2": 0.025, "u3": -0.300}
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
            raise ValueError(f"bad coefficient item: {item}")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in FEATURES:
            raise ValueError(f"unknown coefficient {key}")
        coeffs[key] = float(val)
    return coeffs


def basis_label(mode: str) -> str:
    return {
        "boundary_adapted": "boundary-adapted",
        "ambient_rank_matched": "ambient rank-matched",
        "full_ambient": "full ambient",
        "ambient": "ambient rank-matched",
    }.get(mode, mode)


def fd_baseline_note(args: argparse.Namespace) -> str:
    if "projected_fd" not in args.baseline_modes and "pointwise_fd" not in args.baseline_modes:
        return ""
    if args.reference_mode == "projected_strong" and args.projected_strong_backend == "legacy_masked_fd":
        return (
            "FD baselines are being evaluated against a matched strong-form masked-grid reference. "
            "Use this regime for FD cost studies on irregular domains."
        )
    desc = args.reference_mode
    if args.reference_mode == "projected_strong":
        desc += f"/{args.projected_strong_backend}"
    return (
        "FD baselines are being evaluated against reduced-generated reference trajectories "
        f"({desc}). This is convenient for reduced-model studies, but it is not a matched "
        "strong-form FD reference. For paper-level FD cost studies, prefer "
        "`fd_baseline_cost_study.py` with `projected_strong_backend=legacy_masked_fd`."
    )


def observation_setting_label(row: Dict[str, Any]) -> str:
    if int(row.get("grid_stride", 0)) > 0:
        return f"grid_stride={int(row['grid_stride'])}"
    if int(row.get("n_obs", 0)) > 0:
        return f"N_obs={int(row['n_obs'])}"
    return "dense"


def build_basis_domain(meta: Dict[str, Any], args: argparse.Namespace, basis_mode: str, K: int, rank: int) -> core.DomainReduced:
    if basis_mode == "boundary_adapted":
        args_local = argparse.Namespace(**vars(args))
        args_local.K = int(K)
        args_local.rank = int(rank)
        return core.build_domain(f"{args.shape}_{basis_mode}", args.shape, meta, args_local)

    grid = core.build_grid_and_boundary(args.shape, args)
    Phi = core.eval_basis(grid["x_in"], grid["y_in"], meta)
    Phi_x, Phi_y = core.eval_basis_grad(grid["x_in"], grid["y_in"], meta)
    M = core.assemble_mass(Phi, grid["w"])

    lam = np.asarray(meta["lambda"], dtype=np.float64)
    order = np.argsort(lam)
    ambient_dim = len(order)
    if basis_mode in ("ambient_rank_matched", "ambient"):
        rank_target = int(rank)
        if rank_target <= 0:
            rank_target = int(getattr(args, "rank_effective_boundary", 0))
        r = core.resolve_rank_request(rank_target, ambient_dim)
        ids = order[:r]
    elif basis_mode == "full_ambient":
        cap = int(args.full_ambient_rank_cap)
        if cap > 0:
            ids = order[:min(cap, ambient_dim)]
        else:
            ids = order
        r = len(ids)
    else:
        raise ValueError(f"unknown basis_mode {basis_mode}")

    N = np.eye(ambient_dim, dtype=np.float64)[:, ids]
    Phi_r = Phi[:, ids]
    Phi_rx = Phi_x[:, ids]
    Phi_ry = Phi_y[:, ids]
    M_r = core.assemble_mass(Phi_r, grid["w"])
    M_r_inv = np.linalg.inv(M_r + 1.0e-12 * np.eye(r))
    bd = core.eval_basis(grid["xb"], grid["yb"], meta) @ N
    bres = float(np.max(np.abs(bd)))
    me = np.linalg.eigvalsh(M_r)
    return core.DomainReduced(
        name=f"{args.shape}_{basis_mode}",
        shape=args.shape,
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
        null_info={"mode": basis_mode, "rank": r},
        parent_info={"ambient_selected_rank": r, "mass_eigs_minmax": [float(me.min()), float(me.max())]},
    )


def evaluate_boundary_trace(domain: core.DomainReduced, Z: np.ndarray) -> np.ndarray:
    B = core.eval_basis(domain.grid["xb"], domain.grid["yb"], domain.meta) @ domain.N
    return B @ np.asarray(Z, dtype=np.float64).T


def grid_rhs(U: np.ndarray, coeffs: Dict[str, float], dx: float, mask: np.ndarray) -> np.ndarray:
    U0 = np.where(mask, U, 0.0)
    fv = core.fd_features_one_snapshot(U0, dx)
    out = np.zeros_like(U0)
    for name in FEATURES:
        out = out + float(coeffs.get(name, 0.0)) * fv[name]
    return np.where(mask, out, 0.0)


def grid_rk4_step(U: np.ndarray, coeffs: Dict[str, float], dx: float, mask: np.ndarray, dt: float) -> np.ndarray:
    k1 = grid_rhs(U, coeffs, dx, mask)
    k2 = grid_rhs(U + 0.5 * dt * k1, coeffs, dx, mask)
    k3 = grid_rhs(U + 0.5 * dt * k2, coeffs, dx, mask)
    k4 = grid_rhs(U + dt * k3, coeffs, dx, mask)
    Un = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.where(mask, Un, 0.0)


def rollout_substeps_for_dt(args: argparse.Namespace, dt_sample: float, base_substeps: int) -> int:
    dt_sample = float(dt_sample)
    base_substeps = max(1, int(base_substeps))
    if dt_sample <= 0.0 or float(args.dt_obs) <= 0.0:
        return base_substeps
    return max(base_substeps, int(math.ceil(dt_sample / float(args.dt_obs))))


def grid_rk4_step_sampled(U: np.ndarray, coeffs: Dict[str, float], dx: float, mask: np.ndarray, dt_outer: float, substeps: int) -> np.ndarray:
    substeps = max(1, int(substeps))
    dt_inner = float(dt_outer) / float(substeps)
    Un = np.asarray(U, dtype=np.float64).copy()
    for _ in range(substeps):
        Un = grid_rk4_step(Un, coeffs, dx, mask, dt_inner)
    return Un


def assemble_reduced_laplace_matrix(domain: core.DomainReduced) -> np.ndarray:
    cached = domain.parent_info.get("_reduced_laplace_matrix")
    if isinstance(cached, np.ndarray):
        return cached
    r = domain.N.shape[1]
    eye = np.eye(r, dtype=np.float64)
    L = np.empty((r, r), dtype=np.float64)
    for j in range(r):
        L[:, j] = core.feature_vectors(domain, eye[:, j])["lap"]
    domain.parent_info["_reduced_laplace_matrix"] = L
    return L


def nonlinear_reduced_weak_part(domain: core.DomainReduced, z: np.ndarray, coeffs: Dict[str, float]) -> np.ndarray:
    fv = core.feature_vectors(domain, z)
    out = np.zeros(domain.N.shape[1], dtype=np.float64)
    for name in ("tx", "ty", "u2", "u3"):
        out = out + float(coeffs.get(name, 0.0)) * fv[name]
    return out


def build_projected_reference_cache(domain_ref: core.DomainReduced, args: argparse.Namespace) -> ProjectedReferenceCache:
    args_render = argparse.Namespace(**vars(args))
    args_render.Nx = max(int(args.Nx), int(args.reference_render_Nx))
    args_render.Nb = max(int(args.Nb), int(args.reference_render_Nb))
    render_grid = core.build_grid_and_boundary(args.shape, args_render)
    Phi_render = core.eval_basis(render_grid["x_in"], render_grid["y_in"], domain_ref.meta) @ domain_ref.N
    target_points = np.column_stack([domain_ref.grid["x_in"], domain_ref.grid["y_in"]])
    render_matrix = np.empty((domain_ref.Phi_r.shape[0], domain_ref.N.shape[1]), dtype=np.float64)
    for j in range(domain_ref.N.shape[1]):
        basis_grid = core.lift_to_grid(render_grid, Phi_render[:, j], fill=0.0)
        interp = spi.RegularGridInterpolator(
            (render_grid["xs"], render_grid["ys"]),
            basis_grid,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        render_matrix[:, j] = interp(target_points)
    return ProjectedReferenceCache(render_grid=render_grid, render_matrix=render_matrix)


def render_projected_reference_path(domain_ref: core.DomainReduced, Z: np.ndarray, cache: ProjectedReferenceCache) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U = np.asarray([cache.render_matrix @ z for z in Z], dtype=np.float64)
    U_grid = np.asarray([core.lift_to_grid(domain_ref.grid, u, fill=0.0) for u in U], dtype=np.float64)
    Z_proj = np.asarray([core.project_field_to_z(domain_ref, u) for u in U], dtype=np.float64)
    return U, U_grid, Z_proj


def rollout_projected_strong(
    domain_ref: core.DomainReduced,
    z0: np.ndarray,
    coeffs: Dict[str, float],
    args: argparse.Namespace,
    render_cache: ProjectedReferenceCache | None = None,
    T_horizon: float | None = None,
    dt_sample: float | None = None,
    substeps_override: int | None = None,
    integrator: str = "rk4",
) -> ReferenceTrajectory:
    T_horizon = float(args.T_roll if T_horizon is None else T_horizon)
    dt_sample = float(args.dt_solver if dt_sample is None else dt_sample)
    if args.projected_strong_backend == "dense_render_reproject":
        n_roll = int(round(T_horizon / dt_sample))
        dt = dt_sample
        ts = np.linspace(0.0, n_roll * dt, n_roll + 1)
        base_substeps = int(args.reference_solver_substeps) if int(args.reference_solver_substeps) > 0 else int(args.solver_substeps)
        if str(integrator).strip().lower() == "imex":
            substeps = int(substeps_override) if substeps_override is not None else 1
        else:
            substeps = int(substeps_override) if substeps_override is not None else rollout_substeps_for_dt(args, dt, base_substeps)
        Z, stable = rollout_reduced_sampled(domain_ref, z0, coeffs, dt, n_roll, substeps, float(args.rollout_max_amp), integrator=integrator)
        if Z.shape[0] < n_roll + 1:
            ts = ts[:Z.shape[0]]
        cache = render_cache if render_cache is not None else build_projected_reference_cache(domain_ref, args)
        U, U_grid, _ = render_projected_reference_path(domain_ref, Z, cache)
        return ReferenceTrajectory(ts=ts, U=U, U_grid=U_grid, Z_ref=Z, stable=stable, source_mode="projected_strong_dense_render_reproject")

    n_roll = int(round(T_horizon / dt_sample))
    dt = dt_sample
    ts = np.linspace(0.0, n_roll * dt, n_roll + 1)
    U0 = core.lift_to_grid(domain_ref.grid, domain_ref.Phi_r @ z0, fill=0.0)
    mask = domain_ref.grid["mask"]
    dx = float(domain_ref.grid["dx"])
    U_grid = np.empty((n_roll + 1,) + mask.shape, dtype=np.float64)
    U_grid[0] = U0
    stable = True
    substeps = int(substeps_override) if substeps_override is not None else rollout_substeps_for_dt(args, dt, int(args.reference_solver_substeps) if int(args.reference_solver_substeps) > 0 else int(args.solver_substeps))
    for n in range(n_roll):
        U_grid[n + 1] = grid_rk4_step_sampled(U_grid[n], coeffs, dx, mask, dt, substeps)
        umax = float(np.max(np.abs(U_grid[n + 1][mask])))
        if (not np.isfinite(umax)) or umax > float(args.rollout_max_amp):
            stable = False
            U_grid = U_grid[:n + 2]
            ts = ts[:n + 2]
            break
    U = U_grid[:, mask]
    Z = np.asarray([core.project_field_to_z(domain_ref, u) for u in U], dtype=np.float64)
    return ReferenceTrajectory(ts=ts, U=U, U_grid=U_grid, Z_ref=Z, stable=stable, source_mode="projected_strong_legacy_masked_fd")


def rollout_reduced_sampled(
    domain: core.DomainReduced,
    z0: np.ndarray,
    coeffs: Dict[str, float],
    dt_outer: float,
    n_outer: int,
    substeps: int,
    max_amp: float,
    integrator: str = "rk4",
) -> Tuple[np.ndarray, bool]:
    z = np.asarray(z0, dtype=np.float64).copy()
    substeps = max(1, int(substeps))
    dt_inner = float(dt_outer) / float(substeps)
    path = [z.copy()]
    stable = True
    integrator = str(integrator).strip().lower()
    if integrator not in {"rk4", "imex"}:
        raise ValueError(f"unknown reduced rollout integrator: {integrator}")
    if integrator == "imex":
        r = domain.N.shape[1]
        M = domain.M_r
        L_lap = assemble_reduced_laplace_matrix(domain)
        L_lin = float(coeffs.get("lap", 0.0)) * L_lap + float(coeffs.get("u", 0.0)) * M
        A = M - 0.5 * dt_inner * L_lin
        B = M + 0.5 * dt_inner * L_lin
        lu_A = sla.lu_factor(A + 1.0e-12 * np.eye(r), check_finite=False)
    for _ in range(int(n_outer)):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for _ in range(substeps):
                    if integrator == "rk4":
                        z = core.rk4_step(domain, z, coeffs, dt_inner)
                    else:
                        expl = nonlinear_reduced_weak_part(domain, z, coeffs)
                        rhs = B @ z + dt_inner * expl
                        z = sla.lu_solve(lu_A, rhs, check_finite=False)
        except Exception:
            stable = False
            break
        if not np.all(np.isfinite(z)):
            stable = False
            break
        if float(np.max(np.abs(domain.Phi_r @ z))) > float(max_amp):
            stable = False
            break
        path.append(z.copy())
    return np.asarray(path, dtype=np.float64), stable


def rollout_reduced_consistent(domain_ref: core.DomainReduced, z0: np.ndarray, coeffs: Dict[str, float], args: argparse.Namespace) -> ReferenceTrajectory:
    return rollout_reduced_consistent_sampled(domain_ref, z0, coeffs, args, T_horizon=None, dt_sample=None, integrator=str(args.rollout_integrator))


def rollout_reduced_consistent_sampled(
    domain_ref: core.DomainReduced,
    z0: np.ndarray,
    coeffs: Dict[str, float],
    args: argparse.Namespace,
    T_horizon: float | None = None,
    dt_sample: float | None = None,
    substeps_override: int | None = None,
    integrator: str = "rk4",
) -> ReferenceTrajectory:
    T_horizon = float(args.T_roll if T_horizon is None else T_horizon)
    dt = float(args.dt_solver if dt_sample is None else dt_sample)
    n_roll = int(round(T_horizon / dt))
    ts = np.linspace(0.0, n_roll * dt, n_roll + 1)
    if str(integrator).strip().lower() == "imex":
        substeps = int(substeps_override) if substeps_override is not None else 1
    else:
        substeps = int(substeps_override) if substeps_override is not None else rollout_substeps_for_dt(args, dt, int(args.solver_substeps))
    Z, stable = rollout_reduced_sampled(domain_ref, z0, coeffs, dt, n_roll, substeps, float(args.rollout_max_amp), integrator=integrator)
    if Z.shape[0] < n_roll + 1:
        ts = ts[:Z.shape[0]]
    U = np.asarray([domain_ref.Phi_r @ z for z in Z], dtype=np.float64)
    U_grid = np.asarray([core.lift_to_grid(domain_ref.grid, u, fill=0.0) for u in U], dtype=np.float64)
    return ReferenceTrajectory(ts=ts, U=U, U_grid=U_grid, Z_ref=Z, stable=stable, source_mode="reduced_consistent")


def generate_reference_bundle(domain_ref: core.DomainReduced, coeffs: Dict[str, float], args: argparse.Namespace, rep: int) -> ReferenceBundle:
    n_id_obs = int(round(float(args.T_id) / float(args.dt_obs))) + 1
    n_roll = int(round(float(args.T_roll) / float(args.dt_solver)))
    id_indices = np.arange(n_id_obs, dtype=np.int64)
    obs_indices = id_indices.copy()

    rng = np.random.default_rng(int(args.seed) + 100003 * rep)
    amps = np.asarray(core.parse_float_list(args.amp_list) if str(args.amp_list).strip() else [float(args.amp)], dtype=np.float64)
    if amps.size == 0:
        amps = np.asarray([float(args.amp)], dtype=np.float64)
    render_cache = build_projected_reference_cache(domain_ref, args) if args.reference_mode == "projected_strong" and args.projected_strong_backend == "dense_render_reproject" else None

    def make_candidate_z0(k: int) -> np.ndarray:
        round_id = k // max(int(amps.size), 1)
        amp_scale = 0.72 ** round_id
        amp_n = float(max(1.0e-3, amps[k % amps.size] * amp_scale))
        z0 = core.random_state(domain_ref, rng, amp_n, args.ic_low_dim, args.ic_decay)
        m = min(int(args.ic_low_dim), domain_ref.N.shape[1])
        if m > 0:
            perturb = np.zeros(domain_ref.N.shape[1], dtype=np.float64)
            perturb[:m] = 0.02 * rng.normal(size=m) / (1.0 + np.arange(m, dtype=np.float64)) ** float(args.ic_decay)
            z0 = core.scale_z_to_amp(domain_ref, z0 + perturb, amp_n)
        return z0

    def rollout_candidate(z0: np.ndarray) -> ReferenceTrajectory:
        if args.reference_mode == "reduced_consistent":
            return rollout_reduced_consistent_sampled(domain_ref, z0, coeffs, args, T_horizon=float(args.T_id), dt_sample=float(args.dt_obs), integrator=str(args.train_integrator))
        if args.reference_mode == "projected_strong":
            return rollout_projected_strong(domain_ref, z0, coeffs, args, render_cache, T_horizon=float(args.T_id), dt_sample=float(args.dt_obs), integrator=str(args.train_integrator))
        raise ValueError(args.reference_mode)

    def rollout_holdout(z0: np.ndarray) -> ReferenceTrajectory:
        if args.reference_mode == "reduced_consistent":
            return rollout_reduced_consistent_sampled(domain_ref, z0, coeffs, args, T_horizon=float(args.T_roll), dt_sample=float(args.dt_solver), integrator=str(args.rollout_integrator))
        if args.reference_mode == "projected_strong":
            return rollout_projected_strong(domain_ref, z0, coeffs, args, render_cache, T_horizon=float(args.T_roll), dt_sample=float(args.dt_solver), integrator=str(args.rollout_integrator))
        raise ValueError(args.reference_mode)

    train_trajs: List[ReferenceTrajectory] = []
    heldout: ReferenceTrajectory | None = None
    need_train = int(n_id_obs)
    need_holdout = int(n_roll + 1)
    max_attempts = max(120, 60 * (int(args.n_pairs) + 1))
    attempt = 0
    while (len(train_trajs) < int(args.n_pairs) or heldout is None) and attempt < max_attempts:
        z0 = make_candidate_z0(attempt)
        if len(train_trajs) < int(args.n_pairs):
            traj = rollout_candidate(z0)
            need = need_train
        else:
            traj = rollout_holdout(z0)
            need = need_holdout
        if len(traj.ts) >= need:
            if len(train_trajs) < int(args.n_pairs):
                train_trajs.append(traj)
            else:
                heldout = traj
        attempt += 1
    if len(train_trajs) < int(args.n_pairs) or heldout is None:
        raise RuntimeError(
            f"failed to generate enough stable trajectories for reference_mode={args.reference_mode}; "
            f"got train={len(train_trajs)}/{args.n_pairs}, heldout={heldout is not None}, attempts={attempt}"
        )
    return ReferenceBundle(domain_ref=domain_ref, coeffs_true=coeffs, train_trajs=train_trajs, heldout_traj=heldout, obs_indices=obs_indices, id_indices=id_indices)


def stack_sampled_fields(trajs: Sequence[ReferenceTrajectory], indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    U = np.asarray([traj.U[indices] for traj in trajs], dtype=np.float64)
    U_grid = np.asarray([traj.U_grid[indices] for traj in trajs], dtype=np.float64)
    return U, U_grid


def exact_projected_sequences(domain: core.DomainReduced, U_seq: np.ndarray) -> np.ndarray:
    n_traj, n_time, _ = U_seq.shape
    Z = np.empty((n_traj, n_time, domain.N.shape[1]), dtype=np.float64)
    for i in range(n_traj):
        for j in range(n_time):
            Z[i, j] = core.project_field_to_z(domain, U_seq[i, j])
    return Z


def build_simpson_regression_from_sequences(domain: core.DomainReduced, Z_seq: np.ndarray, dt_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    Z0 = Z_seq[:, :-2, :].reshape(-1, Z_seq.shape[-1])
    Z1 = Z_seq[:, 1:-1, :].reshape(-1, Z_seq.shape[-1])
    Z2 = Z_seq[:, 2:, :].reshape(-1, Z_seq.shape[-1])
    return core.build_simpson_regression_from_triples(domain, Z0, Z1, Z2, dt_obs)


def build_centered_regression_from_sequences(domain: core.DomainReduced, Z_seq: np.ndarray, dt_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    Z0 = Z_seq[:, :-2, :].reshape(-1, Z_seq.shape[-1])
    Z1 = Z_seq[:, 1:-1, :].reshape(-1, Z_seq.shape[-1])
    Z2 = Z_seq[:, 2:, :].reshape(-1, Z_seq.shape[-1])
    return core.build_centered_fd_regression_from_triples(domain, Z0, Z1, Z2, dt_obs)


def add_obs_noise(U_obs: np.ndarray, rng: np.random.Generator, noise_rel: float) -> np.ndarray:
    if float(noise_rel) <= 0.0:
        return U_obs
    noise = rng.normal(size=U_obs.shape)
    scale = np.linalg.norm(U_obs) / (np.linalg.norm(noise) + 1.0e-14)
    return U_obs + float(noise_rel) * scale * noise


def temporal_smooth_sequence(seq: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    kind = str(getattr(args, "temporal_smooth_kind", "none")).strip().lower()
    if kind in ("none", ""):
        return seq
    if kind != "savgol":
        raise ValueError(f"unknown temporal_smooth_kind={kind}")
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"temporal smoothing expects a 3D array, got shape={arr.shape}")
    n_time = int(arr.shape[1])
    win = int(getattr(args, "temporal_smooth_window", 0))
    poly = int(getattr(args, "temporal_smooth_polyorder", 0))
    if win <= 2 or n_time <= 2:
        return arr
    win = min(win, n_time if n_time % 2 == 1 else n_time - 1)
    if win <= poly:
        win = poly + 1
        if win % 2 == 0:
            win += 1
    if win > n_time:
        win = n_time if n_time % 2 == 1 else n_time - 1
    if win <= poly or win <= 2:
        return arr
    return np.asarray(spsig.savgol_filter(arr, window_length=win, polyorder=poly, axis=1, mode="interp"), dtype=np.float64)


def choose_sparse_sensor_indices(domain_ref: core.DomainReduced, n_obs: int, args: argparse.Namespace, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    r_obs = core.effective_observation_rank(int(n_obs), domain_ref.N.shape[1], args)
    idx, cond, smin = core.choose_sensor_indices(domain_ref, int(n_obs), r_obs, rng, args.sensor_retries)
    return idx, {
        "n_obs": int(len(idx)),
        "grid_stride": 0,
        "sensor_condition_best": float(cond),
        "sensor_smallest_sv": float(smin),
        "r_obs_ref": int(r_obs),
    }


def choose_grid_sensor_indices(domain_ref: core.DomainReduced, stride: int) -> Tuple[np.ndarray, Dict[str, float]]:
    stride = max(1, int(stride))
    mask = domain_ref.grid["mask"]
    interior_id = -np.ones(mask.shape, dtype=np.int64)
    interior_id[mask] = np.arange(int(mask.sum()), dtype=np.int64)
    chosen = np.zeros_like(mask, dtype=bool)
    chosen[::stride, ::stride] = True
    idx = interior_id[mask & chosen]
    idx = idx[idx >= 0]
    return np.asarray(idx, dtype=np.int64), {
        "n_obs": int(len(idx)),
        "grid_stride": int(stride),
        "sensor_condition_best": float("nan"),
        "sensor_smallest_sv": float("nan"),
        "r_obs_ref": int(core.effective_observation_rank(len(idx), domain_ref.N.shape[1], args=argparse.Namespace(obs_rank_factor=1.0, obs_rank_cap=0))),
    }


def choose_observation_settings(domain_ref: core.DomainReduced, args: argparse.Namespace, rep: int) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    rng = np.random.default_rng(int(args.seed) + 300011 * rep)
    settings: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    if args.obs_mode == "sparse":
        for n_obs in args.obs_list:
            idx, meta = choose_sparse_sensor_indices(domain_ref, int(n_obs), args, rng)
            meta.update({"obs_mode": "sparse"})
            settings.append((idx, meta))
    elif args.obs_mode == "grid":
        for stride in args.grid_obs_stride_list:
            idx, meta = choose_grid_sensor_indices(domain_ref, int(stride))
            meta.update({"obs_mode": "grid"})
            settings.append((idx, meta))
    elif args.obs_mode == "dense":
        idx = np.arange(domain_ref.Phi_r.shape[0], dtype=np.int64)
        settings.append((idx, {"obs_mode": "dense", "n_obs": int(len(idx)), "grid_stride": 1, "sensor_condition_best": float("nan"), "sensor_smallest_sv": float("nan"), "r_obs_ref": int(domain_ref.N.shape[1])}))
    else:
        raise ValueError(args.obs_mode)
    return settings


def recover_reduced_coordinates(domain: core.DomainReduced, U_obs_flat: np.ndarray, sensor_idx: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float]]:
    Phi_obs = domain.Phi_r[sensor_idx, :]
    r_obs = core.effective_observation_rank(len(sensor_idx), domain.N.shape[1], args)
    Z_hat = core.recover_z_from_sparse_observations(Phi_obs, U_obs_flat, r_obs, args.z_ridge, args.mode_reg_power)
    rec = core.sensor_reconstruction_metrics(Phi_obs, U_obs_flat, Z_hat)
    rec["r_obs"] = int(r_obs)
    return Z_hat, rec


def interpolate_grid_observations(domain: core.DomainReduced, sensor_idx: np.ndarray, U_obs_flat: np.ndarray, stride: int) -> np.ndarray:
    mask = domain.grid["mask"]
    xs = domain.grid["xs"]
    ys = domain.grid["ys"]
    X = domain.grid["X"]
    Y = domain.grid["Y"]
    stride = max(1, int(stride))
    if stride == 1 and len(sensor_idx) == int(mask.sum()):
        U_grid = np.empty((U_obs_flat.shape[0],) + mask.shape, dtype=np.float64)
        for i in range(U_obs_flat.shape[0]):
            U_grid[i] = core.lift_to_grid(domain.grid, U_obs_flat[i], fill=0.0)
        return U_grid

    coarse_x = xs[::stride]
    coarse_y = ys[::stride]
    interior_id = -np.ones(mask.shape, dtype=np.int64)
    interior_id[mask] = np.arange(int(mask.sum()), dtype=np.int64)
    rev = -np.ones(int(mask.sum()), dtype=np.int64)
    rev[np.asarray(sensor_idx, dtype=np.int64)] = np.arange(len(sensor_idx), dtype=np.int64)
    out = np.empty((U_obs_flat.shape[0],) + mask.shape, dtype=np.float64)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    for n in range(U_obs_flat.shape[0]):
        Uc = np.zeros((len(coarse_x), len(coarse_y)), dtype=np.float64)
        for ii, I in enumerate(range(0, mask.shape[0], stride)):
            for jj, J in enumerate(range(0, mask.shape[1], stride)):
                if mask[I, J]:
                    interior = interior_id[I, J]
                    pos = rev[interior]
                    if pos >= 0:
                        Uc[ii, jj] = U_obs_flat[n, pos]
        interp = spi.RegularGridInterpolator((coarse_x, coarse_y), Uc, method="linear", bounds_error=False, fill_value=0.0)
        out[n] = np.where(mask, interp(XY).reshape(mask.shape), 0.0)
    return out


def reconstruct_fd_grids(domain: core.DomainReduced, sensor_idx: np.ndarray, U_obs_flat: np.ndarray, obs_mode: str, grid_stride: int) -> np.ndarray:
    x_obs = domain.grid["x_in"][sensor_idx]
    y_obs = domain.grid["y_in"][sensor_idx]
    if obs_mode == "grid":
        return interpolate_grid_observations(domain, sensor_idx, U_obs_flat, int(grid_stride))
    if obs_mode == "dense":
        return interpolate_grid_observations(domain, sensor_idx, U_obs_flat, 1)
    return core.interpolate_sparse_to_grid(domain, x_obs, y_obs, U_obs_flat)


def build_projected_fd_regression_from_sequences(domain: core.DomainReduced, U_grid_seq: np.ndarray, dt_obs: float, test_rank: int) -> Tuple[np.ndarray, np.ndarray]:
    U0 = U_grid_seq[:, :-2, :, :].reshape(-1, *U_grid_seq.shape[-2:])
    U1 = U_grid_seq[:, 1:-1, :, :].reshape(-1, *U_grid_seq.shape[-2:])
    U2 = U_grid_seq[:, 2:, :, :].reshape(-1, *U_grid_seq.shape[-2:])
    return core.build_projected_fd_regression_from_triples(domain, U0, U1, U2, dt_obs, test_rank)


def build_pointwise_fd_regression_from_sequences(domain: core.DomainReduced, U_grid_seq: np.ndarray, dt_obs: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = domain.grid["mask"]
    valid = core.eroded_mask(mask)
    dx = float(domain.grid["dx"])
    cols: List[List[np.ndarray]] = [[] for _ in FEATURES]
    Ys: List[np.ndarray] = []
    for traj in U_grid_seq:
        for n in range(1, traj.shape[0] - 1):
            U0 = np.where(mask, traj[n - 1], 0.0)
            U1 = np.where(mask, traj[n], 0.0)
            U2 = np.where(mask, traj[n + 1], 0.0)
            Ut = (U2 - U0) / (2.0 * float(dt_obs))
            fv = core.fd_features_one_snapshot(U1, dx)
            Ys.append(Ut[valid])
            for j, name in enumerate(FEATURES):
                cols[j].append(fv[name][valid])
    X = np.column_stack([np.concatenate(c, axis=0) for c in cols])
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def solve_regression_metrics(X: np.ndarray, Y: np.ndarray, coeffs_true: Dict[str, float], args: argparse.Namespace) -> Tuple[Dict[str, Any], np.ndarray]:
    c_true = core.coeff_array(coeffs_true)
    c_hat, info = core.sequential_thresholded_ridge(
        X,
        Y,
        ridge=args.coeff_ridge,
        threshold=args.threshold,
        max_iterations=args.max_threshold_iterations,
    )
    met = core.coefficient_metrics(c_hat, c_true, args.active_tol)
    return {
        "true_coeff_residual": core.regression_true_residual(X, Y, c_true),
        "ls_residual": float(info["residual_rel_l2"]),
        "condition_number_col_normalized": float(info["condition_number_col_normalized"]),
        "mean_active_rel_error": float(met["mean_active_rel_error"]),
        "max_active_rel_error": float(met["max_active_rel_error"]),
        "inactive_l1": float(met["inactive_l1"]),
        "support_ok": bool(met["support_ok"]),
    }, c_hat


def rollout_with_coefficients(domain: core.DomainReduced, coeffs_hat: Dict[str, float], traj_ref: ReferenceTrajectory, args: argparse.Namespace) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    n_steps = min(len(traj_ref.ts) - 1, int(round(float(args.T_roll) / float(args.dt_solver))))
    z0 = core.project_field_to_z(domain, traj_ref.U[0])
    integrator = str(args.rollout_integrator).strip().lower()
    substeps = 1 if integrator == "imex" else rollout_substeps_for_dt(args, float(args.dt_solver), int(args.solver_substeps))
    Z_pred, stable = rollout_reduced_sampled(domain, z0, coeffs_hat, args.dt_solver, n_steps, substeps, float(args.rollout_max_amp), integrator=integrator)
    n = min(len(Z_pred), len(traj_ref.U))
    Z_pred = Z_pred[:n]
    errs: List[float] = []
    bnd_rms: List[float] = []
    for i in range(n):
        u_pred = domain.Phi_r @ Z_pred[i]
        errs.append(core.weighted_rel_l2(u_pred, traj_ref.U[i], domain.grid["w"]))
        bvals = evaluate_boundary_trace(domain, Z_pred[i]).ravel()
        bnd_rms.append(float(np.sqrt(np.mean(bvals * bvals))))
    return {
        "rollout_stable": bool(stable),
        "rollout_steps_compared": int(n - 1),
        "rollout_rel_l2_mean": float(np.mean(errs)) if errs else float("nan"),
        "rollout_rel_l2_max": float(np.max(errs)) if errs else float("nan"),
        "rollout_rel_l2_final": float(errs[-1]) if errs else float("nan"),
        "boundary_rms_mean": float(np.mean(bnd_rms)) if bnd_rms else float("nan"),
        "boundary_rms_max": float(np.max(bnd_rms)) if bnd_rms else float("nan"),
        "boundary_rms_final": float(bnd_rms[-1]) if bnd_rms else float("nan"),
    }, traj_ref.ts[:n], Z_pred


def projection_error_rows(domain: core.DomainReduced, trajs: Sequence[ReferenceTrajectory], tag: str) -> List[float]:
    errs: List[float] = []
    for traj in trajs:
        for u in traj.U:
            z = core.project_field_to_z(domain, u)
            errs.append(core.weighted_rel_l2(domain.Phi_r @ z, u, domain.grid["w"]))
    return errs


def run_projection_diagnostic(args: argparse.Namespace) -> List[Dict[str, Any]]:
    outdir = core.ensure_dir(args.outdir)
    coeffs_true = parse_coeffs(args.coeffs, args.shape)
    rows: List[Dict[str, Any]] = []
    for K in args.K_list:
        meta = core.make_real_trig_basis_metadata(int(K))
        for rank in args.rank_list:
            domain = build_basis_domain(meta, args, args.basis_mode, int(K), int(rank))
            ref_bundle = generate_reference_bundle(domain, coeffs_true, args, rep=0)
            proj_errs = projection_error_rows(domain, ref_bundle.train_trajs + [ref_bundle.heldout_traj], "projection")
            U_train, _ = stack_sampled_fields(ref_bundle.train_trajs, ref_bundle.id_indices)
            Z_proj = exact_projected_sequences(domain, U_train)
            X, Y = build_simpson_regression_from_sequences(domain, Z_proj, args.dt_obs)
            c_true = core.coeff_array(coeffs_true)
            row = {
                "reference_mode": args.reference_mode,
                "basis_mode": args.basis_mode,
                "K": int(K),
                "rank": int(rank),
                "boundary_residual": float(domain.boundary_residual),
                "mass_eig_min": float(np.linalg.eigvalsh(domain.M_r).min()),
                "mass_eig_max": float(np.linalg.eigvalsh(domain.M_r).max()),
                "projection_error_mean": float(np.mean(proj_errs)) if proj_errs else float("nan"),
                "projection_error_max": float(np.max(proj_errs)) if proj_errs else float("nan"),
                "true_coeff_residual": core.regression_true_residual(X, Y, c_true),
                "projection_limited": bool((np.mean(proj_errs) > float(args.projection_error_warn)) or (core.regression_true_residual(X, Y, c_true) > float(args.true_residual_warn))),
            }
            rows.append(row)
            print(f"[projection] K={K:3d}, rank={rank:4d}, Eproj={row['projection_error_mean']:.3e}, Rtrue={row['true_coeff_residual']:.3e}, bres={row['boundary_residual']:.3e}")

    core.write_csv(outdir / "projection_diagnostic_by_K_rank.csv", rows)
    plot_projection_vs_rank(outdir / "projection_error_vs_rank.png", rows, "projection_error_mean", "projection error")
    plot_projection_vs_rank(outdir / "true_residual_vs_rank.png", rows, "true_coeff_residual", "true-coefficient residual")
    return rows


def method_trial_row_base(args: argparse.Namespace, rep: int, obs_meta: Dict[str, Any], method: str, basis_mode: str) -> Dict[str, Any]:
    return {
        "experiment": "shorttime_longrollout",
        "method": method,
        "basis_mode": basis_mode,
        "reference_mode": args.reference_mode,
        "obs_mode": obs_meta["obs_mode"],
        "n_obs": int(obs_meta.get("n_obs", 0)),
        "grid_stride": int(obs_meta.get("grid_stride", 0)),
        "rep": int(rep),
        "dt_obs": float(args.dt_obs),
        "dt_solver": float(args.dt_solver),
        "T_id": float(args.T_id),
        "T_roll": float(args.T_roll),
    }


def evaluate_boundary_adapted_trial(ref_bundle: ReferenceBundle, args: argparse.Namespace, rep: int, sensor_idx: np.ndarray, obs_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    domain = ref_bundle.domain_ref
    coeffs_true = ref_bundle.coeffs_true
    rng = np.random.default_rng(int(args.seed) + 700001 * rep + int(obs_meta.get("n_obs", 0)) + 17 * int(obs_meta.get("grid_stride", 0)))

    U_train, U_train_grid_true = stack_sampled_fields(ref_bundle.train_trajs, ref_bundle.id_indices)
    Z_exact = np.asarray([traj.Z_ref[ref_bundle.id_indices] for traj in ref_bundle.train_trajs], dtype=np.float64)
    c_true = core.coeff_array(coeffs_true)
    coeff_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []

    exact_row = method_trial_row_base(args, rep, obs_meta, "exact_simpson_upper_bound", "boundary_adapted")
    X_exact, Y_exact = build_simpson_regression_from_sequences(domain, Z_exact, args.dt_obs)
    met_exact, c_exact = solve_regression_metrics(X_exact, Y_exact, coeffs_true, args)
    exact_row.update(met_exact)
    exact_row.update({
        "sensor_reconstruction_rel_l2_mean": 0.0,
        "sensor_reconstruction_rel_l2_max": 0.0,
        "boundary_residual_domain": float(domain.boundary_residual),
        "projection_error_mean": float(np.mean(projection_error_rows(domain, ref_bundle.train_trajs, "exact"))),
        "projection_error_max": float(np.max(projection_error_rows(domain, ref_bundle.train_trajs, "exact"))),
    })
    coeff_exact = {name: float(v) for name, v in zip(FEATURES, c_exact)}
    roll_exact, _, _ = rollout_with_coefficients(domain, coeff_exact, ref_bundle.heldout_traj, args)
    exact_row.update(roll_exact)
    trial_rows.append(exact_row)

    U_obs_flat = U_train[:, :, sensor_idx].reshape(-1, len(sensor_idx))
    U_obs_flat = add_obs_noise(U_obs_flat, rng, args.obs_noise_rel)
    smooth_target = str(getattr(args, "temporal_smooth_target", "none")).strip().lower()
    if smooth_target in ("obs", "both"):
        U_obs_flat = temporal_smooth_sequence(U_obs_flat.reshape(U_train.shape[0], U_train.shape[1], -1), args).reshape(-1, len(sensor_idx))
    Z_hat_flat, rec = recover_reduced_coordinates(domain, U_obs_flat, sensor_idx, args)
    Z_hat = Z_hat_flat.reshape(U_train.shape[0], U_train.shape[1], -1)
    if smooth_target in ("z", "both"):
        Z_hat = temporal_smooth_sequence(Z_hat, args)
    ours_row = method_trial_row_base(args, rep, obs_meta, "ours", "boundary_adapted")
    X_ours, Y_ours = build_simpson_regression_from_sequences(domain, Z_hat, args.dt_obs)
    met_ours, c_ours = solve_regression_metrics(X_ours, Y_ours, coeffs_true, args)
    ours_row.update(met_ours)
    ours_row.update(rec)
    ours_row.update({
        "boundary_residual_domain": float(domain.boundary_residual),
        "projection_error_mean": float(np.mean(projection_error_rows(domain, ref_bundle.train_trajs, "ours"))),
        "projection_error_max": float(np.max(projection_error_rows(domain, ref_bundle.train_trajs, "ours"))),
    })
    coeff_ours = {name: float(v) for name, v in zip(FEATURES, c_ours)}
    roll_ours, ts_roll, Z_roll_ours = rollout_with_coefficients(domain, coeff_ours, ref_bundle.heldout_traj, args)
    ours_row.update(roll_ours)
    trial_rows.append(ours_row)

    U_grid_hat = reconstruct_fd_grids(domain, sensor_idx, U_obs_flat, obs_meta["obs_mode"], int(obs_meta.get("grid_stride", 0)))
    U_grid_hat = U_grid_hat.reshape(U_train.shape[0], U_train.shape[1], *domain.grid["mask"].shape)
    fd_row = method_trial_row_base(args, rep, obs_meta, "projected_fd", "boundary_adapted")
    X_fd, Y_fd = build_projected_fd_regression_from_sequences(domain, U_grid_hat, args.dt_obs, domain.N.shape[1])
    met_fd, c_fd = solve_regression_metrics(X_fd, Y_fd, coeffs_true, args)
    fd_row.update(met_fd)
    fd_row.update({
        "sensor_reconstruction_rel_l2_mean": float(np.mean([core.weighted_rel_l2(core.lift_to_grid(domain.grid, uhat[domain.grid['mask']], fill=0.0)[domain.grid["mask"]], utrue, domain.grid["w"]) for uhat, utrue in zip(U_grid_hat.reshape(-1, *domain.grid["mask"].shape), U_train.reshape(-1, U_train.shape[-1]))])) if U_train.size else float("nan"),
        "sensor_reconstruction_rel_l2_max": float(np.max([core.weighted_rel_l2(uhat[domain.grid["mask"]], utrue, domain.grid["w"]) for uhat, utrue in zip(U_grid_hat.reshape(-1, *domain.grid["mask"].shape), U_train.reshape(-1, U_train.shape[-1]))])) if U_train.size else float("nan"),
        "boundary_residual_domain": float(domain.boundary_residual),
        "projection_error_mean": float(np.mean(projection_error_rows(domain, ref_bundle.train_trajs, "fd"))),
        "projection_error_max": float(np.max(projection_error_rows(domain, ref_bundle.train_trajs, "fd"))),
    })
    coeff_fd = {name: float(v) for name, v in zip(FEATURES, c_fd)}
    roll_fd, _, _ = rollout_with_coefficients(domain, coeff_fd, ref_bundle.heldout_traj, args)
    fd_row.update(roll_fd)
    trial_rows.append(fd_row)

    if "pointwise_fd" in args.baseline_modes:
        p_row = method_trial_row_base(args, rep, obs_meta, "pointwise_fd", "boundary_adapted")
        X_p, Y_p = build_pointwise_fd_regression_from_sequences(domain, U_grid_hat, args.dt_obs)
        met_p, c_p = solve_regression_metrics(X_p, Y_p, coeffs_true, args)
        p_row.update(met_p)
        p_row.update({
            "sensor_reconstruction_rel_l2_mean": fd_row["sensor_reconstruction_rel_l2_mean"],
            "sensor_reconstruction_rel_l2_max": fd_row["sensor_reconstruction_rel_l2_max"],
            "boundary_residual_domain": float(domain.boundary_residual),
            "projection_error_mean": float(np.mean(projection_error_rows(domain, ref_bundle.train_trajs, "pointwise_fd"))),
            "projection_error_max": float(np.max(projection_error_rows(domain, ref_bundle.train_trajs, "pointwise_fd"))),
        })
        coeff_p = {name: float(v) for name, v in zip(FEATURES, c_p)}
        roll_p, _, _ = rollout_with_coefficients(domain, coeff_p, ref_bundle.heldout_traj, args)
        p_row.update(roll_p)
        trial_rows.append(p_row)
    else:
        c_p = None

    for method_name, coeff_vec in [("exact_simpson_upper_bound", c_exact), ("ours", c_ours), ("projected_fd", c_fd)]:
        for name, tv, hv in zip(FEATURES, c_true, coeff_vec):
            coeff_rows.append({
                "experiment": "shorttime_longrollout",
                "method": method_name,
                "basis_mode": "boundary_adapted",
                "obs_mode": obs_meta["obs_mode"],
                "n_obs": int(obs_meta.get("n_obs", 0)),
                "grid_stride": int(obs_meta.get("grid_stride", 0)),
                "rep": int(rep),
                "block": name,
                "true": float(tv),
                "identified": float(hv),
                "abs_error": float(abs(hv - tv)),
                "rel_error": float(abs(hv - tv) / (abs(tv) + 1.0e-14)),
            })
    if c_p is not None:
        for name, tv, hv in zip(FEATURES, c_true, c_p):
            coeff_rows.append({
                "experiment": "shorttime_longrollout",
                "method": "pointwise_fd",
                "basis_mode": "boundary_adapted",
                "obs_mode": obs_meta["obs_mode"],
                "n_obs": int(obs_meta.get("n_obs", 0)),
                "grid_stride": int(obs_meta.get("grid_stride", 0)),
                "rep": int(rep),
                "block": name,
                "true": float(tv),
                "identified": float(hv),
                "abs_error": float(abs(hv - tv)),
                "rel_error": float(abs(hv - tv) / (abs(tv) + 1.0e-14)),
            })

    rep_artifacts = {
        "ts_roll": ts_roll,
        "Z_roll_ours": Z_roll_ours,
        "sensor_idx": sensor_idx.copy(),
    }
    return trial_rows, coeff_rows, rep_artifacts


def evaluate_basis_ablation_trial(domain_est: core.DomainReduced, ref_bundle: ReferenceBundle, args: argparse.Namespace, rep: int, sensor_idx: np.ndarray, obs_meta: Dict[str, Any], experiment_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(int(args.seed) + 900001 * rep + 13 * int(obs_meta.get("n_obs", 0)) + 31 * int(obs_meta.get("grid_stride", 0)))
    coeffs_true = ref_bundle.coeffs_true
    U_train, _ = stack_sampled_fields(ref_bundle.train_trajs, ref_bundle.id_indices)
    U_obs_flat = U_train[:, :, sensor_idx].reshape(-1, len(sensor_idx))
    U_obs_flat = add_obs_noise(U_obs_flat, rng, args.obs_noise_rel)
    smooth_target = str(getattr(args, "temporal_smooth_target", "none")).strip().lower()
    if smooth_target in ("obs", "both"):
        U_obs_flat = temporal_smooth_sequence(U_obs_flat.reshape(U_train.shape[0], U_train.shape[1], -1), args).reshape(-1, len(sensor_idx))
    Z_hat_flat, rec = recover_reduced_coordinates(domain_est, U_obs_flat, sensor_idx, args)
    Z_hat = Z_hat_flat.reshape(U_train.shape[0], U_train.shape[1], -1)
    if smooth_target in ("z", "both"):
        Z_hat = temporal_smooth_sequence(Z_hat, args)
    X, Y = build_simpson_regression_from_sequences(domain_est, Z_hat, args.dt_obs)
    met, c_hat = solve_regression_metrics(X, Y, coeffs_true, args)
    row = {
        "experiment": experiment_name,
        "method": basis_label(args.current_basis_mode),
        "basis_mode": args.current_basis_mode,
        "reference_mode": args.reference_mode,
        "obs_mode": obs_meta["obs_mode"],
        "n_obs": int(obs_meta.get("n_obs", 0)),
        "grid_stride": int(obs_meta.get("grid_stride", 0)),
        "rep": int(rep),
        "boundary_residual_domain": float(domain_est.boundary_residual),
        "projection_error_mean": float(np.mean(projection_error_rows(domain_est, ref_bundle.train_trajs, "ablation"))),
        "projection_error_max": float(np.max(projection_error_rows(domain_est, ref_bundle.train_trajs, "ablation"))),
    }
    row.update(rec)
    row.update(met)
    coeff_hat = {name: float(v) for name, v in zip(FEATURES, c_hat)}
    roll, ts_roll, Z_roll = rollout_with_coefficients(domain_est, coeff_hat, ref_bundle.heldout_traj, args)
    row.update(roll)
    coeff_rows: List[Dict[str, Any]] = []
    for name, tv, hv in zip(FEATURES, core.coeff_array(coeffs_true), c_hat):
        coeff_rows.append({
            "experiment": experiment_name,
            "method": basis_label(args.current_basis_mode),
            "basis_mode": args.current_basis_mode,
            "obs_mode": obs_meta["obs_mode"],
            "n_obs": int(obs_meta.get("n_obs", 0)),
            "grid_stride": int(obs_meta.get("grid_stride", 0)),
            "rep": int(rep),
            "block": name,
            "true": float(tv),
            "identified": float(hv),
            "abs_error": float(abs(hv - tv)),
            "rel_error": float(abs(hv - tv) / (abs(tv) + 1.0e-14)),
        })
    return row, coeff_rows, Z_roll


def representative_setting(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any] | None:
    cand = [r for r in rows if r.get("method") == method and r.get("experiment") == "shorttime_longrollout"]
    if not cand:
        return None
    if any(int(r.get("n_obs", 0)) > 0 for r in cand):
        target = max(int(r.get("n_obs", 0)) for r in cand)
        cand = [r for r in cand if int(r.get("n_obs", 0)) == target]
    elif any(int(r.get("grid_stride", 0)) > 0 for r in cand):
        target = min(int(r.get("grid_stride", 0)) for r in cand if int(r.get("grid_stride", 0)) > 0)
        cand = [r for r in cand if int(r.get("grid_stride", 0)) == target]
    return min(cand, key=lambda r: float(r.get("mean_active_rel_error", float("inf"))))


def plot_projection_vs_rank(path: Path, rows: List[Dict[str, Any]], metric: str, ylabel: str) -> None:
    K_vals = sorted({int(r["K"]) for r in rows})
    rank_vals = sorted({int(r["rank"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(K_vals)))
    for color, K in zip(colors, K_vals):
        xs: List[int] = []
        ys: List[float] = []
        for rank in rank_vals:
            rr = [r for r in rows if int(r["K"]) == K and int(r["rank"]) == rank]
            if rr:
                xs.append(rank)
                ys.append(float(rr[0][metric]))
        ax.semilogy(xs, ys, marker="o", color=color, label=f"K={K}")
    ax.set_xlabel("reduced rank")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs rank")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_spatial_curves(path: Path, agg_rows: List[Dict[str, Any]], args: argparse.Namespace, metric: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    methods = [m for m in ["exact_simpson_upper_bound", "ours", "projected_fd", "pointwise_fd"] if any(r["method"] == m for r in agg_rows)]
    styles = {
        "exact_simpson_upper_bound": ("o", "-"),
        "ours": ("s", "-"),
        "projected_fd": ("d", "--"),
        "pointwise_fd": ("^", ":"),
    }
    for method in methods:
        rr = [r for r in agg_rows if r["method"] == method and r["experiment"] == "shorttime_longrollout"]
        rr = sorted(rr, key=lambda r: (int(r.get("grid_stride", 0)) if int(r.get("grid_stride", 0)) > 0 else int(r.get("n_obs", 0))))
        xs = [int(r.get("grid_stride", 0)) if int(r.get("grid_stride", 0)) > 0 else int(r.get("n_obs", 0)) for r in rr]
        ys = [float(r[f"{metric}_mean"]) for r in rr]
        marker, ls = styles[method]
        ax.semilogy(xs, ys, marker=marker, linestyle=ls, label=method.replace("_", " "))
    if args.obs_mode == "grid":
        ax.set_xlabel("grid observation stride")
    else:
        ax.set_xlabel("number of spatial sensors N_obs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_sensor_reconstruction(path: Path, agg_rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    ours = [r for r in agg_rows if r["method"] == "ours" and r["experiment"] == "shorttime_longrollout"]
    ours = sorted(ours, key=lambda r: (int(r.get("grid_stride", 0)) if int(r.get("grid_stride", 0)) > 0 else int(r.get("n_obs", 0))))
    xs = [int(r.get("grid_stride", 0)) if int(r.get("grid_stride", 0)) > 0 else int(r.get("n_obs", 0)) for r in ours]
    ys = [float(r["sensor_reconstruction_rel_l2_mean_mean"]) for r in ours]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(xs, ys, marker="o")
    if args.obs_mode == "grid":
        ax.set_xlabel("grid observation stride")
    else:
        ax.set_xlabel("number of spatial sensors N_obs")
    ax.set_ylabel("sensor reconstruction error")
    ax.set_title("Boundary-adapted reconstruction from observations")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_one_row_heatmap(path: Path, rows: List[Dict[str, Any]], method: str, metric: str, title: str, args: argparse.Namespace) -> None:
    rr = [r for r in rows if r["method"] == method and r["experiment"] == "shorttime_longrollout"]
    rr = sorted(rr, key=lambda r: (int(r.get("grid_stride", 0)) if int(r.get("grid_stride", 0)) > 0 else int(r.get("n_obs", 0))))
    if not rr:
        return
    xs = [int(r.get("grid_stride", 0)) if int(r.get("grid_stride", 0)) > 0 else int(r.get("n_obs", 0)) for r in rr]
    Z = np.asarray([[math.log10(max(float(r[f"{metric}_mean"]), 1.0e-16)) for r in rr]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(0.9 * len(xs) + 2.8, 2.5))
    im = ax.imshow(Z, aspect="auto", origin="lower")
    ax.set_yticks([0])
    ax.set_yticklabels([method.replace("_", " ")])
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_title(title)
    if args.obs_mode == "grid":
        ax.set_xlabel("grid observation stride")
    else:
        ax.set_xlabel("number of spatial sensors N_obs")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"log10({metric})")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_rollout_snapshots(path: Path, domain: core.DomainReduced, traj_ref: ReferenceTrajectory, ts_pred: np.ndarray, Z_pred: np.ndarray, args: argparse.Namespace, title: str) -> None:
    display_grid = core.build_display_grid(domain.shape, args, int(args.rollout_plot_N))
    display_grid["Phi_r"] = core.eval_basis(display_grid["x_in"], display_grid["y_in"], domain.meta) @ domain.N
    n = min(len(ts_pred), len(traj_ref.ts))
    idx = np.unique(np.linspace(0, n - 1, 6, dtype=int))
    vals: List[np.ndarray] = []
    for i in idx:
        vals.append(traj_ref.U[i])
        vals.append(domain.Phi_r @ Z_pred[i])
        vals.append((domain.Phi_r @ Z_pred[i]) - traj_ref.U[i])
    vmax = max(float(np.percentile(np.abs(np.concatenate(vals)), 99.5)), 1.0e-3)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    domain_bg = np.where(display_grid["mask"], 1.0, np.nan)
    fig, axes = plt.subplots(3, len(idx), figsize=(3.0 * len(idx), 7.7), constrained_layout=True)
    for c, i in enumerate(idx):
        for r, label in enumerate(["true", "pred", "error"]):
            ax = axes[r, c]
            ax.imshow(np.ma.masked_invalid(domain_bg).T, origin="lower", extent=display_grid["extent"], cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest", alpha=0.06)
            if label == "true":
                z_true = core.project_field_to_z(domain, traj_ref.U[i])
                img = core.render_state_on_display_grid(display_grid, z_true)
            elif label == "pred":
                img = core.render_state_on_display_grid(display_grid, Z_pred[i])
            else:
                z_err = core.project_field_to_z(domain, (domain.Phi_r @ Z_pred[i]) - traj_ref.U[i])
                img = core.render_state_on_display_grid(display_grid, z_err)
            im = ax.imshow(np.ma.masked_invalid(img).T, origin="lower", extent=display_grid["extent"], cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(f"{label}, t={float(traj_ref.ts[i]):.3f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    fig.suptitle(title)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_rollout_error_curves(path: Path, rollout_rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    styles = {
        "ours": ("o", "-"),
        "projected_fd": ("d", "--"),
        "ambient rank-matched": ("s", "-."),
        "boundary-adapted": ("o", "-"),
        "full ambient": ("^", ":"),
    }
    for row in rollout_rows:
        label = str(row["label"])
        errs = np.asarray(row["errors"], dtype=np.float64)
        marker, ls = styles.get(label, ("o", "-"))
        ax.semilogy(np.arange(len(errs)), errs, marker=marker, linestyle=ls, label=label)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("rollout relative L2 error")
    ax.set_title("Held-out rollout error")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_boundary_curves(path: Path, ts: np.ndarray, curves: List[Tuple[str, np.ndarray]], ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for label, vals in curves:
        ax.semilogy(ts[:len(vals)], vals, marker="o", label=label)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_boundary_ablation_snapshots(path: Path, domains: List[core.DomainReduced], labels: List[str], traj_ref: ReferenceTrajectory, Zs: List[np.ndarray], args: argparse.Namespace) -> None:
    ref_domain = domains[0]
    idx = np.unique(np.linspace(0, min(len(traj_ref.ts), *(len(Z) for Z in Zs)) - 1, 6, dtype=int))
    display_grids: List[Dict[str, Any]] = []
    for domain in domains:
        dg = core.build_display_grid(domain.shape, args, int(args.rollout_plot_N))
        dg["Phi_r"] = core.eval_basis(dg["x_in"], dg["y_in"], domain.meta) @ domain.N
        display_grids.append(dg)
    vals: List[np.ndarray] = []
    for i in idx:
        vals.append(traj_ref.U[i])
        for domain, Z in zip(domains, Zs):
            vals.append(domain.Phi_r @ Z[i])
    vmax = max(float(np.percentile(np.abs(np.concatenate(vals)), 99.5)), 1.0e-3)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    fig, axes = plt.subplots(1 + len(domains), len(idx), figsize=(3.0 * len(idx), 2.45 * (1 + len(domains))), constrained_layout=True)
    if axes.ndim == 1:
        axes = axes[None, :]
    for c, i in enumerate(idx):
        dg0 = display_grids[0]
        for r in range(1 + len(domains)):
            ax = axes[r, c]
            domain_bg = np.where(dg0["mask"], 1.0, np.nan)
            ax.imshow(np.ma.masked_invalid(domain_bg).T, origin="lower", extent=dg0["extent"], cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest", alpha=0.06)
            if r == 0:
                z_true = core.project_field_to_z(ref_domain, traj_ref.U[i])
                img = core.render_state_on_display_grid(dg0, z_true)
                label = "true"
            else:
                dg = display_grids[r - 1]
                img = core.render_state_on_display_grid(dg, Zs[r - 1][i])
                label = labels[r - 1]
            im = ax.imshow(np.ma.masked_invalid(img).T, origin="lower", extent=dg0["extent"], cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(f"{label}, t={float(traj_ref.ts[i]):.3f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    fig.suptitle("Boundary-adaptation ablation snapshots")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boundary-aware short-time identification and long-rollout experiments")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shape", type=str, required=True, choices=["channel", "peanut"])
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--rank", type=int, required=True, help="Retained boundary-adapted rank.")
    p.add_argument("--laplace_checkpoint", type=str, required=True)
    p.add_argument("--transport_checkpoint", type=str, required=True)
    p.add_argument("--block_device", type=str, default="cpu")
    p.add_argument("--K_list", type=core.parse_int_list, default=core.parse_int_list("22,30,40"))
    p.add_argument("--rank_list", type=core.parse_int_list, default=core.parse_int_list("80,120,180,240,300"))
    p.add_argument("--Nx", type=int, required=True)
    p.add_argument("--Nb", type=int, required=True)
    p.add_argument("--tau_rel", type=float, default=1.0e-10)
    p.add_argument("--tau_mass", type=float, default=1.0e-12)
    p.add_argument("--peanut_R0", type=float, default=0.52)
    p.add_argument("--peanut_eps", type=float, default=0.42)
    p.add_argument("--channel_L", type=float, default=0.78)
    p.add_argument("--channel_w", type=float, default=0.18)
    p.add_argument("--channel_A", type=float, default=0.12)
    p.add_argument("--channel_angle", type=float, default=math.pi / 6.0)

    p.add_argument("--n_pairs", type=int, required=True, help="Number of independent short trajectories.")
    p.add_argument("--T_id", type=float, required=True)
    p.add_argument("--T_roll", type=float, required=True)
    p.add_argument("--dt_obs", type=float, required=True)
    p.add_argument("--dt_solver", type=float, required=True, help="Sampling step for held-out long rollout validation.")
    p.add_argument("--solver_substeps", type=int, default=4, help="Internal RK4 substeps per long-rollout sample step.")
    p.add_argument("--train_integrator", type=str, default="rk4", choices=["rk4", "imex"], help="Integrator for short identification-reference trajectories.")
    p.add_argument("--rollout_integrator", type=str, default="imex", choices=["rk4", "imex"], help="Integrator for held-out long rollout reference generation and reduced rollout validation.")
    p.add_argument("--obs_mode", type=str, required=True, choices=["sparse", "grid", "dense"])
    p.add_argument("--obs_list", type=core.parse_int_list, required=True)
    p.add_argument("--grid_obs_stride_list", type=core.parse_int_list, default=core.parse_int_list("1,2,3,4"))
    p.add_argument("--n_repeats", type=int, required=True)
    p.add_argument("--sensor_retries", type=int, default=30)
    p.add_argument("--obs_noise_rel", type=float, default=0.0)

    p.add_argument("--amp", type=float, default=0.24)
    p.add_argument("--amp_list", type=str, default="0.08,0.12,0.16,0.20,0.24")
    p.add_argument("--ic_low_dim", type=int, default=36)
    p.add_argument("--ic_decay", type=float, default=0.35)
    p.add_argument("--rollout_max_amp", type=float, default=100.0, help="Amplitude safety cap for long-rollout trajectories. Use a higher value for T_roll >= 1 studies.")
    p.add_argument("--rollout_plot_N", type=int, default=260)

    p.add_argument("--z_ridge", type=float, default=1.0e-8)
    p.add_argument("--mode_reg_power", type=float, default=2.0)
    p.add_argument("--obs_rank_factor", type=float, default=0.60)
    p.add_argument("--obs_rank_cap", type=int, default=0)
    p.add_argument("--coeff_ridge", type=float, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--max_threshold_iterations", type=int, required=True)
    p.add_argument("--active_tol", type=float, required=True)
    p.add_argument("--temporal_smooth_target", type=str, default="none", choices=["none", "obs", "z", "both"])
    p.add_argument("--temporal_smooth_kind", type=str, default="none", choices=["none", "savgol"])
    p.add_argument("--temporal_smooth_window", type=int, default=5)
    p.add_argument("--temporal_smooth_polyorder", type=int, default=3)

    p.add_argument("--reference_mode", type=str, required=True, choices=["reduced_consistent", "projected_strong"])
    p.add_argument("--projected_strong_backend", type=str, default="dense_render_reproject", choices=["dense_render_reproject", "legacy_masked_fd"])
    p.add_argument("--reference_render_Nx", type=int, default=151)
    p.add_argument("--reference_render_Nb", type=int, default=1400)
    p.add_argument("--reference_solver_substeps", type=int, default=8)
    p.add_argument("--basis_mode", type=str, default="boundary_adapted", choices=["boundary_adapted", "ambient_rank_matched", "full_ambient", "ambient"])
    p.add_argument("--baseline_modes", type=parse_string_list, required=True)
    p.add_argument("--boundary_ablation_modes", type=parse_string_list, default=parse_string_list("boundary_adapted,ambient_rank_matched"))
    p.add_argument("--full_ambient_rank_cap", type=int, default=0)
    p.add_argument("--coeffs", type=str, default="")

    p.add_argument("--run_projection_diagnostic_only", type=int, default=0)
    p.add_argument("--skip_projection_diagnostic", type=int, required=True)
    p.add_argument("--skip_boundary_ablation", type=int, default=0)
    p.add_argument("--projection_error_warn", type=float, default=1.0e-3)
    p.add_argument("--true_residual_warn", type=float, default=1.0e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    core.configure_frozen_blocks(
        fourier_cutoff=int(args.K),
        laplace_checkpoint=args.laplace_checkpoint,
        transport_checkpoint=args.transport_checkpoint,
        device=args.block_device,
    )
    outdir = core.ensure_dir(args.outdir)
    np.random.seed(args.seed)
    print(f"[version] {CODE_VERSION}")
    print(f"[args] shape={args.shape}, reference_mode={args.reference_mode}, basis_mode={args.basis_mode}, obs_mode={args.obs_mode}")
    fd_note = fd_baseline_note(args)
    if fd_note:
        print(f"[fd-note] {fd_note}")

    projection_rows: List[Dict[str, Any]] = []
    if not int(args.skip_projection_diagnostic) or int(args.run_projection_diagnostic_only):
        projection_rows = run_projection_diagnostic(args)
        if int(args.run_projection_diagnostic_only):
            core.write_json(outdir / "summary_all.json", {
                "version": CODE_VERSION,
                "mode": "projection_diagnostic_only",
                "args": vars(args),
                "fd_baseline_note": fd_note,
                "projection_rows": projection_rows,
            })
            print("[done] projection diagnostic only")
            return

    coeffs_true = parse_coeffs(args.coeffs, args.shape)
    meta = core.make_real_trig_basis_metadata(int(args.K))
    ref_domain = build_basis_domain(meta, args, "boundary_adapted", int(args.K), int(args.rank))
    args.rank_effective_boundary = int(ref_domain.N.shape[1])
    if int(args.rank) <= 0:
        print(f"[rank] auto mode enabled: requested rank={args.rank}, effective boundary-adapted rank={args.rank_effective_boundary}")

    trial_rows: List[Dict[str, Any]] = []
    coeff_rows: List[Dict[str, Any]] = []
    rollout_curve_rows: List[Dict[str, Any]] = []
    representative_payload: Dict[str, Any] = {}
    reference_cache: Dict[int, Dict[str, Any]] = {}

    spatial_summary_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    boundary_coeff_rows: List[Dict[str, Any]] = []
    boundary_curves: List[Tuple[str, np.ndarray]] = []
    boundary_rollout_curves: List[Tuple[str, np.ndarray]] = []
    boundary_domains: List[core.DomainReduced] = []
    boundary_snapshots: List[np.ndarray] = []
    boundary_labels: List[str] = []
    boundary_ref_traj: ReferenceTrajectory | None = None
    boundary_ts: np.ndarray | None = None

    for rep in range(int(args.n_repeats)):
        print("\n" + "=" * 96)
        print(f"[repeat] rep={rep}")
        print("=" * 96)
        ref_bundle = generate_reference_bundle(ref_domain, coeffs_true, args, rep)
        settings = choose_observation_settings(ref_domain, args, rep)
        rep_artifacts_by_setting: Dict[str, Dict[str, Any]] = {}
        cache_entry: Dict[str, Any] = {"ref_bundle": ref_bundle, "settings": {}}
        for sensor_idx, obs_meta in settings:
            label = observation_setting_label(obs_meta)
            print(f"  [setting] {label}")
            rows_one, coeffs_one, rep_artifacts = evaluate_boundary_adapted_trial(ref_bundle, args, rep, sensor_idx, obs_meta)
            trial_rows.extend(rows_one)
            coeff_rows.extend(coeffs_one)
            rep_artifacts_by_setting[label] = rep_artifacts
            cache_entry["settings"][label] = {
                "sensor_idx": np.asarray(sensor_idx, dtype=np.int64).copy(),
                "obs_meta": dict(obs_meta),
                "rep_artifacts": rep_artifacts,
            }
            ours_row = [r for r in rows_one if r["method"] == "ours"][0]
            fd_row = [r for r in rows_one if r["method"] == "projected_fd"][0]
            print(
                f"    ours coef err={ours_row['mean_active_rel_error']:.2e}, "
                f"fd coef err={fd_row['mean_active_rel_error']:.2e}, "
                f"ours rollout={ours_row['rollout_rel_l2_final']:.2e}, "
                f"fd rollout={fd_row['rollout_rel_l2_final']:.2e}"
            )
        reference_cache[int(rep)] = cache_entry

        if not int(args.skip_boundary_ablation):
            if args.obs_mode == "grid":
                ablation_setting = min(settings, key=lambda item: int(item[1]["grid_stride"]))
            else:
                ablation_setting = max(settings, key=lambda item: int(item[1]["n_obs"]))
            sensor_idx_ablation, obs_meta_ablation = ablation_setting
            for mode in args.boundary_ablation_modes:
                args.current_basis_mode = mode
                domain_est = build_basis_domain(meta, args, mode, int(args.K), int(args.rank))
                row, coeffs_mode, Z_roll = evaluate_basis_ablation_trial(domain_est, ref_bundle, args, rep, sensor_idx_ablation, obs_meta_ablation, "boundary_ablation")
                boundary_rows.append(row)
                boundary_coeff_rows.extend(coeffs_mode)
                if rep == 0:
                    coeffs_hat = {r["block"]: r["identified"] for r in coeffs_mode}
                    roll, ts_roll, Zpred = rollout_with_coefficients(domain_est, coeffs_hat, ref_bundle.heldout_traj, args)
                    errs = []
                    brms = []
                    n = min(len(Zpred), len(ref_bundle.heldout_traj.U))
                    for i in range(n):
                        errs.append(core.weighted_rel_l2(domain_est.Phi_r @ Zpred[i], ref_bundle.heldout_traj.U[i], domain_est.grid["w"]))
                        b = evaluate_boundary_trace(domain_est, Zpred[i]).ravel()
                        brms.append(float(np.sqrt(np.mean(b * b))))
                    boundary_curves.append((basis_label(mode), np.asarray(brms, dtype=np.float64)))
                    boundary_rollout_curves.append((basis_label(mode), np.asarray(errs, dtype=np.float64)))
                    boundary_domains.append(domain_est)
                    boundary_snapshots.append(Zpred)
                    boundary_labels.append(basis_label(mode))
                    boundary_ref_traj = ref_bundle.heldout_traj
                    boundary_ts = ts_roll

    metric_keys = [
        "sensor_reconstruction_rel_l2_mean",
        "sensor_reconstruction_rel_l2_max",
        "true_coeff_residual",
        "ls_residual",
        "condition_number_col_normalized",
        "mean_active_rel_error",
        "max_active_rel_error",
        "inactive_l1",
        "projection_error_mean",
        "projection_error_max",
        "rollout_rel_l2_mean",
        "rollout_rel_l2_max",
        "rollout_rel_l2_final",
        "boundary_rms_mean",
        "boundary_rms_max",
        "boundary_rms_final",
        "boundary_residual_domain",
    ]
    agg_rows = mean_std_rows(trial_rows, ["experiment", "method", "basis_mode", "obs_mode", "n_obs", "grid_stride", "reference_mode"], metric_keys)
    boundary_agg_rows = mean_std_rows(boundary_rows, ["experiment", "method", "basis_mode", "obs_mode", "n_obs", "grid_stride", "reference_mode"], metric_keys)

    summary_rows = agg_rows + boundary_agg_rows
    coeff_rows_all = coeff_rows + boundary_coeff_rows

    core.write_csv(outdir / "summary_by_trial.csv", trial_rows + boundary_rows)
    core.write_csv(outdir / "summary_by_setting.csv", summary_rows)
    core.write_csv(outdir / "coefficients_by_trial.csv", coeff_rows_all)

    spatial_summary_rows = [r for r in agg_rows if r["experiment"] == "shorttime_longrollout"]
    core.write_csv(outdir / "spatial_sampling_summary.csv", spatial_summary_rows)
    core.write_csv(outdir / "coefficient_error_summary.csv", spatial_summary_rows)
    rollout_error_summary_rows = [
        {
            "experiment": r["experiment"],
            "method": r["method"],
            "basis_mode": r["basis_mode"],
            "obs_mode": r["obs_mode"],
            "n_obs": r["n_obs"],
            "grid_stride": r["grid_stride"],
            "rollout_rel_l2_final_mean": r["rollout_rel_l2_final_mean"],
            "rollout_rel_l2_final_std": r["rollout_rel_l2_final_std"],
            "rollout_rel_l2_mean_mean": r["rollout_rel_l2_mean_mean"],
            "boundary_rms_final_mean": r["boundary_rms_final_mean"],
        }
        for r in spatial_summary_rows
    ]
    core.write_csv(outdir / "rollout_error_summary.csv", rollout_error_summary_rows)
    core.write_json(outdir / "shorttime_identification_summary.json", {
        "version": CODE_VERSION,
        "args": vars(args),
        "coeffs_true": coeffs_true,
        "spatial_summary_rows": spatial_summary_rows,
    })

    if boundary_rows:
        boundary_summary_rows = [r for r in boundary_agg_rows if r["experiment"] == "boundary_ablation"]
        core.write_csv(outdir / "boundary_ablation_summary.csv", boundary_summary_rows)
    else:
        boundary_summary_rows = []

    rep_ours = representative_setting(trial_rows, "ours")
    rep_fd = representative_setting(trial_rows, "projected_fd")
    rep_exact = representative_setting(trial_rows, "exact_simpson_upper_bound")
    rep_coeff_rows: List[Dict[str, Any]] = []
    for ref_method, rep_row in [("ours", rep_ours), ("projected_fd", rep_fd), ("exact_simpson_upper_bound", rep_exact)]:
        if rep_row is None:
            continue
        for row in coeff_rows_all:
            if row["experiment"] == "shorttime_longrollout" and row["method"] == ref_method and int(row["rep"]) == int(rep_row["rep"]) and int(row["n_obs"]) == int(rep_row["n_obs"]) and int(row["grid_stride"]) == int(rep_row["grid_stride"]):
                rep_coeff_rows.append(row)
    core.write_csv(outdir / "identified_coefficients_reference.csv", rep_coeff_rows)

    plot_spatial_curves(outdir / "space_sparsity_curves.png", spatial_summary_rows, args, "mean_active_rel_error", "Coefficient error vs spatial observation", "coefficient relative error")
    plot_sensor_reconstruction(outdir / "sensor_reconstruction_error.png", spatial_summary_rows, args)
    plot_one_row_heatmap(outdir / "heatmap_ours_coef_error.png", spatial_summary_rows, "ours", "mean_active_rel_error", "Ours coefficient error", args)
    plot_one_row_heatmap(outdir / "heatmap_fdproj_coef_error.png", spatial_summary_rows, "projected_fd", "mean_active_rel_error", "Projected FD coefficient error", args)

    ratio_rows: List[Dict[str, Any]] = []
    for ours in [r for r in spatial_summary_rows if r["method"] == "ours"]:
        fd = [r for r in spatial_summary_rows if r["method"] == "projected_fd" and r["obs_mode"] == ours["obs_mode"] and int(r["n_obs"]) == int(ours["n_obs"]) and int(r["grid_stride"]) == int(ours["grid_stride"])]
        if fd:
            rr = dict(ours)
            rr["method"] = "fd_over_ours"
            rr["mean_active_rel_error_mean"] = float(fd[0]["mean_active_rel_error_mean"]) / max(float(ours["mean_active_rel_error_mean"]), 1.0e-16)
            ratio_rows.append(rr)
    plot_one_row_heatmap(outdir / "heatmap_fd_over_ours_ratio.png", ratio_rows, "fd_over_ours", "mean_active_rel_error", "FD / Ours coefficient error ratio", args)

    if rep_ours is not None:
        rep_id = int(rep_ours["rep"])
        rep_label = observation_setting_label(rep_ours)
        rep_cache = reference_cache.get(rep_id, {})
        ref_bundle_rep = rep_cache.get("ref_bundle")
        rep_setting = rep_cache.get("settings", {}).get(rep_label)
        if ref_bundle_rep is None or rep_setting is None:
            ref_bundle_rep = generate_reference_bundle(ref_domain, coeffs_true, args, rep_id)
            sensor_idx, obs_meta = next(
                (idx_meta for idx_meta in choose_observation_settings(ref_domain, args, rep_id) if int(idx_meta[1].get("n_obs", 0)) == int(rep_ours["n_obs"]) and int(idx_meta[1].get("grid_stride", 0)) == int(rep_ours["grid_stride"])),
                choose_observation_settings(ref_domain, args, rep_id)[0],
            )
        else:
            sensor_idx = rep_setting["sensor_idx"]
            obs_meta = rep_setting["obs_meta"]
        coeff_lookup: Dict[str, Dict[str, float]] = {}
        for method_name in ["ours", "projected_fd", "exact_simpson_upper_bound"]:
            coeff_lookup[method_name] = {
                row["block"]: float(row["identified"])
                for row in coeff_rows_all
                if row["experiment"] == "shorttime_longrollout"
                and row["method"] == method_name
                and int(row["rep"]) == rep_id
                and int(row["n_obs"]) == int(rep_ours["n_obs"])
                and int(row["grid_stride"]) == int(rep_ours["grid_stride"])
            }
        rollout_rows_plot: List[Dict[str, Any]] = []
        if coeff_lookup.get("ours"):
            coeff_ours = coeff_lookup["ours"]
            roll, ts_pred, Z_pred = rollout_with_coefficients(ref_domain, coeff_ours, ref_bundle_rep.heldout_traj, args)
            errs = [core.weighted_rel_l2(ref_domain.Phi_r @ z, ref_bundle_rep.heldout_traj.U[i], ref_domain.grid["w"]) for i, z in enumerate(Z_pred[:len(ref_bundle_rep.heldout_traj.U)])]
            rollout_rows_plot.append({"label": "ours", "errors": errs})
            plot_rollout_snapshots(outdir / "rollout_snapshots_true_pred_error.png", ref_domain, ref_bundle_rep.heldout_traj, ts_pred, Z_pred, args, "Held-out rollout: true / prediction / error")
        if coeff_lookup.get("projected_fd"):
            coeff_fd = coeff_lookup["projected_fd"]
            roll, ts_pred, Z_pred_fd = rollout_with_coefficients(ref_domain, coeff_fd, ref_bundle_rep.heldout_traj, args)
            errs = [core.weighted_rel_l2(ref_domain.Phi_r @ z, ref_bundle_rep.heldout_traj.U[i], ref_domain.grid["w"]) for i, z in enumerate(Z_pred_fd[:len(ref_bundle_rep.heldout_traj.U)])]
            rollout_rows_plot.append({"label": "projected_fd", "errors": errs})
        if boundary_rollout_curves:
            for label, vals in boundary_rollout_curves:
                if label == "ambient rank-matched":
                    rollout_rows_plot.append({"label": label, "errors": vals})
        if rollout_rows_plot:
            plot_rollout_error_curves(outdir / "rollout_error_curve.png", rollout_rows_plot, args)

    if boundary_rows and boundary_ts is not None and boundary_ref_traj is not None:
        plot_boundary_curves(outdir / "boundary_residual_curve.png", boundary_ts, boundary_curves, "boundary residual", "Boundary residual during held-out rollout")
        plot_boundary_curves(outdir / "ambient_vs_boundary_rollout_error.png", boundary_ts, boundary_rollout_curves, "rollout relative L2 error", "Boundary-adapted vs ambient rollout error")
        plot_boundary_ablation_snapshots(outdir / "boundary_ablation_snapshots.png", boundary_domains, boundary_labels, boundary_ref_traj, boundary_snapshots, args)

    core.write_json(outdir / "summary_all.json", {
        "version": CODE_VERSION,
        "args": vars(args),
        "coeffs_true": coeffs_true,
        "fd_baseline_note": fd_note,
        "projection_rows": projection_rows,
        "trial_rows": trial_rows,
        "boundary_rows": boundary_rows,
        "aggregate_rows": summary_rows,
        "identified_reference_setting": rep_ours,
    })
    print("[done] saved to", outdir)


if __name__ == "__main__":
    main()
