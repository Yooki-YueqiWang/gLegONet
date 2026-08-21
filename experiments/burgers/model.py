#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare pretrained ambient Burgers/Laplace blocks against exact S3
operators on the same boundary-adapted reduced space for the inner-square
vector Burgers problem.

The reduced space is built on Omega=[-h,h]^2 from the ambient real
trigonometric basis on Q=[-1,1]^2 with radial cutoff k^2+ell^2 <= K^2.  Two
rollouts are then compared:

1. exact S3:
   same reduced space + exact projected operators;
2. learned block:
   same reduced space + transferred learned Laplace block and learned scalar
   primitive h'_theta for the single-component transport blocks, while the
   mixed terms remain the exact reduced projections.

This isolates block error from boundary transfer and reduced-space truncation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as bc

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def torch_load(path: str | Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def inv_softplus(y: torch.Tensor | float) -> torch.Tensor:
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return y_t + torch.log(-torch.expm1(-y_t))


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
    kind = np.array([e[2] for e in entries])
    r = np.sqrt(k_arr.astype(np.float64) ** 2 + ell_arr.astype(np.float64) ** 2)
    lam = (math.pi ** 2) * (r ** 2)
    return {
        "entries": entries,
        "pairs": pairs,
        "pair_indices": pair_indices,
        "k": k_arr,
        "ell": ell_arr,
        "kind": kind,
        "lambda": lam.astype(np.float64),
        "sigma": ((1.0 + r) ** (-0.5)).astype(np.float64),
    }


def eval_basis_and_grad(x: np.ndarray, y: np.ndarray, meta: Dict[str, Any], L: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    M = len(meta["k"])
    Phi = np.empty((x.size, M), dtype=np.float64)
    Phix = np.empty_like(Phi)
    Phiy = np.empty_like(Phi)
    sqrt2 = math.sqrt(2.0)
    sf = math.pi / float(L)
    for j, (k, ell, kind) in enumerate(zip(meta["k"], meta["ell"], meta["kind"])):
        if kind == "const":
            Phi[:, j] = 1.0
            Phix[:, j] = 0.0
            Phiy[:, j] = 0.0
        else:
            phase = math.pi * (float(k) * x + float(ell) * y) / float(L)
            if kind == "cos":
                Phi[:, j] = sqrt2 * np.cos(phase)
                Phix[:, j] = -sqrt2 * sf * float(k) * np.sin(phase)
                Phiy[:, j] = -sqrt2 * sf * float(ell) * np.sin(phase)
            else:
                Phi[:, j] = sqrt2 * np.sin(phase)
                Phix[:, j] = sqrt2 * sf * float(k) * np.cos(phase)
                Phiy[:, j] = sqrt2 * sf * float(ell) * np.cos(phase)
    return Phi, Phix, Phiy


def assemble_mass(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = Phi.T @ (w[:, None] * Phi)
    return 0.5 * (M + M.T)


def nullspace_from_C(C: np.ndarray, tau_rel: float) -> Dict[str, Any]:
    _, S, Vh = np.linalg.svd(C, full_matrices=True)
    tau = float(tau_rel) * max(float(S[0]) if S.size else 1.0, 1.0)
    r = int(np.sum(S > tau))
    return {
        "rank": r,
        "null_dim": int(Vh.shape[1] - r),
        "S": S,
        "tau": tau,
        "N_raw": Vh[r:, :].T.copy(),
    }


def mass_orthonormalize(N_raw: np.ndarray, M: np.ndarray, tau_mass: float = 1.0e-12) -> Tuple[np.ndarray, Dict[str, Any]]:
    G = 0.5 * (N_raw.T @ M @ N_raw + (N_raw.T @ M @ N_raw).T)
    e, V = np.linalg.eigh(G)
    order = np.argsort(e)[::-1]
    e = e[order]
    V = V[:, order]
    keep = e > float(tau_mass) * max(float(e[0]), 1.0)
    if not np.any(keep):
        raise RuntimeError("empty mass-positive nullspace")
    Z = N_raw @ (V[:, keep] / np.sqrt(e[keep])[None, :])
    return Z, {
        "positive_mass_dim": int(Z.shape[1]),
        "mass_eigs_minmax": [float(e[keep].min()), float(e[keep].max())],
        "dropped": int(N_raw.shape[1] - np.sum(keep)),
    }


class PositiveDiagonalLaplace(nn.Module):
    def __init__(self, lambda_vec: torch.Tensor, init_scaled_diag: float = 0.10):
        super().__init__()
        self.register_buffer("lambda_vec", lambda_vec.float())
        scale = float(torch.max(lambda_vec).item())
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("nonzero_mask", (lambda_vec > 0).float())
        raw0 = inv_softplus(init_scaled_diag).expand_as(lambda_vec).clone()
        self.raw_diag = nn.Parameter(raw0)

    def scaled_diag(self) -> torch.Tensor:
        return F.softplus(self.raw_diag) * self.nonzero_mask

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return -self.scale * self.scaled_diag().unsqueeze(0) * a


class DensityNet(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4, act: str = "gelu"):
        super().__init__()
        act_l = act.lower()
        if act_l == "gelu":
            Act = nn.GELU
        elif act_l == "silu":
            Act = nn.SiLU
        elif act_l == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"unknown activation {act}")
        layers: List[nn.Module] = []
        in_features = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(Act())
            in_features = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u.reshape(-1, 1)).reshape_as(u)


def tabulate_density_derivative(model: DensityNet, device: torch.device, smin: float, smax: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = torch.linspace(float(smin), float(smax), int(n), dtype=torch.float32, device=device)
    with torch.enable_grad():
        xreq = xs.detach().clone().requires_grad_(True)
        h = model(xreq)
        g = torch.autograd.grad(h.sum(), xreq, create_graph=False)[0]
    return xs.detach().cpu().numpy().astype(np.float64), g.detach().cpu().numpy().astype(np.float64)


@dataclass
class TorchReducedVectorBurgersModel:
    name: str
    mode: str
    device: torch.device
    dtype: torch.dtype
    E: torch.Tensor
    Ex: torch.Tensor
    Ey: torch.Tensor
    P0: torch.Tensor
    Gx: torch.Tensor
    Gy: torch.Tensor
    D: torch.Tensor
    w: torch.Tensor
    C_eval: Optional[torch.Tensor]
    table_x: Optional[torch.Tensor] = None
    table_g: Optional[torch.Tensor] = None
    build_time: float = 0.0
    memory_mib: float = 0.0
    observed_absmax: float = 0.0

    def primitive(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == "exact":
            return 0.5 * u * u
        if self.table_x is None or self.table_g is None:
            raise RuntimeError("learned primitive table is missing")
        u_flat = u.reshape(-1)
        umax = float(torch.max(torch.abs(u_flat)).item())
        if umax > self.observed_absmax:
            self.observed_absmax = umax
        x = torch.clamp(u_flat, min=float(self.table_x[0].item()), max=float(self.table_x[-1].item()))
        idx = torch.searchsorted(self.table_x, x, right=False)
        idx = torch.clamp(idx, 1, self.table_x.numel() - 1)
        x0 = self.table_x[idx - 1]
        x1 = self.table_x[idx]
        y0 = self.table_g[idx - 1]
        y1 = self.table_g[idx]
        t = (x - x0) / torch.clamp(x1 - x0, min=1.0e-12)
        out = y0 + t * (y1 - y0)
        return out.reshape_as(u)

    def lift(self, z: torch.Tensor) -> torch.Tensor:
        return self.E @ z

    def boundary_residual(self, z: torch.Tensor) -> float:
        if self.C_eval is None:
            return 0.0
        vals = self.C_eval @ z
        den = torch.linalg.norm(z) + 1.0e-14
        return float((torch.linalg.norm(vals) / den).item())

    def kinetic_energy(self, zu: torch.Tensor, zv: torch.Tensor) -> float:
        u = self.E @ zu
        v = self.E @ zv
        return 0.5 * float(torch.sum(self.w * (u * u + v * v)).item())

    def nonlinear_rhs(self, zu: torch.Tensor, zv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.E @ zu
        v = self.E @ zv
        uy = self.Ey @ zu
        vx = self.Ex @ zv
        bx_u = self.Gx @ self.primitive(u)
        by_v = self.Gy @ self.primitive(v)
        cy = self.P0 @ (-(v * uy))
        cx = self.P0 @ (-(u * vx))
        return bx_u + cy, cx + by_v

    def rk4_nonlinear(self, zu: torch.Tensor, zv: torch.Tensor, tau: float, nsub: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if abs(tau) < 1.0e-15:
            return zu, zv
        hstep = float(tau) / max(1, int(nsub))
        u = zu
        v = zv
        for _ in range(max(1, int(nsub))):
            k1u, k1v = self.nonlinear_rhs(u, v)
            k2u, k2v = self.nonlinear_rhs(u + 0.5 * hstep * k1u, v + 0.5 * hstep * k1v)
            k3u, k3v = self.nonlinear_rhs(u + 0.5 * hstep * k2u, v + 0.5 * hstep * k2v)
            k4u, k4v = self.nonlinear_rhs(u + hstep * k3u, v + hstep * k3v)
            u = u + (hstep / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            v = v + (hstep / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return u, v

    def strang_step(self, zu: torch.Tensor, zv: torch.Tensor, dt: float, nonlinear_substeps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        zu, zv = self.rk4_nonlinear(zu, zv, 0.5 * dt, nonlinear_substeps)
        zu = self.D @ zu
        zv = self.D @ zv
        zu, zv = self.rk4_nonlinear(zu, zv, 0.5 * dt, nonlinear_substeps)
        return zu, zv


def combined_rel_l2_torch(u: torch.Tensor, v: torch.Tensor, ur: torch.Tensor, vr: torch.Tensor, w: torch.Tensor) -> float:
    num = torch.sqrt(torch.clamp(torch.sum(w * ((u - ur) ** 2 + (v - vr) ** 2)), min=0.0))
    den = torch.sqrt(torch.clamp(torch.sum(w * (ur ** 2 + vr ** 2)), min=0.0)) + 1.0e-14
    return float((num / den).item())


def weighted_rel_l2_torch(u: torch.Tensor, ur: torch.Tensor, w: torch.Tensor) -> float:
    num = torch.sqrt(torch.clamp(torch.sum(w * (u - ur) ** 2), min=0.0))
    den = torch.sqrt(torch.clamp(torch.sum(w * ur ** 2), min=0.0)) + 1.0e-14
    return float((num / den).item())


def build_models(args: argparse.Namespace) -> Tuple[TorchReducedVectorBurgersModel, TorchReducedVectorBurgersModel, Dict[str, Any]]:
    t0 = time.time()
    meta = make_real_trig_basis_metadata(args.K)
    x, y, w, xv, yv = bc.inner_square_points(args.n_quad, args.inner_halfwidth)
    Phi, Phix, Phiy = eval_basis_and_grad(x, y, meta, args.box_halfwidth)
    M = assemble_mass(Phi, w)
    xb, yb = bc.boundary_points(args.Nb_build, args.inner_halfwidth)
    xd, yd = bc.boundary_points(args.Nb_dense, args.inner_halfwidth)
    Cb, _, _ = eval_basis_and_grad(xb, yb, meta, args.box_halfwidth)
    Cd, _, _ = eval_basis_and_grad(xd, yd, meta, args.box_halfwidth)

    ns = nullspace_from_C(Cb, args.tau_rel)
    parent, pinfo = mass_orthonormalize(ns["N_raw"], M, tau_mass=args.tau_mass)
    lam = np.asarray(meta["lambda"], dtype=np.float64)
    Lp = parent.T @ (M @ (-(lam[:, None] * parent)))
    Ap = 0.5 * (-(Lp + Lp.T))
    eig, V = np.linalg.eigh(Ap)
    order = np.argsort(eig)
    modes = parent @ V[:, order]
    r = min(int(args.rank), modes.shape[1])
    N = modes[:, :r].copy()
    E = Phi @ N
    Ex = Phix @ N
    Ey = Phiy @ N
    Mr = assemble_mass(E, w)
    P0 = np.linalg.solve(Mr, E.T * w[None, :])
    Gx = np.linalg.solve(Mr, Ex.T * w[None, :])
    Gy = np.linalg.solve(Mr, Ey.T * w[None, :])
    C_eval = Cd @ N

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.float64 if args.use_double else torch.float32

    E_t = torch.as_tensor(E, dtype=dtype, device=device)
    Ex_t = torch.as_tensor(Ex, dtype=dtype, device=device)
    Ey_t = torch.as_tensor(Ey, dtype=dtype, device=device)
    P0_t = torch.as_tensor(P0, dtype=dtype, device=device)
    Gx_t = torch.as_tensor(Gx, dtype=dtype, device=device)
    Gy_t = torch.as_tensor(Gy, dtype=dtype, device=device)
    w_t = torch.as_tensor(w, dtype=dtype, device=device)
    C_eval_t = torch.as_tensor(C_eval, dtype=dtype, device=device)
    Mr_t = torch.as_tensor(Mr, dtype=dtype, device=device)

    # Exact reduced Laplace block.
    Lr_exact = np.linalg.solve(Mr, N.T @ (M @ (-(lam[:, None] * N))))
    Lr_exact_t = torch.as_tensor(Lr_exact, dtype=dtype, device=device)
    I_t = torch.eye(r, dtype=dtype, device=device)
    A_exact = I_t - 0.5 * float(args.dt) * float(args.nu) * Lr_exact_t
    B_exact = I_t + 0.5 * float(args.dt) * float(args.nu) * Lr_exact_t
    D_exact = torch.linalg.solve(A_exact, B_exact)

    # Learned reduced Laplace block.
    lap_ckpt = torch_load(args.laplace_ckpt, device)
    if int(lap_ckpt.get("K", args.K)) != int(args.K) or int(lap_ckpt.get("M", len(meta["k"]))) != len(meta["k"]):
        raise RuntimeError("Laplace checkpoint basis mismatch")
    lap = PositiveDiagonalLaplace(torch.as_tensor(meta["lambda"], dtype=torch.float32, device=device)).to(device)
    lap.load_state_dict(lap_ckpt["model_state"], strict=True)
    lap.eval()
    for pp in lap.parameters():
        pp.requires_grad_(False)
    with torch.no_grad():
        N_batch = torch.as_tensor(N.T, dtype=torch.float32, device=device)
        lap_cols = lap(N_batch).detach().cpu().numpy().T.astype(np.float64)
    Lr_learn = np.linalg.solve(Mr, N.T @ (M @ lap_cols))
    Lr_learn_t = torch.as_tensor(Lr_learn, dtype=dtype, device=device)
    A_learn = I_t - 0.5 * float(args.dt) * float(args.nu) * Lr_learn_t
    B_learn = I_t + 0.5 * float(args.dt) * float(args.nu) * Lr_learn_t
    D_learn = torch.linalg.solve(A_learn, B_learn)

    # Learned primitive table.
    den_ckpt = torch_load(args.burgers_ckpt, device)
    if int(den_ckpt.get("K", args.K)) != int(args.K) or int(den_ckpt.get("M", len(meta["k"]))) != len(meta["k"]):
        raise RuntimeError("Burgers density checkpoint basis mismatch")
    cfg = den_ckpt.get("config", {})
    den = DensityNet(width=int(cfg.get("width", 128)), depth=int(cfg.get("depth", 4)), act=str(cfg.get("act", "gelu"))).to(device)
    den.load_state_dict(den_ckpt["model_state"], strict=True)
    den.eval()
    for pp in den.parameters():
        pp.requires_grad_(False)
    table_x_np, table_g_np = tabulate_density_derivative(den, device, args.table_min, args.table_max, args.table_n)
    table_x_t = torch.as_tensor(table_x_np, dtype=dtype, device=device)
    table_g_t = torch.as_tensor(table_g_np, dtype=dtype, device=device)

    mem = (
        E.nbytes + Ex.nbytes + Ey.nbytes + P0.nbytes + Gx.nbytes + Gy.nbytes +
        N.nbytes + Mr.nbytes + C_eval.nbytes + Lr_exact.nbytes + Lr_learn.nbytes
    ) / (1024.0 ** 2)
    build_time = time.time() - t0

    exact_model = TorchReducedVectorBurgersModel(
        name="Exact S3 (same-space)",
        mode="exact",
        device=device,
        dtype=dtype,
        E=E_t,
        Ex=Ex_t,
        Ey=Ey_t,
        P0=P0_t,
        Gx=Gx_t,
        Gy=Gy_t,
        D=D_exact,
        w=w_t,
        C_eval=C_eval_t,
        build_time=build_time,
        memory_mib=mem,
    )
    learned_model = TorchReducedVectorBurgersModel(
        name="Learned block (same-space)",
        mode="learned",
        device=device,
        dtype=dtype,
        E=E_t,
        Ex=Ex_t,
        Ey=Ey_t,
        P0=P0_t,
        Gx=Gx_t,
        Gy=Gy_t,
        D=D_learn,
        w=w_t,
        C_eval=C_eval_t,
        table_x=table_x_t,
        table_g=table_g_t,
        build_time=build_time,
        memory_mib=mem,
    )
    info = {
        "meta_dim": int(len(meta["k"])),
        "null_info": ns,
        "parent_info": pinfo,
        "rank_used": int(r),
        "laplace_energy_cutoff": float(eig[order][r - 1]) if r > 0 else 0.0,
        "mass_eigs": [float(np.linalg.eigvalsh(Mr).min()), float(np.linalg.eigvalsh(Mr).max())],
        "device": str(device),
        "dtype": str(dtype),
        "checkpoints": {
            "laplace_ckpt": str(args.laplace_ckpt),
            "burgers_ckpt": str(args.burgers_ckpt),
        },
    }
    grid = {"x": x, "y": y, "w": w, "xv": xv, "yv": yv, "h": args.inner_halfwidth, "n_quad": args.n_quad}
    return exact_model, learned_model, {"grid": grid, "info": info}


def run_pairwise_comparison(exact_model: TorchReducedVectorBurgersModel, learned_model: TorchReducedVectorBurgersModel,
                            grid: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    device = exact_model.device
    dtype = exact_model.dtype
    x = np.asarray(grid["x"], dtype=np.float64)
    y = np.asarray(grid["y"], dtype=np.float64)
    w_t = exact_model.w
    u0_np, v0_np = bc.rescaled_sine_ic(x, y, args.inner_halfwidth)
    u0_t = torch.as_tensor(u0_np, dtype=dtype, device=device)
    v0_t = torch.as_tensor(v0_np, dtype=dtype, device=device)
    zu0 = exact_model.P0 @ u0_t
    zv0 = exact_model.P0 @ v0_t
    ze_u = zu0.clone()
    ze_v = zv0.clone()
    zl_u = zu0.clone()
    zl_v = zv0.clone()

    nsteps = int(round(float(args.T) / float(args.dt)))
    rows: List[Dict[str, Any]] = []
    t_start = time.time()
    for step in range(nsteps + 1):
        ue = exact_model.lift(ze_u)
        ve = exact_model.lift(ze_v)
        ul = learned_model.lift(zl_u)
        vl = learned_model.lift(zl_v)
        row = {
            "time": step * float(args.dt),
            "rel_u_learned_vs_exact": weighted_rel_l2_torch(ul, ue, w_t),
            "rel_v_learned_vs_exact": weighted_rel_l2_torch(vl, ve, w_t),
            "rel_combined_learned_vs_exact": combined_rel_l2_torch(ul, vl, ue, ve, w_t),
            "bc_exact_u": exact_model.boundary_residual(ze_u),
            "bc_exact_v": exact_model.boundary_residual(ze_v),
            "bc_learned_u": learned_model.boundary_residual(zl_u),
            "bc_learned_v": learned_model.boundary_residual(zl_v),
            "energy_exact": exact_model.kinetic_energy(ze_u, ze_v),
            "energy_learned": learned_model.kinetic_energy(zl_u, zl_v),
        }
        rows.append(row)
        if step == nsteps:
            break
        ze_u, ze_v = exact_model.strang_step(ze_u, ze_v, args.dt, args.nonlinear_substeps)
        zl_u, zl_v = learned_model.strang_step(zl_u, zl_v, args.dt, args.nonlinear_substeps)
        if args.log_every > 0 and (((step + 1) % args.log_every == 0) or (step + 1 == nsteps)):
            print(
                f"[pair] step={step+1:05d}/{nsteps:05d} t={(step+1)*args.dt:.3f} "
                f"rel={rows[-1]['rel_combined_learned_vs_exact']:.3e}",
                flush=True,
            )
    rollout_time = time.time() - t_start
    final = rows[-1]
    return {
        "setup": vars(args),
        "history": rows,
        "final": final,
        "summary": {
            "max_rel_combined_learned_vs_exact": float(max(r["rel_combined_learned_vs_exact"] for r in rows)),
            "mean_rel_combined_learned_vs_exact": float(sum(r["rel_combined_learned_vs_exact"] for r in rows) / len(rows)),
            "max_rel_u_learned_vs_exact": float(max(r["rel_u_learned_vs_exact"] for r in rows)),
            "max_rel_v_learned_vs_exact": float(max(r["rel_v_learned_vs_exact"] for r in rows)),
            "rollout_time_sec": float(rollout_time),
            "learned_primitive_absmax_seen": float(learned_model.observed_absmax),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare pretrained learned blocks against exact S3 on the same inner-square reduced space.")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--box_halfwidth", type=float, required=True)
    p.add_argument("--inner_halfwidth", type=float, required=True)
    p.add_argument("--n_quad", type=int, required=True)
    p.add_argument("--Nb_build", type=int, required=True)
    p.add_argument("--Nb_dense", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--nu", type=float, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--nonlinear_substeps", type=int, required=True)
    p.add_argument("--tau_rel", type=float, default=1.0e-10)
    p.add_argument("--tau_mass", type=float, default=1.0e-12)
    p.add_argument("--laplace_ckpt", type=str, required=True)
    p.add_argument("--burgers_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_double", action="store_true")
    p.add_argument("--table_min", type=float, default=-1.5)
    p.add_argument("--table_max", type=float, default=1.5)
    p.add_argument("--table_n", type=int, default=200001)
    p.add_argument("--log_every", type=int, default=100)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("Inner-square Burgers: learned block vs exact S3 (same reduced space)", flush=True)
    print("=" * 100, flush=True)
    print(
        f"[setup] K={args.K} rank={args.rank} h={args.inner_halfwidth} nu={args.nu} "
        f"dt={args.dt} T={args.T} Nb={args.Nb_build}",
        flush=True,
    )
    exact_model, learned_model, payload = build_models(args)
    grid = payload["grid"]
    info = payload["info"]
    print(
        f"[basis] ambient_dim={info['meta_dim']} null_dim={info['null_info']['null_dim']} "
        f"rank_used={info['rank_used']} mass={info['mass_eigs'][0]:.3e}/{info['mass_eigs'][1]:.3e}",
        flush=True,
    )
    print(f"[build] device={info['device']} dtype={info['dtype']} memory={exact_model.memory_mib:.1f} MiB", flush=True)

    result = run_pairwise_comparison(exact_model, learned_model, grid, args)
    result["basis_info"] = info
    result["models"] = {
        "exact": {"name": exact_model.name, "build_time": exact_model.build_time, "memory_mib": exact_model.memory_mib},
        "learned": {"name": learned_model.name, "build_time": learned_model.build_time, "memory_mib": learned_model.memory_mib},
    }

    summary_path = outdir / "summary.json"
    history_csv = outdir / "history.csv"
    summary_path.write_text(json.dumps(to_jsonable(result), indent=2), encoding="utf-8")
    write_csv(history_csv, result["history"])

    print("\n[final]", flush=True)
    print(json.dumps(to_jsonable(result["final"]), indent=2), flush=True)
    print("\n[summary]", flush=True)
    print(json.dumps(to_jsonable(result["summary"]), indent=2), flush=True)
    print(f"[done] wrote {summary_path} and {history_csv}", flush=True)


if __name__ == "__main__":
    main()
