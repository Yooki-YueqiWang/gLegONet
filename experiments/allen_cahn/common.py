#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section-3 Allen--Cahn rollout on a disk embedded in a fixed ambient box [-1,1]^2.

Changes relative to the original live-diagnostic script:
- ambient box is fixed by box_halfwidth, default 1.0
- disk size is controlled by disk_diameter, default 0.8
- keeps the Section-3 tangent-space construction and reduced z-space dynamics
- loads the pretrained ambient-square neural Laplace block by default
- removes expensive live projection-ceiling diagnostics and CSV writing
- stores only sampled FEM/reduced states instead of the full trajectory
- keeps the main plots and adds a disk_geometry.png figure

PDE:
    volume-constrained Allen--Cahn
        u_t = eps2 * B_{Delta,theta}^Omega u + u - u^3 - lambda(t),
        lambda(t) = average(u - u^3),
    with homogeneous Neumann BC on the disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
from scipy.spatial import Delaunay

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)


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


def weighted_mean(u: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * u) / np.sum(w))


def bc_violation(a: np.ndarray, C_dense: np.ndarray) -> float:
    return float(np.linalg.norm(C_dense @ a) / (np.linalg.norm(a) + 1e-14))


# -----------------------------------------------------------------------------
# Basis / geometry
# -----------------------------------------------------------------------------

def build_trig_basis_meta(K: int, radial_truncation: bool = True) -> List[Tuple[str, int, int]]:
    """Metadata for the same real Fourier half-plane basis used by train_laplace_block.py.

    The basis on Q=[-Lx,Lx]x[-Ly,Ly] is
        1,
        sqrt(2) cos(pi*(k*x/Lx + ell*y/Ly)),
        sqrt(2) sin(pi*(k*x/Lx + ell*y/Ly)),
    for one representative of each nonzero conjugate pair:
        k > 0, or k == 0 and ell > 0.

    For K=22 this gives M=1517, matching the manuscript ambient space.
    """
    entries: List[Tuple[str, int, int]] = [("const", 0, 0)]
    pairs: List[Tuple[int, int]] = []
    for k in range(-K, K + 1):
        for ell in range(-K, K + 1):
            if k == 0 and ell == 0:
                continue
            if radial_truncation and k * k + ell * ell > K * K:
                continue
            if k > 0 or (k == 0 and ell > 0):
                pairs.append((k, ell))
    pairs.sort(key=lambda q: (q[0] * q[0] + q[1] * q[1], q[0], q[1]))
    for k, ell in pairs:
        entries.append(("cos", k, ell))
        entries.append(("sin", k, ell))
    return entries


def eval_trig_basis_from_meta(x: np.ndarray, y: np.ndarray, meta: List[Tuple[str, int, int]], Lx: float, Ly: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Phi = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    sqrt2 = math.sqrt(2.0)
    for j, (kind, k, ell) in enumerate(meta):
        if kind == "const":
            Phi[:, j] = 1.0
        else:
            phase = np.pi * (float(k) * x / float(Lx) + float(ell) * y / float(Ly))
            if kind == "cos":
                Phi[:, j] = sqrt2 * np.cos(phase)
            elif kind == "sin":
                Phi[:, j] = sqrt2 * np.sin(phase)
            else:
                raise ValueError(kind)
    return Phi


def eval_trig_basis_grad_from_meta(x: np.ndarray, y: np.ndarray, meta: List[Tuple[str, int, int]], Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dphix = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    dphiy = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    sqrt2 = math.sqrt(2.0)
    for j, (kind, k, ell) in enumerate(meta):
        kf = float(k)
        ef = float(ell)
        if kind == "const":
            dphix[:, j] = 0.0
            dphiy[:, j] = 0.0
        else:
            phase = np.pi * (kf * x / float(Lx) + ef * y / float(Ly))
            px = np.pi * kf / float(Lx)
            py = np.pi * ef / float(Ly)
            if kind == "cos":
                sphase = np.sin(phase)
                dphix[:, j] = -sqrt2 * px * sphase
                dphiy[:, j] = -sqrt2 * py * sphase
            elif kind == "sin":
                cphase = np.cos(phase)
                dphix[:, j] = sqrt2 * px * cphase
                dphiy[:, j] = sqrt2 * py * cphase
            else:
                raise ValueError(kind)
    return dphix, dphiy


def circle_boundary_points(Nb: int, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, Nb, endpoint=False, dtype=np.float64)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    nx = np.cos(theta)
    ny = np.sin(theta)
    return x, y, nx, ny


def build_disk_in_box_grid(Nxy: int, box_halfwidth: float, radius: float) -> Dict[str, Any]:
    x = np.linspace(-box_halfwidth, box_halfwidth, Nxy, dtype=np.float64)
    y = np.linspace(-box_halfwidth, box_halfwidth, Nxy, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = X * X + Y * Y <= radius * radius + 1e-14
    h = x[1] - x[0]
    return {
        "X": X,
        "Y": Y,
        "mask": mask,
        "x_in": X[mask].copy(),
        "y_in": Y[mask].copy(),
        "extent": (-box_halfwidth, box_halfwidth, -box_halfwidth, box_halfwidth),
        "weight": (h * h) * np.ones(int(mask.sum()), dtype=np.float64),
        "box_halfwidth": float(box_halfwidth),
        "radius": float(radius),
    }


def assemble_domain_mass_stiffness(Phi: np.ndarray, dPhix: np.ndarray, dPhiy: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = Phi.T @ (Phi * w[:, None])
    K = dPhix.T @ (dPhix * w[:, None]) + dPhiy.T @ (dPhiy * w[:, None])
    return M, K


# -----------------------------------------------------------------------------
# Pretrained ambient neural blocks
# -----------------------------------------------------------------------------

def exact_laplace_symbol_from_meta(meta: List[Tuple[str, int, int]], Lx: float, Ly: float) -> np.ndarray:
    """Strong-form Laplace symbol in the ambient trig basis."""
    lam = np.empty(len(meta), dtype=np.float64)
    sx = np.pi / float(Lx)
    sy = np.pi / float(Ly)
    for j, (_, k, l) in enumerate(meta):
        lam[j] = -((sx * k) ** 2 + (sy * l) ** 2)
    return lam


def _torch_tensor_to_numpy(x: Any) -> np.ndarray | None:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x, dtype=np.float64)
        except Exception:
            return None
    return None


def _softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 30.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def _torch_load_checkpoint(path: str) -> Any:
    """Load the user's local checkpoint.

    PyTorch 2.6 changed torch.load's default to weights_only=True.  The block
    training script stores numpy arrays such as lambda and sigma in the same
    checkpoint, so we explicitly set weights_only=False.  Use this only for
    trusted local checkpoints generated by train_laplace_block.py.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load a pretrained neural block, but torch is not importable.")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_tensor_by_keywords(obj: Any, keywords: Tuple[str, ...], max_depth: int = 5) -> Tuple[str, np.ndarray] | None:
    """Recursively search a checkpoint for a tensor whose key contains one keyword."""
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k).lower()
            if any(q in ks for q in keywords):
                arr = _torch_tensor_to_numpy(v)
                if arr is not None and arr.size > 0:
                    return str(k), arr
        for k, v in obj.items():
            ans = _find_tensor_by_keywords(v, keywords, max_depth=max_depth - 1)
            if ans is not None:
                key, arr = ans
                return f"{k}.{key}", arr
    return None


def _extract_scalar_from_dict(obj: Any, keys: Tuple[str, ...], max_depth: int = 4) -> float | None:
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys:
                arr = _torch_tensor_to_numpy(v)
                if arr is not None and arr.size == 1:
                    return float(arr.reshape(-1)[0])
                try:
                    return float(v)
                except Exception:
                    pass
        for v in obj.values():
            ans = _extract_scalar_from_dict(v, keys, max_depth=max_depth - 1)
            if ans is not None:
                return ans
    return None


def _map_array_to_meta(arr: np.ndarray, meta: List[Tuple[str, int, int]]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != len(meta):
            raise ValueError(f"Loaded vector has length {arr.size}, but current basis has M={len(meta)}.")
        return arr.copy()
    flat = arr.reshape(-1)
    if flat.size == len(meta):
        return flat.copy()
    raise ValueError(f"Loaded tensor shape {arr.shape} cannot be mapped to M={len(meta)} basis coefficients.")


def _current_cutoff_from_meta(meta: List[Tuple[str, int, int]]) -> int:
    return int(max(round(math.sqrt(k * k + ell * ell)) for _, k, ell in meta))


def load_learned_laplace_symbol(
    path: str,
    meta: List[Tuple[str, int, int]],
    Lx: float,
    Ly: float,
    sign_convention: str = "auto",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load the trained K=22 positive diagonal Laplace block.

    The training script saves
        model_state.raw_diag, model_state.scale, model_state.nonzero_mask,
    and the physical diagonal is
        c_theta = scale * softplus(raw_diag) * nonzero_mask.
    The strong-form Laplace block is therefore
        B_Delta(a) = -c_theta * a.
    """
    if not path:
        raise ValueError("Empty laplace block path.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Laplace block checkpoint not found: {path}")
    if sign_convention not in {"auto", "positive_stiffness", "laplace_symbol"}:
        raise ValueError("sign_convention must be auto, positive_stiffness, or laplace_symbol")

    ckpt = _torch_load_checkpoint(path)
    exact_lam = exact_laplace_symbol_from_meta(meta, Lx=Lx, Ly=Ly)
    current_K = _current_cutoff_from_meta(meta)

    ckpt_K = None
    ckpt_M = None
    if isinstance(ckpt, dict):
        if "K" in ckpt:
            try:
                ckpt_K = int(ckpt["K"])
            except Exception:
                ckpt_K = None
        if "M" in ckpt:
            try:
                ckpt_M = int(ckpt["M"])
            except Exception:
                ckpt_M = None
    if ckpt_K is not None and ckpt_K != current_K:
        raise ValueError(
            f"Checkpoint was trained with K={ckpt_K}, but this rollout uses K={current_K}. "
            f"Use --K {ckpt_K}, or retrain the Laplace block at K={current_K}."
        )
    if ckpt_M is not None and ckpt_M != len(meta):
        raise ValueError(
            f"Checkpoint has M={ckpt_M}, but current basis has M={len(meta)}. "
            "The rollout basis must match the trained block basis."
        )

    source_key = None
    source_kind = None
    vals = None
    scale = None
    mask = None

    state = None
    if isinstance(ckpt, dict):
        maybe_state = ckpt.get("model_state", None)
        if isinstance(maybe_state, dict):
            state = maybe_state
        elif all(isinstance(k, str) for k in ckpt.keys()) and "raw_diag" in ckpt:
            state = ckpt

    if isinstance(state, dict) and "raw_diag" in state:
        raw = _map_array_to_meta(_torch_tensor_to_numpy(state["raw_diag"]), meta)
        scale_arr = _torch_tensor_to_numpy(state.get("scale", None)) if "scale" in state else None
        if scale_arr is not None and scale_arr.size == 1:
            scale = float(scale_arr.reshape(-1)[0])
        else:
            scale = _extract_scalar_from_dict(ckpt, ("scale",))
        if scale is None:
            lam_top = _torch_tensor_to_numpy(ckpt.get("lambda", None)) if isinstance(ckpt, dict) else None
            scale = float(np.max(lam_top)) if lam_top is not None else float(np.max(np.abs(exact_lam)))
        mask_arr = _torch_tensor_to_numpy(state.get("nonzero_mask", None)) if "nonzero_mask" in state else None
        if mask_arr is not None:
            mask = _map_array_to_meta(mask_arr, meta)
        else:
            mask = (np.abs(exact_lam) > 0.0).astype(np.float64)
        c = float(scale) * _softplus_np(raw) * mask
        vals = c
        source_key = "model_state.raw_diag"
        source_kind = "positive_stiffness_from_raw_diag"
    else:
        coeff_keys = ("physical_diag", "learned_diag", "diag", "stiffness", "coeff", "c")
        symbol_keys = ("lambda_learn", "laplace_symbol", "symbol", "eig", "eigen", "lambda")
        found_coeff = _find_tensor_by_keywords(ckpt, coeff_keys)
        found_symbol = _find_tensor_by_keywords(ckpt, symbol_keys)
        if found_coeff is not None:
            source_key, raw = found_coeff
            vals = _map_array_to_meta(raw, meta)
            source_kind = "coefficient_or_stiffness"
        elif found_symbol is not None:
            source_key, raw = found_symbol
            vals = _map_array_to_meta(raw, meta)
            source_kind = "symbol_or_exact_lambda"
        else:
            raise KeyError(
                "Could not find model_state.raw_diag or a Laplace diagonal tensor in the checkpoint."
            )
        scale = _extract_scalar_from_dict(ckpt, ("scale", "laplace_scale", "target_scale")) or 1.0

    vals = np.asarray(vals, dtype=np.float64)
    max_exact = float(np.max(np.abs(exact_lam))) + 1e-14
    max_abs = float(np.max(np.abs(vals))) + 1e-14

    if sign_convention == "laplace_symbol":
        lam = vals.copy()
        if scale is not None and float(scale) != 1.0 and max_abs <= 2.5 and max_exact > 10.0:
            lam = lam * float(scale)
        inferred = "laplace_symbol"
    elif sign_convention == "positive_stiffness":
        c = vals.copy()
        if scale is not None and float(scale) != 1.0 and max_abs <= 2.5 and max_exact > 10.0:
            c = c * float(scale)
        lam = -np.abs(c)
        inferred = "positive_stiffness"
    else:
        # The trained block stores positive stiffness c_theta.  Negative tensors are
        # treated as already being strong-form Laplace symbols.
        nonzero = vals[np.abs(vals) > 1e-14]
        median_val = float(np.median(nonzero)) if nonzero.size else 0.0
        if median_val < 0.0:
            lam = vals.copy()
            inferred = "laplace_symbol"
        else:
            c = vals.copy()
            if source_kind != "positive_stiffness_from_raw_diag" and scale is not None and float(scale) != 1.0 and max_abs <= 2.5 and max_exact > 10.0:
                c = c * float(scale)
            lam = -np.abs(c)
            inferred = "positive_stiffness"

    for j, (_, k, ell) in enumerate(meta):
        if k == 0 and ell == 0:
            lam[j] = 0.0

    rel_to_exact = float(np.linalg.norm(lam - exact_lam) / (np.linalg.norm(exact_lam) + 1e-14))
    info = {
        "path": path,
        "source_key": source_key,
        "source_kind": source_kind,
        "auto_inferred": inferred,
        "checkpoint_K": ckpt_K,
        "checkpoint_M": ckpt_M,
        "current_M": len(meta),
        "scale": float(scale) if scale is not None else None,
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "lambda_abs_max": float(np.max(np.abs(lam))),
        "rel_to_exact_symbol": rel_to_exact,
    }
    return lam.astype(np.float64), info


def build_reduced_laplace_mass_from_symbol(N: np.ndarray, M_dom: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Return L_r = N^T M_Omega diag(lambda_vec) N for z_t = eps2 M_r^{-1} L_r z."""
    if lambda_vec.shape[0] != N.shape[0]:
        raise ValueError(f"lambda_vec length {lambda_vec.shape[0]} does not match ambient dimension {N.shape[0]}.")
    AN = lambda_vec[:, None] * N
    return N.T @ (M_dom @ AN)


# -----------------------------------------------------------------------------
# Section 3 tangent space
# -----------------------------------------------------------------------------

def tangent_space_from_boundary(C_np: np.ndarray, tau_rel: float) -> Dict[str, Any]:
    _, S, Vh = np.linalg.svd(C_np, full_matrices=True)
    tau = tau_rel * np.max(S)
    r = int(np.sum(S > tau))
    N = Vh[r:, :].T.copy()
    return {"S": S, "tau": float(tau), "rank": r, "null_dim": int(N.shape[1]), "N": N}


def robust_m_orthonormalize(N_raw: np.ndarray, M_dom: np.ndarray, max_rank: int | None, eig_rel_tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    G = 0.5 * (N_raw.T @ M_dom @ N_raw + (N_raw.T @ M_dom @ N_raw).T)
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    keep = evals > eig_rel_tol * max(float(evals[0]), 1.0)
    if max_rank is not None and max_rank > 0:
        idx = np.where(keep)[0][:max_rank]
        mask = np.zeros_like(keep, dtype=bool)
        mask[idx] = True
        keep = mask
    evals_keep = evals[keep]
    evecs_keep = evecs[:, keep]
    T = evecs_keep / np.sqrt(evals_keep)[None, :]
    N = N_raw @ T
    return N, evals_keep


def build_section3_basis(C_build: np.ndarray, M_dom: np.ndarray, tau_rel: float, reduced_rank: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    sec3 = tangent_space_from_boundary(C_build, tau_rel=tau_rel)
    N_raw = sec3["N"]
    max_rank = None if reduced_rank <= 0 else reduced_rank
    N, kept = robust_m_orthonormalize(N_raw, M_dom, max_rank=max_rank)
    Mr = N.T @ M_dom @ N
    mass_eigs = np.linalg.eigvalsh(0.5 * (Mr + Mr.T))
    info = {
        "rank_C": int(sec3["rank"]),
        "null_dim": int(sec3["null_dim"]),
        "tau": float(sec3["tau"]),
        "sv_head": sec3["S"][:8].tolist(),
        "final_rank": int(N.shape[1]),
        "kept_mass_eigs": [float(kept.min()), float(kept.max())] if kept.size else [0.0, 0.0],
        "mass_eigs": [float(mass_eigs.min()), float(mass_eigs.max())],
    }
    return N, info


def weighted_project_to_reduced(values: np.ndarray, Phi_r: np.ndarray, w: np.ndarray, lam: float = 1e-12) -> np.ndarray:
    A = Phi_r.T @ (Phi_r * w[:, None]) + lam * np.eye(Phi_r.shape[1])
    b = Phi_r.T @ (w * values)
    return np.linalg.solve(A, b)


# -----------------------------------------------------------------------------
# Direct reduced-space z0 construction
# -----------------------------------------------------------------------------

def generalized_smooth_modes(Kr: np.ndarray, Mr: np.ndarray, nmodes: int) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = la.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    order = np.argsort(evals)
    evals = evals[order][:nmodes]
    Q = evecs[:, order][:, :nmodes]
    return evals, Q


def construct_reduced_space_interface_z0(
    Phi_r: np.ndarray,
    Mr: np.ndarray,
    Kr: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    w: np.ndarray,
    ic_amp: float,
    bend: float,
    nmodes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    _, Q = generalized_smooth_modes(Kr, Mr, nmodes=max(nmodes, 12))
    fields = [Phi_r @ Q[:, j] for j in range(Q.shape[1])]
    xref = X[mask].copy()
    y2ref = (Y[mask] * Y[mask] - np.mean(Y[mask] * Y[mask])).copy()
    corr_x = []
    corr_y2 = []
    for uj in fields:
        ujc = uj - weighted_mean(uj, w)
        nx = np.linalg.norm(ujc) * np.linalg.norm(xref) + 1e-14
        ny = np.linalg.norm(ujc) * np.linalg.norm(y2ref) + 1e-14
        corr_x.append(abs(float(np.dot(ujc, xref) / nx)))
        corr_y2.append(abs(float(np.dot(ujc, y2ref) / ny)))
    search_ids = list(range(1, min(Q.shape[1], 10)))
    idx_x = max(search_ids, key=lambda j: corr_x[j])
    idx_y2 = max(search_ids, key=lambda j: corr_y2[j] if j != idx_x else -1.0)
    z = 1.0 * Q[:, idx_x] - bend * Q[:, idx_y2]
    for j in range(1, min(Q.shape[1], 8)):
        if j not in (idx_x, idx_y2):
            z += 0.08 * rng.standard_normal() * Q[:, j]
    u = Phi_r @ z
    u = u - weighted_mean(u, w)
    cur = float(np.max(np.abs(u)))
    if cur > 1e-14:
        z *= (ic_amp / cur)
        u = Phi_r @ z
    u = u - weighted_mean(u, w)
    z = weighted_project_to_reduced(u, Phi_r, w)
    u = Phi_r @ z
    cur = float(np.max(np.abs(u)))
    if cur > 1e-14:
        z *= (ic_amp / cur)
        u = Phi_r @ z
    u = u - weighted_mean(u, w)
    z = weighted_project_to_reduced(u, Phi_r, w)
    u = Phi_r @ z
    info = {
        "idx_x_mode": int(idx_x),
        "idx_y2_mode": int(idx_y2),
        "mean_u0": weighted_mean(u, w),
        "u0_min": float(np.min(u)),
        "u0_max": float(np.max(u)),
        "corr_x": float(corr_x[idx_x]),
        "corr_y2": float(corr_y2[idx_y2]),
    }
    return z, u, info


# -----------------------------------------------------------------------------
# Volume-constrained reaction flow
# -----------------------------------------------------------------------------

def volume_constrained_reaction_exact(u: np.ndarray, tau: float, w: np.ndarray) -> np.ndarray:
    if abs(tau) < 1e-16:
        return u.copy()
    mean0 = weighted_mean(u, w)

    def reaction_exact_pointwise(v: np.ndarray) -> np.ndarray:
        e = math.exp(-2.0 * tau)
        den = np.maximum(v * v + (1.0 - v * v) * e, 1e-14)
        return v / np.sqrt(den)

    def F(c: float) -> float:
        v = reaction_exact_pointwise(u + c)
        return weighted_mean(v, w) - mean0

    a, b = -4.0, 4.0
    Fa, Fb = F(a), F(b)
    n_expand = 0
    while Fa * Fb > 0.0 and n_expand < 20:
        a *= 1.5
        b *= 1.5
        Fa, Fb = F(a), F(b)
        n_expand += 1
    if Fa * Fb > 0.0:
        return reaction_exact_pointwise(u)
    c_star = brentq(F, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
    return reaction_exact_pointwise(u + c_star)


# -----------------------------------------------------------------------------
# FEM
# -----------------------------------------------------------------------------

def generate_circle_mesh(radius: float, n_boundary: int, n_interior: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    xb, yb, _, _ = circle_boundary_points(n_boundary, radius=radius)
    pb = np.column_stack([xb, yb])
    rho = radius * np.sqrt(rng.random(n_interior))
    theta = 2.0 * np.pi * rng.random(n_interior)
    pi = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    points = np.vstack([pb, pi])
    tri = Delaunay(points)
    simplices = tri.simplices.copy()
    centroids = points[simplices].mean(axis=1)
    keep = np.sum(centroids ** 2, axis=1) <= radius * radius + 1e-12
    simplices = simplices[keep]
    return {"points": points, "tris": simplices}


def assemble_p1_mk(points: np.ndarray, tris: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    n = points.shape[0]
    rows_M: List[int] = []
    cols_M: List[int] = []
    data_M: List[float] = []
    rows_K: List[int] = []
    cols_K: List[int] = []
    data_K: List[float] = []
    for tri in tris:
        verts = points[tri]
        x1, y1 = verts[0]
        x2, y2 = verts[1]
        x3, y3 = verts[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area <= 1e-15:
            continue
        Mloc = (area / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64)
        Kloc = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for a in range(3):
            ia = int(tri[a])
            for bb in range(3):
                ib = int(tri[bb])
                rows_M.append(ia); cols_M.append(ib); data_M.append(float(Mloc[a, bb]))
                rows_K.append(ia); cols_K.append(ib); data_K.append(float(Kloc[a, bb]))
    M = sp.coo_matrix((data_M, (rows_M, cols_M)), shape=(n, n)).tocsr()
    K = sp.coo_matrix((data_K, (rows_K, cols_K)), shape=(n, n)).tocsr()
    return M, K


def fem_interpolate_to_grid(points: np.ndarray, tris: np.ndarray, u: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    tri = Delaunay(points)
    pts = np.column_stack([xq, yq])
    simplices = tri.find_simplex(pts)
    bad = simplices < 0
    simplices_safe = simplices.copy()
    simplices_safe[bad] = 0
    X = tri.transform[simplices_safe, :2]
    Y = pts - tri.transform[simplices_safe, 2]
    bary12 = np.einsum("nij,nj->ni", X, Y)
    bary = np.column_stack([bary12, 1.0 - bary12.sum(axis=1)])
    verts = tri.simplices[simplices_safe]
    vals = np.einsum("ni,ni->n", u[verts], bary)
    vals[bad] = 0.0
    return vals


def fem_l2_project_initial_condition(mesh: Dict[str, Any], u0_cont: np.ndarray, grid_pack: Dict[str, Any]) -> np.ndarray:
    points = mesh["points"]
    tris = mesh["tris"]
    M, _ = assemble_p1_mk(points, tris)
    x_in = grid_pack["x_in"]
    y_in = grid_pack["y_in"]
    triq = Delaunay(np.column_stack([x_in, y_in]))
    centroids = points[tris].mean(axis=1)
    simplices = triq.find_simplex(centroids)
    bad = simplices < 0
    simplices_safe = simplices.copy(); simplices_safe[bad] = 0
    X = triq.transform[simplices_safe, :2]
    Y = centroids - triq.transform[simplices_safe, 2]
    bary12 = np.einsum("nij,nj->ni", X, Y)
    bary = np.column_stack([bary12, 1.0 - bary12.sum(axis=1)])
    verts = triq.simplices[simplices_safe]
    uc = np.einsum("ni,ni->n", u0_cont[verts], bary)
    uc[bad] = 0.0
    b = np.zeros(points.shape[0], dtype=np.float64)
    for t_idx, tri in enumerate(tris):
        verts_pts = points[tri]
        x1, y1 = verts_pts[0]; x2, y2 = verts_pts[1]; x3, y3 = verts_pts[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area <= 1e-15:
            continue
        b[tri] += uc[t_idx] * area / 3.0
    solve_M = spla.factorized(M.tocsc())
    return solve_M(b)


def fem_energy(M: sp.csr_matrix, K: sp.csr_matrix, u: np.ndarray, eps2: float) -> float:
    bulk = 0.25 * (u * u - 1.0) ** 2
    return float(0.5 * eps2 * (u @ (K @ u)) + np.sum((M @ bulk)))


def fem_strang_rollout_sampled(
    mesh: Dict[str, Any],
    u0_nodes: np.ndarray,
    eps2: float,
    dt: float,
    T: float,
    sample_steps: set[int],
) -> Dict[str, Any]:
    points = mesh["points"]
    tris = mesh["tris"]
    M, K = assemble_p1_mk(points, tris)
    nsteps = int(round(T / dt))
    A_l = (M + 0.5 * dt * eps2 * K).tocsc()
    A_r = (M - 0.5 * dt * eps2 * K).tocsr()
    solve_cn = spla.factorized(A_l)
    w_fem = np.asarray(M.sum(axis=1)).ravel()
    u = u0_nodes.copy()
    node_samples: Dict[int, np.ndarray] = {}
    energy_samples: Dict[int, float] = {}
    mean_samples: Dict[int, float] = {}
    if 0 in sample_steps:
        node_samples[0] = u.copy()
        energy_samples[0] = fem_energy(M, K, u, eps2)
        mean_samples[0] = weighted_mean(u, w_fem)
    for k in range(nsteps):
        u = volume_constrained_reaction_exact(u, 0.5 * dt, w_fem)
        u = solve_cn(A_r @ u)
        u = volume_constrained_reaction_exact(u, 0.5 * dt, w_fem)
        s = k + 1
        if s in sample_steps:
            node_samples[s] = u.copy()
            energy_samples[s] = fem_energy(M, K, u, eps2)
            mean_samples[s] = weighted_mean(u, w_fem)
    return {
        "points": points,
        "tris": tris,
        "node_samples": node_samples,
        "energy_samples": energy_samples,
        "mean_samples": mean_samples,
    }


# -----------------------------------------------------------------------------
# Reduced rollout
# -----------------------------------------------------------------------------

def reduced_energy(z: np.ndarray, Phi_r: np.ndarray, w: np.ndarray, K_r: np.ndarray, eps2: float) -> float:
    u = Phi_r @ z
    bulk = 0.25 * (u * u - 1.0) ** 2
    return float(0.5 * eps2 * (z @ (K_r @ z)) + np.sum(w * bulk))


def build_diffusion_cn_solver(Mr: np.ndarray, Kr: np.ndarray, eps2: float, dt: float) -> Tuple[np.ndarray, spla.SuperLU]:
    # Exact Galerkin fallback: z_t = -eps2 * Mr^{-1} Kr z.
    Lr_mass = -Kr
    return build_linear_cn_solver_from_mass_operator(Mr, Lr_mass, eps2, dt)


def build_linear_cn_solver_from_mass_operator(Mr: np.ndarray, Lr_mass: np.ndarray, eps2: float, dt: float) -> Tuple[np.ndarray, spla.SuperLU]:
    """Crank--Nicolson for Mr z_t = eps2 * Lr_mass z."""
    A_l = sp.csc_matrix(Mr - 0.5 * dt * eps2 * Lr_mass)
    A_r = Mr + 0.5 * dt * eps2 * Lr_mass
    solve = spla.splu(A_l)
    return A_r, solve


def build_dense_spd_solver(A: np.ndarray) -> Tuple[np.ndarray, bool]:
    A_sym = 0.5 * (A + A.T)
    c, lower = la.cho_factor(A_sym, lower=True, check_finite=False)
    return c, lower


def solve_dense_spd(fact: Tuple[np.ndarray, bool], b: np.ndarray) -> np.ndarray:
    return la.cho_solve(fact, b, check_finite=False)


@dataclass
class ReducedReactionBlock:
    Phi_r: np.ndarray
    Mr_fact: Tuple[np.ndarray, bool]
    quad_weights: np.ndarray
    mean_weights: np.ndarray


@dataclass
class ReducedReactionOrthoBlock:
    Phi_r: np.ndarray
    quad_weights: np.ndarray
    mean_weights: np.ndarray


@dataclass
class ReducedPointwiseProjectReactionBlock:
    Phi_r: np.ndarray
    quad_weights: np.ndarray
    projector: np.ndarray


@dataclass
class AmbientGalerkinReactionBlock:
    Phi: np.ndarray
    N: np.ndarray
    quad_weights: np.ndarray
    mean_weights: np.ndarray
    z_from_a: np.ndarray
    projector: np.ndarray


def build_reduced_reaction_block(Phi_r: np.ndarray, w: np.ndarray, Mr: np.ndarray) -> ReducedReactionBlock:
    quad_weights = np.asarray(w, dtype=np.float64).copy()
    mean_weights = quad_weights.copy()
    mean_weights /= max(float(np.sum(mean_weights)), 1e-14)
    Mr_fact = build_dense_spd_solver(Mr)
    return ReducedReactionBlock(
        Phi_r=np.asarray(Phi_r, dtype=np.float64),
        Mr_fact=Mr_fact,
        quad_weights=quad_weights,
        mean_weights=mean_weights,
    )


def build_reduced_reaction_ortho_block(Phi_r: np.ndarray, w: np.ndarray) -> ReducedReactionOrthoBlock:
    quad_weights = np.asarray(w, dtype=np.float64).copy()
    mean_weights = quad_weights.copy()
    mean_weights /= max(float(np.sum(mean_weights)), 1e-14)
    return ReducedReactionOrthoBlock(
        Phi_r=np.asarray(Phi_r, dtype=np.float64),
        quad_weights=quad_weights,
        mean_weights=mean_weights,
    )


def build_reduced_pointwise_project_reaction_block(
    Phi_r: np.ndarray,
    w: np.ndarray,
    project_reg: float = 1e-12,
) -> ReducedPointwiseProjectReactionBlock:
    quad_weights = np.asarray(w, dtype=np.float64).copy()
    A = Phi_r.T @ (Phi_r * quad_weights[:, None]) + float(project_reg) * np.eye(Phi_r.shape[1], dtype=np.float64)
    A_fact = build_dense_spd_solver(A)
    projector = solve_dense_spd(A_fact, Phi_r.T * quad_weights[None, :])
    return ReducedPointwiseProjectReactionBlock(
        Phi_r=np.asarray(Phi_r, dtype=np.float64),
        quad_weights=quad_weights,
        projector=np.asarray(projector, dtype=np.float64),
    )


def build_ambient_galerkin_reaction_block(
    Phi: np.ndarray,
    M_dom: np.ndarray,
    N: np.ndarray,
    w: np.ndarray,
    Mr: np.ndarray,
    reg: float = 1e-10,
) -> AmbientGalerkinReactionBlock:
    quad_weights = np.asarray(w, dtype=np.float64).copy()
    mean_weights = quad_weights.copy()
    mean_weights /= max(float(np.sum(mean_weights)), 1e-14)
    Mdom_fact = build_dense_spd_solver(M_dom + float(reg) * np.eye(M_dom.shape[0], dtype=np.float64))
    Mr_fact = build_dense_spd_solver(Mr)
    z_from_a = solve_dense_spd(Mr_fact, N.T @ M_dom)
    projector = solve_dense_spd(Mdom_fact, Phi.T * quad_weights[None, :])
    return AmbientGalerkinReactionBlock(
        Phi=np.asarray(Phi, dtype=np.float64),
        N=np.asarray(N, dtype=np.float64),
        quad_weights=quad_weights,
        mean_weights=mean_weights,
        z_from_a=np.asarray(z_from_a, dtype=np.float64),
        projector=np.asarray(projector, dtype=np.float64),
    )


def reduced_reaction_rhs_block(z: np.ndarray, block: ReducedReactionBlock) -> np.ndarray:
    u = block.Phi_r @ z
    reac = u - u * u * u
    lam = float(block.mean_weights @ reac)
    rhs_field = reac - lam
    rhs_red = block.Phi_r.T @ (rhs_field * block.quad_weights)
    return solve_dense_spd(block.Mr_fact, rhs_red)


def reduced_reaction_rhs_ortho_block(z: np.ndarray, block: ReducedReactionOrthoBlock) -> np.ndarray:
    u = block.Phi_r @ z
    reac = u - u * u * u
    lam = float(block.mean_weights @ reac)
    rhs_field = reac - lam
    return block.Phi_r.T @ (rhs_field * block.quad_weights)


def reduced_reaction_exact_pointwise_project_block(
    z: np.ndarray,
    tau: float,
    block: ReducedPointwiseProjectReactionBlock,
) -> np.ndarray:
    if abs(tau) < 1e-16:
        return z.copy()
    u = block.Phi_r @ z
    u_next = volume_constrained_reaction_exact(u, tau, block.quad_weights)
    return block.projector @ u_next


def reduced_reaction_rhs(z: np.ndarray, Phi_r: np.ndarray, w: np.ndarray, Mr_fact: Tuple[np.ndarray, bool]) -> np.ndarray:
    u = Phi_r @ z
    reac = u - u * u * u
    lam = weighted_mean(reac, w)
    rhs_field = reac - lam
    rhs_red = Phi_r.T @ (w * rhs_field)
    return solve_dense_spd(Mr_fact, rhs_red)


def ambient_galerkin_reaction_rhs_a(a: np.ndarray, block: AmbientGalerkinReactionBlock) -> np.ndarray:
    u = block.Phi @ a
    reac = u - u * u * u
    lam = float(block.mean_weights @ reac)
    rhs_field = reac - lam
    return block.projector @ rhs_field


def ambient_galerkin_reaction_rk4_from_z(
    z: np.ndarray,
    tau: float,
    block: AmbientGalerkinReactionBlock,
    nsub: int,
) -> np.ndarray:
    if abs(tau) < 1e-16:
        return z.copy()
    nsub = max(int(nsub), 1)
    h = tau / float(nsub)
    acur = block.N @ z
    for _ in range(nsub):
        k1 = ambient_galerkin_reaction_rhs_a(acur, block)
        k2 = ambient_galerkin_reaction_rhs_a(acur + 0.5 * h * k1, block)
        k3 = ambient_galerkin_reaction_rhs_a(acur + 0.5 * h * k2, block)
        k4 = ambient_galerkin_reaction_rhs_a(acur + h * k3, block)
        acur = acur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return block.z_from_a @ acur


def reduced_reaction_rk4_block(z: np.ndarray, tau: float, block: ReducedReactionBlock, nsub: int) -> np.ndarray:
    if abs(tau) < 1e-16:
        return z.copy()
    nsub = max(int(nsub), 1)
    h = tau / float(nsub)
    zcur = z.copy()
    for _ in range(nsub):
        k1 = reduced_reaction_rhs_block(zcur, block)
        k2 = reduced_reaction_rhs_block(zcur + 0.5 * h * k1, block)
        k3 = reduced_reaction_rhs_block(zcur + 0.5 * h * k2, block)
        k4 = reduced_reaction_rhs_block(zcur + h * k3, block)
        zcur = zcur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return zcur


def reduced_reaction_rk4(z: np.ndarray, tau: float, Phi_r: np.ndarray, w: np.ndarray, Mr_fact: Tuple[np.ndarray, bool], nsub: int) -> np.ndarray:
    if abs(tau) < 1e-16:
        return z.copy()
    nsub = max(int(nsub), 1)
    h = tau / float(nsub)
    zcur = z.copy()
    for _ in range(nsub):
        k1 = reduced_reaction_rhs(zcur, Phi_r, w, Mr_fact)
        k2 = reduced_reaction_rhs(zcur + 0.5 * h * k1, Phi_r, w, Mr_fact)
        k3 = reduced_reaction_rhs(zcur + 0.5 * h * k2, Phi_r, w, Mr_fact)
        k4 = reduced_reaction_rhs(zcur + h * k3, Phi_r, w, Mr_fact)
        zcur = zcur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return zcur


def reduced_reaction_rk4_ortho_block(z: np.ndarray, tau: float, block: ReducedReactionOrthoBlock, nsub: int) -> np.ndarray:
    if abs(tau) < 1e-16:
        return z.copy()
    nsub = max(int(nsub), 1)
    h = tau / float(nsub)
    zcur = z.copy()
    for _ in range(nsub):
        k1 = reduced_reaction_rhs_ortho_block(zcur, block)
        k2 = reduced_reaction_rhs_ortho_block(zcur + 0.5 * h * k1, block)
        k3 = reduced_reaction_rhs_ortho_block(zcur + 0.5 * h * k2, block)
        k4 = reduced_reaction_rhs_ortho_block(zcur + h * k3, block)
        zcur = zcur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return zcur


def reduced_strang_step(z: np.ndarray, dt: float, Phi_r: np.ndarray, w: np.ndarray, Mr_fact: Tuple[np.ndarray, bool], react_substeps: int, A_r_diff: np.ndarray, solve_diff: spla.SuperLU) -> np.ndarray:
    z = reduced_reaction_rk4(z, 0.5 * dt, Phi_r, w, Mr_fact, react_substeps)
    z = solve_diff.solve(A_r_diff @ z)
    z = reduced_reaction_rk4(z, 0.5 * dt, Phi_r, w, Mr_fact, react_substeps)
    return z


def rollout_reduced_method_sampled(
    N: np.ndarray,
    Phi: np.ndarray,
    K_dom: np.ndarray,
    M_dom: np.ndarray,
    C_dense: np.ndarray,
    grid_pack: Dict[str, Any],
    z0: np.ndarray,
    eps2: float,
    dt: float,
    T: float,
    sample_steps: set[int],
    log_every: int,
    react_substeps: int,
    fem_grid_samples: Dict[int, np.ndarray],
    laplace_mass_op_r: np.ndarray | None = None,
    laplace_label: str = "exact Galerkin",
) -> Dict[str, Any]:
    w = grid_pack["weight"]
    Phi_r = Phi @ N
    Mr = N.T @ M_dom @ N
    Kr = N.T @ K_dom @ N  # used for physical energy diagnostics
    if laplace_mass_op_r is None:
        A_r_diff, solve_diff = build_diffusion_cn_solver(Mr, Kr, eps2, dt)
    else:
        A_r_diff, solve_diff = build_linear_cn_solver_from_mass_operator(Mr, laplace_mass_op_r, eps2, dt)
    reaction_block = build_reduced_reaction_block(Phi_r, w, Mr)
    print(f"[diffusion block] using {laplace_label}")
    z = z0.copy()
    nsteps = int(round(T / dt))

    field_samples: Dict[int, np.ndarray] = {}
    energy_samples: Dict[int, float] = {}
    mean_samples: Dict[int, float] = {}
    bc_samples: Dict[int, float] = {}
    rel_samples: Dict[int, float] = {}

    u0 = Phi_r @ z
    a0 = N @ z
    if 0 in sample_steps:
        field_samples[0] = u0.copy()
        energy_samples[0] = reduced_energy(z, Phi_r, w, Kr, eps2)
        mean_samples[0] = weighted_mean(u0, w)
        bc_samples[0] = bc_violation(a0, C_dense)
        rel_samples[0] = weighted_rel_l2(u0, fem_grid_samples[0], w)

    print(f"[Section3 IC] reduced-space initialization error = 0 by construction | dense_bc={bc_samples[0]:.3e} | mean={mean_samples[0]:+.3e}")
    print(f"[Section3] step={0:05d}/{nsteps:05d} t={0.0:.3f} | relL2_vs_fem={rel_samples[0]:.3e} | bc={bc_samples[0]:.3e} | mean={mean_samples[0]:+.3e} | E={energy_samples[0]:.6e}")

    for k in range(nsteps):
        z = reduced_reaction_rk4_block(z, 0.5 * dt, reaction_block, react_substeps)
        z = solve_diff.solve(A_r_diff @ z)
        z = reduced_reaction_rk4_block(z, 0.5 * dt, reaction_block, react_substeps)
        s = k + 1
        if s in sample_steps:
            u = Phi_r @ z
            a = N @ z
            field_samples[s] = u.copy()
            energy_samples[s] = reduced_energy(z, Phi_r, w, Kr, eps2)
            mean_samples[s] = weighted_mean(u, w)
            bc_samples[s] = bc_violation(a, C_dense)
            rel_samples[s] = weighted_rel_l2(u, fem_grid_samples[s], w)
            if (s % log_every == 0) or (s == nsteps):
                print(
                    f"[Section3] step={s:05d}/{nsteps:05d} t={s*dt:.3f} | relL2_vs_fem={rel_samples[s]:.3e} | "
                    f"bc={bc_samples[s]:.3e} | mean={mean_samples[s]:+.3e} | E={energy_samples[s]:.6e}"
                )
    return {
        "field_samples": field_samples,
        "energy_samples": energy_samples,
        "mean_samples": mean_samples,
        "bc_samples": bc_samples,
        "rel_samples": rel_samples,
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def lift_to_grid(grid_pack: Dict[str, Any], v: np.ndarray) -> np.ndarray:
    Z = np.full_like(grid_pack["X"], np.nan, dtype=np.float64)
    Z[grid_pack["mask"]] = v
    return Z


def draw_disk_outline(ax: Any, radius: float) -> None:
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(radius * np.cos(th), radius * np.sin(th), "k--", lw=0.8)


def save_curve(x: np.ndarray, ys: List[Tuple[str, np.ndarray]], title: str, ylabel: str, outpath: str, semilogy: bool = False) -> None:
    plt.figure(figsize=(7, 4.5))
    for label, y in ys:
        if semilogy:
            plt.semilogy(x, y, label=label)
        else:
            plt.plot(x, y, label=label)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()


def save_snapshot_compare(grid_pack: Dict[str, Any], snaps_a: Dict[int, np.ndarray], snaps_b: Dict[int, np.ndarray], label_a: str, label_b: str, snap_times: Dict[int, float], outpath: str) -> None:
    ids = sorted(snaps_a.keys())
    ncols = len(ids)
    fig, axes = plt.subplots(3, ncols, figsize=(3.8 * ncols, 8.8), constrained_layout=True)
    box_halfwidth = grid_pack["box_halfwidth"]
    radius = grid_pack["radius"]
    for j, k in enumerate(ids):
        Za = lift_to_grid(grid_pack, snaps_a[k])
        Zb = lift_to_grid(grid_pack, snaps_b[k])
        Ze = lift_to_grid(grid_pack, snaps_a[k] - snaps_b[k])
        im0 = axes[0, j].imshow(Za.T, origin="lower", extent=grid_pack["extent"], aspect="equal", vmin=-1.0, vmax=1.0)
        axes[0, j].set_title(f"{label_a}\nstep={k}, t={snap_times[k]:.3f}")
        axes[0, j].set_xlim(-box_halfwidth, box_halfwidth); axes[0, j].set_ylim(-box_halfwidth, box_halfwidth)
        draw_disk_outline(axes[0, j], radius)
        plt.colorbar(im0, ax=axes[0, j], fraction=0.046)
        im1 = axes[1, j].imshow(Zb.T, origin="lower", extent=grid_pack["extent"], aspect="equal", vmin=-1.0, vmax=1.0)
        axes[1, j].set_title(label_b)
        axes[1, j].set_xlim(-box_halfwidth, box_halfwidth); axes[1, j].set_ylim(-box_halfwidth, box_halfwidth)
        draw_disk_outline(axes[1, j], radius)
        plt.colorbar(im1, ax=axes[1, j], fraction=0.046)
        im2 = axes[2, j].imshow(Ze.T, origin="lower", extent=grid_pack["extent"], aspect="equal", cmap="coolwarm")
        axes[2, j].set_title("Pointwise error")
        axes[2, j].set_xlim(-box_halfwidth, box_halfwidth); axes[2, j].set_ylim(-box_halfwidth, box_halfwidth)
        draw_disk_outline(axes[2, j], radius)
        plt.colorbar(im2, ax=axes[2, j], fraction=0.046)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def save_disk_geometry(grid_pack: Dict[str, Any], Nb: int, outpath: str) -> None:
    radius = grid_pack["radius"]
    box_halfwidth = grid_pack["box_halfwidth"]
    xb, yb, _, _ = circle_boundary_points(Nb, radius)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([ -box_halfwidth,  box_halfwidth,  box_halfwidth, -box_halfwidth, -box_halfwidth],
            [ -box_halfwidth, -box_halfwidth,  box_halfwidth,  box_halfwidth, -box_halfwidth], 'k-', lw=1.2, label='ambient box')
    ax.plot(xb, yb, 'b-', lw=2, label=f'disk boundary, diameter={2*radius:.3f}')
    ax.scatter(xb[::max(1, Nb // 200)], yb[::max(1, Nb // 200)], s=8, c='red', label='boundary samples')
    ax.set_xlim(-box_halfwidth, box_halfwidth)
    ax.set_ylim(-box_halfwidth, box_halfwidth)
    ax.set_aspect('equal')
    ax.set_title('Ambient box and embedded disk')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

@dataclass
class Args:
    outdir: str
    box_halfwidth: float
    disk_diameter: float
    K: int
    Nb_build: int
    Nb_dense: int
    tau_rel: float
    reduced_rank: int
    Nx_eval: int
    eps2: float
    T: float
    dt: float
    ic_amp: float
    interface_bend: float
    smooth_mode_count: int
    react_substeps: int
    fem_boundary_nodes: int
    fem_interior_nodes: int
    seed: int
    log_every: int
    use_learned_laplace: int
    laplace_block_path: str
    laplace_block_sign: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description='Section-3 Allen--Cahn on a disk embedded in a fixed box.')
    p.add_argument('--outdir', type=str, required=True)
    p.add_argument('--box_halfwidth', type=float, required=True)
    p.add_argument('--disk_diameter', type=float, required=True)
    p.add_argument('--K', type=int, required=True)
    p.add_argument('--Nb_build', type=int, required=True)
    p.add_argument('--Nb_dense', type=int, required=True)
    p.add_argument('--tau_rel', type=float, default=1e-10)
    p.add_argument('--reduced_rank', type=int, required=True,
                   help='Reduced tangent-space rank. Use <=0 to keep the full approximate null space.')
    p.add_argument('--Nx_eval', type=int, required=True)
    p.add_argument('--eps2', type=float, required=True)
    p.add_argument('--T', type=float, required=True)
    p.add_argument('--dt', type=float, required=True)
    p.add_argument('--ic_amp', type=float, default=0.70)
    p.add_argument('--interface_bend', type=float, default=0.28)
    p.add_argument('--smooth_mode_count', type=int, default=18)
    p.add_argument('--react_substeps', type=int, required=True)
    p.add_argument('--fem_boundary_nodes', type=int, required=True)
    p.add_argument('--fem_interior_nodes', type=int, required=True)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--log_every', type=int, default=400)
    p.add_argument('--use_learned_laplace', type=int, default=1,
                   help='1: load the pretrained ambient square Laplace neural block; 0: use exact Galerkin stiffness.')
    p.add_argument('--laplace_block_path', type=str, required=True,
                   help='Path to the pretrained ambient-square Laplace block checkpoint.')
    p.add_argument('--laplace_block_sign', type=str, default='auto', choices=['auto', 'positive_stiffness', 'laplace_symbol'],
                   help='How to interpret the loaded diagonal tensor.')
    return Args(**vars(p.parse_args()))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    radius = 0.5 * args.disk_diameter

    print('=' * 100)
    print('Section-3 Allen--Cahn on a disk embedded in a fixed box [-1,1]^2')
    print('=' * 100)
    print(f'[setup] eps2={args.eps2:.3e} | dt={args.dt:.3e} | T={args.T:.3f} | K={args.K}')
    print(f'[setup] ambient box=[-{args.box_halfwidth:.3f},{args.box_halfwidth:.3f}]^2 | disk diameter={args.disk_diameter:.3f}')
    print('[setup] z0 is built directly in reduced Section-3 space; expensive live projection diagnostics removed')

    meta = build_trig_basis_meta(args.K, radial_truncation=True)
    grid_pack = build_disk_in_box_grid(args.Nx_eval, box_halfwidth=args.box_halfwidth, radius=radius)
    save_disk_geometry(grid_pack, args.Nb_build, os.path.join(args.outdir, 'disk_geometry.png'))
    x_in, y_in, w = grid_pack['x_in'], grid_pack['y_in'], grid_pack['weight']

    Phi = eval_trig_basis_from_meta(x_in, y_in, meta, Lx=args.box_halfwidth, Ly=args.box_halfwidth)
    dPhix, dPhiy = eval_trig_basis_grad_from_meta(x_in, y_in, meta, Lx=args.box_halfwidth, Ly=args.box_halfwidth)
    M_dom, K_dom = assemble_domain_mass_stiffness(Phi, dPhix, dPhiy, w)

    xb, yb, nxb, nyb = circle_boundary_points(args.Nb_build, radius=radius)
    xd, yd, nxd, nyd = circle_boundary_points(args.Nb_dense, radius=radius)
    dphix_b, dphiy_b = eval_trig_basis_grad_from_meta(xb, yb, meta, Lx=args.box_halfwidth, Ly=args.box_halfwidth)
    dphix_d, dphiy_d = eval_trig_basis_grad_from_meta(xd, yd, meta, Lx=args.box_halfwidth, Ly=args.box_halfwidth)
    C_build = nxb[:, None] * dphix_b + nyb[:, None] * dphiy_b
    C_dense = nxd[:, None] * dphix_d + nyd[:, None] * dphiy_d

    N, infoN = build_section3_basis(C_build, M_dom, tau_rel=args.tau_rel, reduced_rank=args.reduced_rank)
    print(f"[Section3 basis] rank(C)={infoN['rank_C']} | null_dim={infoN['null_dim']} | final_rank={infoN['final_rank']} | mass_eigs={infoN['mass_eigs']}")
    if args.reduced_rank <= 0:
        print('[Section3 basis] using FULL approximate null space after M-orthonormalization')

    laplace_mass_op_r = None
    laplace_block_info: Dict[str, Any] | None = None
    laplace_label = 'exact Galerkin stiffness'
    if args.use_learned_laplace:
        try:
            lambda_learn, laplace_block_info = load_learned_laplace_symbol(
                args.laplace_block_path, meta, Lx=args.box_halfwidth, Ly=args.box_halfwidth,
                sign_convention=args.laplace_block_sign,
            )
            laplace_mass_op_r = build_reduced_laplace_mass_from_symbol(N, M_dom, lambda_learn)
            laplace_label = f"pretrained ambient Laplace block: {args.laplace_block_path}"
            print(
                f"[loaded Laplace block] path={args.laplace_block_path} | key={laplace_block_info['source_key']} | "
                f"lambda=[{laplace_block_info['lambda_min']:.3e},{laplace_block_info['lambda_max']:.3e}] | "
                f"rel_to_exact_symbol={laplace_block_info['rel_to_exact_symbol']:.3e}"
            )
        except Exception:
            raise

    Phi_r = Phi @ N
    Mr = N.T @ M_dom @ N
    Kr = N.T @ K_dom @ N
    z0, u0_cont, z0_info = construct_reduced_space_interface_z0(
        Phi_r, Mr, Kr, grid_pack['X'], grid_pack['Y'], grid_pack['mask'], w,
        ic_amp=args.ic_amp, bend=args.interface_bend, nmodes=args.smooth_mode_count, seed=args.seed
    )
    a0 = N @ z0
    print(
        f"[IC reduced-space direct design] mean(u0)={z0_info['mean_u0']:+.3e} | range(u0)=[{z0_info['u0_min']:.3f},{z0_info['u0_max']:.3f}] | dense_bc={bc_violation(a0, C_dense):.3e}"
    )
    print(
        f"[IC modal design] x-like mode={z0_info['idx_x_mode']} corr_x={z0_info['corr_x']:.3f} | y2-like mode={z0_info['idx_y2_mode']} corr_y2={z0_info['corr_y2']:.3f}"
    )

    nsteps = int(round(args.T / args.dt))
    snap_steps = sorted(set([0, nsteps // 4, nsteps // 2, (3 * nsteps) // 4, nsteps]))
    snap_times = {k: k * args.dt for k in snap_steps}
    log_steps = set(range(0, nsteps + 1, args.log_every))
    log_steps.add(nsteps)
    sample_steps = set(snap_steps) | log_steps | {0}

    mesh = generate_circle_mesh(radius, args.fem_boundary_nodes, args.fem_interior_nodes, args.seed)
    u0_nodes_l2 = fem_l2_project_initial_condition(mesh, u0_cont, grid_pack)
    fem_ref = fem_strang_rollout_sampled(mesh, u0_nodes_l2, args.eps2, args.dt, args.T, sample_steps)
    fem_grid_samples: Dict[int, np.ndarray] = {}
    for s, u_nodes in fem_ref['node_samples'].items():
        fem_grid_samples[s] = fem_interpolate_to_grid(fem_ref['points'], fem_ref['tris'], u_nodes, x_in, y_in)

    t0_rel = weighted_rel_l2(u0_cont, fem_grid_samples[0], w)
    print(f"[FEM] nodes={mesh['points'].shape[0]} | tris={mesh['tris'].shape[0]}")
    print(f"[FEM t0 diagnostic] L2-projected rel-L2 at t=0 = {t0_rel:.3e}")

    res = rollout_reduced_method_sampled(
        N, Phi, K_dom, M_dom, C_dense, grid_pack, z0,
        args.eps2, args.dt, args.T, sample_steps, args.log_every, args.react_substeps,
        fem_grid_samples,
        laplace_mass_op_r=laplace_mass_op_r,
        laplace_label=laplace_label,
    )

    times_sampled = np.array(sorted(sample_steps), dtype=np.int64)
    tvals = times_sampled.astype(np.float64) * args.dt
    rel_vals = np.array([res['rel_samples'][int(s)] for s in times_sampled], dtype=np.float64)
    energy_sec3 = np.array([res['energy_samples'][int(s)] for s in times_sampled], dtype=np.float64)
    energy_fem = np.array([fem_ref['energy_samples'][int(s)] for s in times_sampled], dtype=np.float64)
    mean_sec3 = np.array([res['mean_samples'][int(s)] for s in times_sampled], dtype=np.float64)
    mean_fem = np.array([fem_ref['mean_samples'][int(s)] for s in times_sampled], dtype=np.float64)
    bc_vals = np.array([res['bc_samples'][int(s)] for s in times_sampled], dtype=np.float64)

    save_curve(tvals, [('Section3', rel_vals)], 'Rollout weighted rel-L2 vs FEM (sampled)', 'weighted rel-L2', os.path.join(args.outdir, 'relL2_vs_fem.png'), semilogy=True)
    save_curve(tvals, [('Section3', energy_sec3), ('FEM', energy_fem)], 'Volume-constrained Allen--Cahn energy decay (sampled)', 'energy', os.path.join(args.outdir, 'energy_compare.png'))
    save_curve(tvals, [('Section3', mean_sec3), ('FEM', mean_fem)], 'Spatial mean history (sampled)', 'mean(u)', os.path.join(args.outdir, 'mean_compare.png'))
    save_curve(tvals, [('Section3', bc_vals)], 'Boundary violation history (sampled)', 'relative violation', os.path.join(args.outdir, 'bc_history.png'), semilogy=True)

    sec3_snaps = {k: res['field_samples'][k] for k in snap_steps}
    fem_snaps = {k: fem_grid_samples[k] for k in snap_steps}
    save_snapshot_compare(grid_pack, sec3_snaps, fem_snaps, 'Section3', 'FEM', snap_times, os.path.join(args.outdir, 'snapshots_section3_vs_fem.png'))

    Z0 = lift_to_grid(grid_pack, u0_cont)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    im = ax.imshow(Z0.T, origin='lower', extent=grid_pack['extent'], aspect='equal', vmin=-1.0, vmax=1.0)
    ax.set_title('Initial field inside reduced space')
    ax.set_xlim(-args.box_halfwidth, args.box_halfwidth); ax.set_ylim(-args.box_halfwidth, args.box_halfwidth)
    draw_disk_outline(ax, radius)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(args.outdir, 'initial_field.png'), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
    Zs = lift_to_grid(grid_pack, res['field_samples'][nsteps])
    Zf = lift_to_grid(grid_pack, fem_grid_samples[nsteps])
    Ze = lift_to_grid(grid_pack, res['field_samples'][nsteps] - fem_grid_samples[nsteps])
    for ax in axes.ravel():
        ax.set_xlim(-args.box_halfwidth, args.box_halfwidth); ax.set_ylim(-args.box_halfwidth, args.box_halfwidth)
        draw_disk_outline(ax, radius)
    im0 = axes[0, 0].imshow(Z0.T, origin='lower', extent=grid_pack['extent'], aspect='equal', vmin=-1.0, vmax=1.0)
    axes[0, 0].set_title('Initial field')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    im1 = axes[0, 1].imshow(Zs.T, origin='lower', extent=grid_pack['extent'], aspect='equal', vmin=-1.0, vmax=1.0)
    axes[0, 1].set_title('Section3 final')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    im2 = axes[1, 0].imshow(Zf.T, origin='lower', extent=grid_pack['extent'], aspect='equal', vmin=-1.0, vmax=1.0)
    axes[1, 0].set_title('FEM final')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    im3 = axes[1, 1].imshow(Ze.T, origin='lower', extent=grid_pack['extent'], aspect='equal', cmap='coolwarm')
    axes[1, 1].set_title('Final pointwise error')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    fig.savefig(os.path.join(args.outdir, 'final_compare.png'), dpi=170)
    plt.close(fig)

    summary = {
        'args': to_jsonable(vars(args)),
        'effective_radius': float(radius),
        'laplace_block_label': laplace_label,
        'laplace_block_info': to_jsonable(laplace_block_info) if laplace_block_info is not None else None,
        't0_l2proj_relL2': float(t0_rel),
        'final_relL2': float(rel_vals[-1]),
        'sampled_mean_relL2': float(np.mean(rel_vals[1:])) if rel_vals.size > 1 else float(rel_vals[0]),
        'energy_drop_section3': float(energy_sec3[0] - energy_sec3[-1]),
        'energy_drop_fem': float(energy_fem[0] - energy_fem[-1]),
        'final_bc': float(bc_vals[-1]),
        'final_mean_section3': float(mean_sec3[-1]),
        'final_mean_fem': float(mean_fem[-1]),
        'mode_info': to_jsonable(z0_info),
        'sample_times': to_jsonable(tvals),
        'sample_relL2_vs_fem': to_jsonable(rel_vals),
        'sample_energy_section3': to_jsonable(energy_sec3),
        'sample_energy_fem': to_jsonable(energy_fem),
    }
    with open(os.path.join(args.outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print('-' * 100)
    print(f"[summary] final rel-L2={rel_vals[-1]:.3e} | sampled mean rel-L2={summary['sampled_mean_relL2']:.3e} | final bc={bc_vals[-1]:.3e}")
    print(f"[summary] energy drops: Section3={energy_sec3[0]-energy_sec3[-1]:.6e} | FEM={energy_fem[0]-energy_fem[-1]:.6e}")


if __name__ == '__main__':
    main()
