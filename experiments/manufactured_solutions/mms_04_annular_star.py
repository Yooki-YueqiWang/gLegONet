
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Section-3 manufactured-solution benchmark on an embedded irregular domain.
The diffusion operator is loaded from a pretrained ambient checkpoint.
"""
from __future__ import annotations

import argparse, json, math, os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import torch
import time

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


def build_trig_basis_meta(K: int, radial_truncation: bool = True) -> List[Tuple[str, int, int]]:
    idx: List[Tuple[int, int]] = []
    for k in range(K + 1):
        for l in range(K + 1):
            if (not radial_truncation) or (k * k + l * l <= K * K):
                idx.append((k, l))
    meta: List[Tuple[str, int, int]] = []
    for k, l in idx:
        meta.append(("coscos", k, l))
        if l > 0: meta.append(("cossin", k, l))
        if k > 0: meta.append(("sincos", k, l))
        if k > 0 and l > 0: meta.append(("sinsin", k, l))
    return meta


def eval_trig_basis_from_meta(x: np.ndarray, y: np.ndarray, meta: List[Tuple[str, int, int]], L: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    Phi = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    ax = np.pi * x / L; ay = np.pi * y / L
    for j, (kind, k, l) in enumerate(meta):
        if kind == "coscos": Phi[:, j] = np.cos(k * ax) * np.cos(l * ay)
        elif kind == "cossin": Phi[:, j] = np.cos(k * ax) * np.sin(l * ay)
        elif kind == "sincos": Phi[:, j] = np.sin(k * ax) * np.cos(l * ay)
        elif kind == "sinsin": Phi[:, j] = np.sin(k * ax) * np.sin(l * ay)
        else: raise ValueError(kind)
    return Phi


def eval_trig_basis_grad_from_meta(x: np.ndarray, y: np.ndarray, meta: List[Tuple[str, int, int]], L: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    dphix = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    dphiy = np.empty((x.shape[0], len(meta)), dtype=np.float64)
    ax = np.pi * x / L; ay = np.pi * y / L; sf = np.pi / L
    for j, (kind, k, l) in enumerate(meta):
        kf = float(k); lf = float(l)
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
        else: raise ValueError(kind)
    return dphix, dphiy


def laplace_diag_from_meta(meta: List[Tuple[str, int, int]], L: float = 1.0) -> np.ndarray:
    sf2 = (np.pi / L) ** 2
    return np.array([-sf2 * float(k*k + l*l) for _, k, l in meta], dtype=np.float64)


def build_grid_from_mask(mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], Nxy: int) -> Dict[str, Any]:
    x = np.linspace(-1.0, 1.0, Nxy, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, Nxy, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = mask_fn(X, Y)
    h = x[1] - x[0]
    return {"X": X, "Y": Y, "mask": mask, "x_in": X[mask].copy(), "y_in": Y[mask].copy(),
            "weight": (h*h) * np.ones(int(mask.sum()), dtype=np.float64), "extent": (-1.0, 1.0, -1.0, 1.0)}


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
    U = sec["U"][:, :r]; S = sec["S"][:r]; V = sec["Vh"][:r, :].T
    return V @ ((U.T @ d) / S)


def m_orthonormalize(N_raw: np.ndarray, M: np.ndarray, max_rank: int | None) -> Tuple[np.ndarray, np.ndarray]:
    G = 0.5 * (N_raw.T @ M @ N_raw + (N_raw.T @ M @ N_raw).T)
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    keep = evals > 1e-11 * max(float(evals[0]), 1.0)
    if max_rank is not None and max_rank > 0:
        ids = np.where(keep)[0][:max_rank]
        k2 = np.zeros_like(keep, dtype=bool); k2[ids] = True; keep = k2
    T = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    return N_raw @ T, evals[keep]


def weighted_lstsq_fit(Phi_r: np.ndarray, values: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    A = Phi_r.T @ (Phi_r * w[:, None]) + lam * np.eye(Phi_r.shape[1])
    b = Phi_r.T @ (w * values)
    return np.linalg.solve(0.5*(A + A.T), b)


def bc_violation(a: np.ndarray, C: np.ndarray, d: np.ndarray | None = None) -> float:
    rhs = 0.0 if d is None else d
    return float(np.linalg.norm(C @ a - rhs) / (np.linalg.norm(a) + np.linalg.norm(rhs) + 1e-14))


def rk4_step(z: np.ndarray, t: float, dt: float, rhs_fn) -> np.ndarray:
    k1 = rhs_fn(z, t)
    k2 = rhs_fn(z + 0.5*dt*k1, t + 0.5*dt)
    k3 = rhs_fn(z + 0.5*dt*k2, t + 0.5*dt)
    k4 = rhs_fn(z + dt*k3, t + dt)
    return z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def lift_to_grid(grid: Dict[str, Any], values: np.ndarray) -> np.ndarray:
    Z = np.full_like(grid["X"], np.nan, dtype=np.float64)
    Z[grid["mask"]] = values
    return Z


def plot_error_curve(times: np.ndarray, rel: np.ndarray, bc: np.ndarray, outpath: str) -> None:
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


def plot_scalar_snapshots(grid: Dict[str, Any], snaps: Dict[int, Dict[str, Any]], outpath: str, boundaries: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    ids = sorted(snaps.keys())
    fig, axes = plt.subplots(3, len(ids), figsize=(4.0*len(ids), 9.0), constrained_layout=True)
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
    for bx, by, label in boundaries: ax.plot(bx, by, lw=1.6, label=label)
    ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=180); plt.close(fig)


def polar_boundary(R_fn, Rp_fn, nb: int, *, normal: str = "outer"):
    th = np.linspace(0.0, 2*np.pi, nb, endpoint=False)
    R = R_fn(th); Rp = Rp_fn(th)
    x = R*np.cos(th); y = R*np.sin(th)
    tx = Rp*np.cos(th) - R*np.sin(th); ty = Rp*np.sin(th) + R*np.cos(th)
    nx = ty.copy(); ny = -tx.copy()
    rad_dot = nx*np.cos(th) + ny*np.sin(th)
    if normal == "outer":
        flip = rad_dot < 0
    elif normal == "inner":
        flip = rad_dot > 0
    else:
        raise ValueError(normal)
    nx[flip] *= -1; ny[flip] *= -1
    nn = np.sqrt(nx*nx + ny*ny) + 1e-15
    return x, y, nx/nn, ny/nn, th


def theta_np(x, y):
    return np.arctan2(y, x)


def theta_torch(x, y):
    return torch.atan2(y, x)

# =============================================================================
# MMS-IV: coupled autocatalytic reaction--diffusion on annular star, mixed BC
# Outer boundary: homogeneous Dirichlet. Inner boundary: homogeneous Neumann.
# =============================================================================

def Rout(theta): return 0.70 + 0.105*np.cos(5*theta) + 0.040*np.sin(2*theta)
def Routp(theta): return -0.525*np.sin(5*theta) + 0.080*np.cos(2*theta)
def Rin(theta): return 0.23 + 0.055*np.cos(3*theta + np.pi/5)
def Rinp(theta): return -0.165*np.sin(3*theta + np.pi/5)

def Rout_t(theta): return 0.70 + 0.105*torch.cos(5*theta) + 0.040*torch.sin(2*theta)
def Rin_t(theta): return 0.23 + 0.055*torch.cos(3*theta + np.pi/5)

def mask_fn(X,Y):
    th=np.arctan2(Y,X); rr=np.sqrt(X*X+Y*Y)
    return (rr <= Rout(th)) & (rr >= Rin(th))

def qout_torch(x,y):
    th=torch.atan2(y,x); r2=x*x+y*y
    return Rout_t(th)**2 - r2

def qin_torch(x,y):
    th=torch.atan2(y,x); r2=x*x+y*y
    return r2 - Rin_t(th)**2

def exact_uv_torch(x,y,t,amp_u,amp_v,omega):
    th=torch.atan2(y,x)
    qout=qout_torch(x,y)
    qin=qin_torch(x,y)
    s=qout*(qin*qin)
    eta=qin/(qin+qout+1e-12)
    band_inner=torch.exp(-((eta-0.26)/0.12)**2)
    band_mid=torch.exp(-((eta-0.50)/0.15)**2)
    band_outer=torch.exp(-((eta-0.73)/0.13)**2)
    phase_main=omega*t + 0.40*torch.sin(1.25*t)
    phase_aux=-0.60*omega*t + 0.55*torch.cos(0.95*t)
    spot_main=torch.exp(3.1*(torch.cos(th-phase_main)-1.0))
    spot_pair=torch.exp(2.8*(torch.cos(2.0*(th-phase_aux)-1.1)-1.0))
    spot_inner=torch.exp(2.9*(torch.cos(th+0.45*omega*t+0.8)-1.0))
    stripe=torch.cos(4.0*th - 1.1*t + 0.35*torch.sin(0.7*t))
    transfer=torch.tanh(3.0*(t-0.42))
    breathing=torch.sin(np.pi*(eta-0.18) - 1.05*t)
    u_shape=(
        1.35*(0.65+0.35*torch.sin(1.15*t)**2)*band_mid*spot_main
        + 0.95*(0.55+0.45*torch.cos(0.85*t))*band_outer*spot_pair
        - 0.80*transfer*band_inner*spot_inner
        + 0.32*band_mid*stripe*breathing
    )
    v_shape=(
        1.00*(0.60+0.40*torch.cos(1.05*t))*band_mid*torch.exp(2.7*(torch.cos(th-phase_main-0.85)-1.0))
        - 0.72*band_outer*spot_pair
        + 0.58*(0.55+0.45*torch.sin(0.9*t+0.6)**2)*band_inner*torch.cos(3.0*th + 0.55*t)
        + 0.30*torch.sin(2.0*np.pi*eta - 0.85*t)*torch.cos(2.0*th - 0.45*t)
    )
    return amp_u*s*u_shape, amp_v*s*v_shape

def exact_force(x_np,y_np,t_scalar,*,Du,Dv,a_param,b_param,amp_u,amp_v,omega,device):
    x=torch.tensor(x_np,dtype=DTYPE,device=device,requires_grad=True); y=torch.tensor(y_np,dtype=DTYPE,device=device,requires_grad=True)
    t=torch.full_like(x,float(t_scalar),requires_grad=True)
    u,v=exact_uv_torch(x,y,t,amp_u,amp_v,omega); ones=torch.ones_like(u)
    ux=torch.autograd.grad(u,x,grad_outputs=ones,create_graph=True)[0]; uy=torch.autograd.grad(u,y,grad_outputs=ones,create_graph=True)[0]
    vx=torch.autograd.grad(v,x,grad_outputs=ones,create_graph=True)[0]; vy=torch.autograd.grad(v,y,grad_outputs=ones,create_graph=True)[0]
    uxx=torch.autograd.grad(ux,x,grad_outputs=torch.ones_like(ux),create_graph=True)[0]; uyy=torch.autograd.grad(uy,y,grad_outputs=torch.ones_like(uy),create_graph=True)[0]
    vxx=torch.autograd.grad(vx,x,grad_outputs=torch.ones_like(vx),create_graph=True)[0]; vyy=torch.autograd.grad(vy,y,grad_outputs=torch.ones_like(vy),create_graph=True)[0]
    ut=torch.autograd.grad(u,t,grad_outputs=ones,create_graph=True)[0]; vt=torch.autograd.grad(v,t,grad_outputs=ones,create_graph=True)[0]
    fu=ut - Du*(uxx+uyy) - (a_param - u + u*u*v)
    fv=vt - Dv*(vxx+vyy) - (b_param - u*u*v)
    return (u.detach().cpu().numpy(),v.detach().cpu().numpy()),(fu.detach().cpu().numpy(),fv.detach().cpu().numpy())

def combined_rel(u_pred,u_ref,v_pred,v_ref,w):
    num=float(np.sum(w*(u_pred-u_ref)**2)+np.sum(w*(v_pred-v_ref)**2))
    den=float(np.sum(w*u_ref*u_ref)+np.sum(w*v_ref*v_ref))+1e-14
    return math.sqrt(num/den)

def plot_component(grid,snaps,outpath,bounds,comp):
    ids=sorted(snaps.keys())
    fig,axes=plt.subplots(3,len(ids),figsize=(4*len(ids),9),constrained_layout=True)
    if len(ids)==1:
        axes=axes.reshape(3,1)
    all_vals=[]
    all_errs=[]
    for k in ids:
        all_vals.append(snaps[k][f"pred_{comp}"])
        all_vals.append(snaps[k][f"ref_{comp}"])
        all_errs.append(snaps[k][f"pred_{comp}"]-snaps[k][f"ref_{comp}"])
    vmin=float(min(np.min(v) for v in all_vals))
    vmax=float(max(np.max(v) for v in all_vals))
    emax=float(max(np.max(np.abs(e)) for e in all_errs)) + 1e-14
    for j,k in enumerate(ids):
        pred=lift_to_grid(grid,snaps[k][f"pred_{comp}"])
        ref=lift_to_grid(grid,snaps[k][f"ref_{comp}"])
        err=lift_to_grid(grid,snaps[k][f"pred_{comp}"]-snaps[k][f"ref_{comp}"])
        im0=axes[0,j].imshow(pred.T,origin="lower",extent=grid["extent"],vmin=vmin,vmax=vmax,aspect="equal")
        axes[0,j].set_title(f"predicted solution, t={snaps[k]['t']:.3f}")
        fig.colorbar(im0,ax=axes[0,j],fraction=0.046)
        im1=axes[1,j].imshow(ref.T,origin="lower",extent=grid["extent"],vmin=vmin,vmax=vmax,aspect="equal")
        axes[1,j].set_title(f"reference solution, t={snaps[k]['t']:.3f}")
        fig.colorbar(im1,ax=axes[1,j],fraction=0.046)
        im2=axes[2,j].imshow(
            err.T,
            origin="lower",
            extent=grid["extent"],
            cmap="coolwarm",
            vmin=-emax,
            vmax=emax,
            aspect="equal",
        )
        axes[2,j].set_title("pointwise error")
        fig.colorbar(im2,ax=axes[2,j],fraction=0.046)
        for ax in axes[:,j]:
            for bx,by in bounds:
                ax.plot(bx,by,"k-",lw=1.0)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
    fig.savefig(outpath,dpi=180)
    plt.close(fig)


def plot_component_change_from_initial(grid,snaps,outpath,bounds,comp):
    ids=sorted(snaps.keys())
    if not ids:
        return
    ref0=snaps[ids[0]][f"pred_{comp}"]
    diffs=[snaps[k][f"pred_{comp}"]-ref0 for k in ids]
    emax=float(max(np.max(np.abs(d)) for d in diffs)) + 1e-14
    fig,axes=plt.subplots(1,len(ids),figsize=(3.8*len(ids),3.4),constrained_layout=True)
    if len(ids)==1:
        axes=[axes]
    for j,k in enumerate(ids):
        Z=lift_to_grid(grid,diffs[j])
        im=axes[j].imshow(
            Z.T,
            origin="lower",
            extent=grid["extent"],
            cmap="coolwarm",
            vmin=-emax,
            vmax=emax,
            aspect="equal",
        )
        for bx,by in bounds:
            axes[j].plot(bx,by,"k-",lw=1.0)
        axes[j].set_xlim(-1,1)
        axes[j].set_ylim(-1,1)
        axes[j].set_title(f"{comp}(t)-{comp}(0), t={snaps[k]['t']:.3f}")
        fig.colorbar(im,ax=axes[j],fraction=0.046)
    fig.savefig(outpath,dpi=180)
    plt.close(fig)


def _rms_from_sq_count(sq_sum: float, count: int) -> float:
    return float(np.sqrt(float(sq_sum) / max(int(count), 1)))


def _write_table_metrics_json(outdir: str, summary: Dict[str, Any], rel_arr: np.ndarray,
                              bc_arr: np.ndarray, solution_sq_sum: float,
                              solution_count: int) -> None:
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
    path = os.path.join(outdir, "table_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics), f, indent=2)
    print(f"[table] wrote {path}")


def _write_vector_rollout_csv(out_path: str, x: np.ndarray, y: np.ndarray,
                              field_records: List[Dict[str, Any]],
                              times: np.ndarray, rel_arr: np.ndarray) -> None:
    header = [
        "row_type", "time_index", "time", "point_index", "x", "y",
        "reference_u", "reference_v", "pred_u", "pred_v",
        "pointwise_error_u", "pointwise_error_v", "relative_l2_error",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for k, t in enumerate(times):
            f.write(f"history,{k},{float(t):.17g},,,,,,,,,,{float(rel_arr[k]):.17g}\n")
        for rec in field_records:
            k = int(rec["k"])
            t = float(rec["t"])
            ru = np.asarray(rec["ref_u"], dtype=np.float64).reshape(-1)
            rv = np.asarray(rec["ref_v"], dtype=np.float64).reshape(-1)
            pu = np.asarray(rec["pred_u"], dtype=np.float64).reshape(-1)
            pv = np.asarray(rec["pred_v"], dtype=np.float64).reshape(-1)
            for q in range(ru.size):
                eu = pu[q] - ru[q]
                ev = pv[q] - rv[q]
                f.write(
                    f"field,{k},{t:.17g},{q},{float(x[q]):.17g},{float(y[q]):.17g},"
                    f"{float(ru[q]):.17g},{float(rv[q]):.17g},{float(pu[q]):.17g},{float(pv[q]):.17g},"
                    f"{float(eu):.17g},{float(ev):.17g},{float(rec['rel']):.17g}\n"
                )
    print(f"[csv] wrote {out_path}")

@dataclass
class Args:
    outdir:str; K:int; Nx_eval:int; Nb_outer:int; Nb_inner:int; Nb_dense:int; reduced_rank:int; tau_rel:float
    T:float; dt:float; Du:float; Dv:float; a_param:float; b_param:float; amp_u:float; amp_v:float; omega:float; fit_lam:float; seed:int; device:str; log_every:int; field_time_stride:int; save_operators:bool; laplace_checkpoint:str

def parse_args()->Args:
    p=argparse.ArgumentParser(description="MMS-IV dynamic annular-star coupled RD with mixed Section 3 boundary conditions.")
    p.add_argument("--outdir",type=str,required=True)
    p.add_argument("--K",type=int,required=True)
    p.add_argument("--Nx_eval",type=int,required=True)
    p.add_argument("--Nb_outer",type=int,required=True)
    p.add_argument("--Nb_inner",type=int,required=True)
    p.add_argument("--Nb_dense",type=int,required=True)
    p.add_argument("--reduced_rank",type=int,required=True)
    p.add_argument("--tau_rel",type=float,default=1e-10)
    p.add_argument("--T",type=float,required=True)
    p.add_argument("--dt",type=float,required=True)
    p.add_argument("--Du",type=float,required=True)
    p.add_argument("--Dv",type=float,required=True)
    p.add_argument("--a_param",type=float,required=True)
    p.add_argument("--b_param",type=float,required=True)
    p.add_argument("--amp_u",type=float,required=True)
    p.add_argument("--amp_v",type=float,required=True)
    p.add_argument("--omega",type=float,required=True)
    p.add_argument("--fit_lam",type=float,default=1e-11);
    p.add_argument("--seed",type=int,default=1234);
    p.add_argument("--device",type=str,default="cpu");
    p.add_argument("--log_every",type=int,default=50)
    p.add_argument("--field_time_stride", type=int, default=1, help="Save full-field CSV rows every this many time steps; use <=0 to save history only.")
    p.add_argument("--save_operators", action="store_true", help="Save C, M_omega, and N_omega for inspection.")
    p.add_argument("--laplace_checkpoint",type=str,required=True)
    return Args(**vars(p.parse_args()))

def main():
    args=parse_args(); wall_time_start = time.perf_counter(); os.makedirs(args.outdir,exist_ok=True); set_seed(args.seed); device=torch.device(args.device)
    meta=build_trig_basis_meta(args.K,radial_truncation=True)
    laplace_operator,laplace_diagnostics=load_laplace_operator(meta,args.K,args.laplace_checkpoint)
    grid=build_grid_from_mask(mask_fn,args.Nx_eval); x_in,y_in,wq=grid["x_in"],grid["y_in"],grid["weight"]
    Phi=eval_trig_basis_from_meta(x_in,y_in,meta); dPhix,dPhiy=eval_trig_basis_grad_from_meta(x_in,y_in,meta); M=Phi.T@(Phi*wq[:,None])
    xo,yo,nxo,nyo,_=polar_boundary(Rout,Routp,args.Nb_outer,normal="outer"); xi,yi,nxi,nyi,_=polar_boundary(Rin,Rinp,args.Nb_inner,normal="inner")
    xod,yod,_,_,_=polar_boundary(Rout,Routp,args.Nb_dense,normal="outer"); xid,yid,nxid,nyid,_=polar_boundary(Rin,Rinp,args.Nb_dense,normal="inner")
    # Build mixed C: outer Dirichlet + inner Neumann
    C_outer=eval_trig_basis_from_meta(xo,yo,meta)
    dpxi,dpyi=eval_trig_basis_grad_from_meta(xi,yi,meta); C_inner=nxi[:,None]*dpxi+nyi[:,None]*dpyi
    C=np.vstack([C_outer,C_inner])
    C_outer_d=eval_trig_basis_from_meta(xod,yod,meta)
    dpxid,dpyid=eval_trig_basis_grad_from_meta(xid,yid,meta); C_inner_d=nxid[:,None]*dpxid+nyid[:,None]*dpyid
    C_dense=np.vstack([C_outer_d,C_inner_d])
    sec=tangent_space_from_boundary(C,args.tau_rel); N,_=m_orthonormalize(sec["N"],M,args.reduced_rank); PhiN=Phi@N
    constraint_residual=float(np.linalg.norm(C@N)/(np.linalg.norm(C)*np.linalg.norm(N)+1e-14))
    mass_orthonormality_residual=float(np.linalg.norm(N.T@M@N-np.eye(N.shape[1]))/max(np.sqrt(N.shape[1]),1.0))
    print(f"[Section3] M={len(meta)} rank(C)={sec['rank']} null={sec['null_dim']} reduced={N.shape[1]} | mixed outer D / inner N")
    print(f"[Section3] ||C N_omega||/(||C|| ||N_omega||)={constraint_residual:.3e} mass_residual={mass_orthonormality_residual:.3e}")
    if args.save_operators:
        np.savez_compressed(
            os.path.join(args.outdir,"boundary_operators.npz"),
            C=C,
            M_omega=M,
            N_omega=N,
            outer_x=xo,
            outer_y=yo,
            inner_x=xi,
            inner_y=yi,
            inner_normal_x=nxi,
            inner_normal_y=nyi,
        )
    plot_domain(grid,os.path.join(args.outdir,"domain.png"),[(xod,yod,"outer Dirichlet"),(xid,yid,"inner Neumann")])
    (u0,v0),_=exact_force(x_in,y_in,0.0,Du=args.Du,Dv=args.Dv,a_param=args.a_param,b_param=args.b_param,amp_u=args.amp_u,amp_v=args.amp_v,omega=args.omega,device=device)
    zu=weighted_lstsq_fit(PhiN,u0,wq,args.fit_lam); zv=weighted_lstsq_fit(PhiN,v0,wq,args.fit_lam)
    nsteps=int(round(args.T/args.dt)); times=np.linspace(0,args.T,nsteps+1)
    rel_hist=[]; bc_hist=[]; change_hist=[]; snaps={}; snap_ids={0,nsteps//2,nsteps}
    field_records=[]; solution_sq_sum=0.0; solution_count=0
    up0=None; vp0=None
    max_abs_u=0.0; max_abs_v=0.0
    def rhs_comp(a_u,a_v,t):
        u=Phi@a_u; v=Phi@a_v; lapu=Phi@(laplace_operator@a_u); lapv=Phi@(laplace_operator@a_v)
        _,(fu,fv)=exact_force(x_in,y_in,t,Du=args.Du,Dv=args.Dv,a_param=args.a_param,b_param=args.b_param,amp_u=args.amp_u,amp_v=args.amp_v,omega=args.omega,device=device)
        rhsu=args.Du*lapu + (args.a_param - u + u*u*v) + fu
        rhsv=args.Dv*lapv + (args.b_param - u*u*v) + fv
        return rhsu,rhsv
    def rhs_fn(zcat,t):
        r=N.shape[1]; au=N@zcat[:r]; av=N@zcat[r:]
        rhsu,rhsv=rhs_comp(au,av,t)
        return np.concatenate([weighted_lstsq_fit(PhiN,rhsu,wq,args.fit_lam), weighted_lstsq_fit(PhiN,rhsv,wq,args.fit_lam)])
    zcat=np.concatenate([zu,zv])
    for k,t in enumerate(times):
        r=N.shape[1]; au=N@zcat[:r]; av=N@zcat[r:]
        up=Phi@au; vp=Phi@av; (ur,vr),_=exact_force(x_in,y_in,float(t),Du=args.Du,Dv=args.Dv,a_param=args.a_param,b_param=args.b_param,amp_u=args.amp_u,amp_v=args.amp_v,omega=args.omega,device=device)
        if up0 is None:
            up0=up.copy()
            vp0=vp.copy()
        rel=combined_rel(up,ur,vp,vr,wq)
        solution_sq_sum += float(np.sum(ur*ur + vr*vr))
        solution_count += int(ur.size)
        if args.field_time_stride > 0 and (k % args.field_time_stride == 0 or k == nsteps):
            field_records.append({"k": int(k), "t": float(t), "ref_u": ur.copy(), "ref_v": vr.copy(), "pred_u": up.copy(), "pred_v": vp.copy(), "rel": float(rel)})
        bc=max(bc_violation(au,C_dense),bc_violation(av,C_dense))
        change=combined_rel(up,up0,vp,vp0,wq)
        rel_hist.append(rel); bc_hist.append(bc); change_hist.append(change)
        max_abs_u=max(max_abs_u,float(np.max(np.abs(ur))),float(np.max(np.abs(up))))
        max_abs_v=max(max_abs_v,float(np.max(np.abs(vr))),float(np.max(np.abs(vp))))
        if k in snap_ids: snaps[k]={"pred_u":up.copy(),"ref_u":ur.copy(),"pred_v":vp.copy(),"ref_v":vr.copy(),"t":float(t)}
        if k<nsteps: zcat=rk4_step(zcat,float(t),args.dt,rhs_fn)
        if k%args.log_every==0 or k==nsteps: print(f"[rollout] step={k:04d}/{nsteps:04d} t={t:.3f} rel={rel:.3e} mixed_bc={bc:.3e} change={change:.3e}")
    rel_arr=np.array(rel_hist); bc_arr=np.array(bc_hist); change_arr=np.array(change_hist)
    plot_error_curve(times,rel_arr,bc_arr,os.path.join(args.outdir,"error_curves.png"))
    bounds=[(xod,yod),(xid,yid)]
    plot_component(grid,snaps,os.path.join(args.outdir,"snapshots_section3_vs_exact_u.png"),bounds,"u")
    plot_component(grid,snaps,os.path.join(args.outdir,"snapshots_section3_vs_exact_v.png"),bounds,"v")
    plot_component_change_from_initial(grid,snaps,os.path.join(args.outdir,"change_from_initial_u.png"),bounds,"u")
    plot_component_change_from_initial(grid,snaps,os.path.join(args.outdir,"change_from_initial_v.png"),bounds,"v")
    summary={"args":to_jsonable(vars(args)),"basis_dim":len(meta),"constraint_rank":int(sec["rank"]),"raw_null_dimension":int(sec["null_dim"]),"reduced_rank":int(N.shape[1]),"constraint_residual":constraint_residual,"mass_orthonormality_residual":mass_orthonormality_residual,"final_rel_l2":float(rel_arr[-1]),"mean_rel_l2":float(np.mean(rel_arr)),"max_rel_l2":float(np.max(rel_arr)),"final_mixed_bc":float(bc_arr[-1]),"final_change_from_initial":float(change_arr[-1]),"max_change_from_initial":float(np.max(change_arr)),"max_abs_u":float(max_abs_u),"max_abs_v":float(max_abs_v),"laplace_checkpoint_diagnostics":laplace_diagnostics,"total_time_sec":float(time.perf_counter()-wall_time_start)}
    with open(os.path.join(args.outdir,"summary.json"),"w",encoding="utf-8") as f: json.dump(to_jsonable(summary),f,indent=2)
    _write_table_metrics_json(args.outdir, summary, rel_arr, bc_arr, solution_sq_sum, solution_count)
    _write_vector_rollout_csv(os.path.join(args.outdir,"rollout_fields_and_relerr.csv"), x_in, y_in, field_records, times, rel_arr)
    print(f"[summary] final_rel={rel_arr[-1]:.3e} max_rel={rel_arr.max():.3e} final_bc={bc_arr[-1]:.3e} final_change={change_arr[-1]:.3e}")
if __name__=="__main__": main()
