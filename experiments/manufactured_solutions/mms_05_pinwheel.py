#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous-MMS analytic-reference benchmark for scalar 2D Burgers on the
pinwheel-shell irregular domain.

This variant keeps the same Section-3 homogeneous Dirichlet transfer space as
the original MMS-V script, but changes the analytic manufactured solution so it
contains deliberate off-manifold oscillatory content.  The reported rollout
error is always measured against the analytic continuum field, not merely
against the reduced projection.  The projection floor is written explicitly so
the experiment can be interpreted as a genuine continuous-MMS transfer test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from block_checkpoints import (
    load_laplace_operator,
    load_transport_model,
    transport_multiplier,
)

DTYPE = torch.float64


def _load_base_module():
    base_path = Path(__file__).with_name("mms_05_base.py")
    spec = importlib.util.spec_from_file_location("mms5_section3_dynamic_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base MMS-V source: {base_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_base_module()

# Re-export the geometry/boundary helpers expected by the baseline scripts.
mask_fn = base.mask_fn
boundary_pinwheel_shell = base.boundary_pinwheel_shell
q_torch = base.q_torch
q_np = base.q_np


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_jsonable(obj: Any) -> Any:
    return base.to_jsonable(obj)


def weighted_rel_l2(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return base.weighted_rel_l2(u, v, w)


def exact_u_torch(x, y, t, amp: float, omega: float):
    """Analytic Burgers field with explicit off-manifold high-frequency packets.

    The extra packets keep the field homogeneous Dirichlet because the whole
    expression is multiplied by the same geometry factor q(x,y), but they are
    not tailored to the reduced Section-3 basis.  This avoids the near-exact
    reduced-space fit seen in the old MMS-V construction.
    """
    phase = omega * t
    geom = q_torch(x, y) / 0.45

    # Keep the original transport-like structure as the low-frequency backbone.
    smooth_backbone = 0.82 * base.exact_u_torch(x, y, t, amp, omega)

    # Add two moving oscillatory packets and one sheared wake that stay
    # perfectly boundary-admissible but are less aligned with the transfer
    # space.  Their frequencies sit near / slightly above the operator
    # baselines' spectral truncations, while still being continuous and smooth.
    xi1 = 0.74 * x - 0.36 * y - 0.14 * torch.sin(0.43 * phase)
    xi2 = 0.29 * x + 0.81 * y + 0.11 * torch.cos(0.37 * phase)
    packet_cx = -0.06 + 0.18 * torch.cos(0.41 * phase + 0.5)
    packet_cy = 0.03 - 0.16 * torch.sin(0.46 * phase - 0.2)
    packet_env = torch.exp(-(((x - packet_cx) / 0.44) ** 2 + ((y - packet_cy) / 0.34) ** 2))
    hi_packet = torch.sin(21.0 * math.pi * xi1) * torch.sin(17.0 * math.pi * xi2)

    ridge_coord = 0.63 * x + 0.28 * y
    shear_coord = 0.52 * x - 0.61 * y + 0.10 * torch.cos(0.33 * phase)
    shear_wave = torch.sin(19.0 * math.pi * ridge_coord - 0.95 * phase) * torch.exp(-(shear_coord / 0.28) ** 2)

    swirl_cx = 0.14 * torch.sin(0.57 * phase + 0.8)
    swirl_cy = -0.17 * torch.cos(0.49 * phase - 0.4)
    swirl_env = torch.exp(-(((x - swirl_cx) / 0.24) ** 2 + ((y - swirl_cy) / 0.24) ** 2))
    swirl_wave = torch.cos(17.0 * math.pi * (x + 0.22 * y) + 0.61 * phase) * swirl_env

    off_manifold = amp * geom * (
        0.022 * packet_env * hi_packet
        + 0.013 * shear_wave
        + 0.010 * swirl_wave
    )
    amplitude_mod = 1.0 + 0.08 * torch.sin(0.31 * phase)
    return smooth_backbone + amplitude_mod * off_manifold


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
    return base.snapshot_ids_for(nsteps)


def parse_args() -> base.Args:
    p = argparse.ArgumentParser(
        description="Continuous-MMS analytic-reference Section-3 benchmark for MMS-V scalar Burgers."
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
    p.add_argument("--laplace_checkpoint", type=str, required=True)
    p.add_argument("--transport_checkpoint", type=str, required=True)
    p.add_argument(
        "--field_time_stride",
        type=int,
        default=1,
        help="Save full-field CSV rows every this many time steps; use <=0 to save history only.",
    )
    return base.Args(**vars(p.parse_args()))


def main():
    args = parse_args()
    wall_time_start = time.perf_counter()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    meta = base.build_trig_basis_meta(args.K, radial_truncation=True)
    laplace_operator, laplace_diagnostics = load_laplace_operator(meta, args.K, args.laplace_checkpoint)
    transport_model = load_transport_model(args.K, args.transport_checkpoint, device)
    grid = base.build_grid_from_mask(mask_fn, args.Nx_eval)
    x_in, y_in, wq = grid["x_in"], grid["y_in"], grid["weight"]
    Phi = base.eval_trig_basis_from_meta(x_in, y_in, meta)
    dPhix, dPhiy = base.eval_trig_basis_grad_from_meta(x_in, y_in, meta)
    M = Phi.T @ (Phi * wq[:, None])

    xb, yb, _, _ = boundary_pinwheel_shell(args.Nb)
    xd, yd, _, _ = boundary_pinwheel_shell(args.Nb_dense)
    C = base.eval_trig_basis_from_meta(xb, yb, meta)
    C_dense = base.eval_trig_basis_from_meta(xd, yd, meta)
    sec = base.tangent_space_from_boundary(C, args.tau_rel)
    N, _ = base.m_orthonormalize(sec["N"], M, args.reduced_rank)
    PhiN = Phi @ N

    print(f"[Section3] M={len(meta)} rank(C)={sec['rank']} null={sec['null_dim']} reduced={N.shape[1]}")
    base.plot_domain(grid, os.path.join(args.outdir, "domain.png"), [(xd, yd, "Dirichlet boundary")])

    def analytic_values_in_domain(t: float) -> np.ndarray:
        return exact_and_force(x_in, y_in, float(t), nu=args.nu, amp=args.amp, omega=args.omega, device=device)[0]

    def analytic_source_in_domain(t: float) -> np.ndarray:
        return exact_and_force(x_in, y_in, float(t), nu=args.nu, amp=args.amp, omega=args.omega, device=device)[1]

    def project_exact_to_reduced(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        u_true = analytic_values_in_domain(t)
        z_ref = base.weighted_lstsq_fit(PhiN, u_true, wq, args.fit_lam)
        a_ref = N @ z_ref
        u_proj = Phi @ a_ref
        proj_rel = weighted_rel_l2(u_proj, u_true, wq)
        return z_ref, a_ref, u_proj, proj_rel

    z, a_init_ref, u_init_ref, init_proj_rel = project_exact_to_reduced(0.0)
    u_init_analytic = analytic_values_in_domain(0.0)
    init_bc = base.bc_violation(a_init_ref, C_dense)
    print(
        f"[init reduced projection] rel_to_analytic={init_proj_rel:.3e} "
        f"rel_to_projected=0.000e+00 dirichlet_bc={init_bc:.3e} "
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
        a = N @ zv
        u = Phi @ a
        ux = dPhix @ a
        uy = dPhiy @ a
        lap = Phi @ (laplace_operator @ a)
        f_val = analytic_source_in_domain(float(t))
        learned_multiplier = transport_multiplier(transport_model, u, device)
        rhs_values = args.nu * lap - learned_multiplier * (ux + uy) + f_val
        return base.weighted_lstsq_fit(PhiN, rhs_values, wq, args.fit_lam)

    def rhs_fn(zv, t):
        return reduced_rhs(zv, float(t))

    for k, t in enumerate(times):
        t_float = float(t)
        a = N @ z
        pred = Phi @ a
        z_ref, a_ref, projected_ref, proj_rel = project_exact_to_reduced(t_float)
        analytic_ref = analytic_values_in_domain(t_float)
        rel = weighted_rel_l2(pred, analytic_ref, wq)
        rel_projected = weighted_rel_l2(pred, projected_ref, wq)
        bc = base.bc_violation(a, C_dense)
        change = weighted_rel_l2(pred, u_init_analytic, wq)
        max_u = max(float(np.max(np.abs(pred))), float(np.max(np.abs(analytic_ref))))
        solution_sq_sum += float(np.sum(analytic_ref * analytic_ref))
        solution_count += int(analytic_ref.size)
        if args.field_time_stride > 0 and (k % args.field_time_stride == 0 or k == nsteps):
            field_records.append(
                {"k": int(k), "t": t_float, "ref": analytic_ref.copy(), "pred": pred.copy(), "rel": float(rel)}
            )
        rel_hist.append(rel)
        rel_projected_hist.append(rel_projected)
        proj_hist.append(proj_rel)
        bc_hist.append(bc)
        change_hist.append(change)
        max_hist.append(max_u)
        if k in snap_ids:
            snaps[k] = {"pred": pred.copy(), "ref": analytic_ref.copy(), "t": t_float}
        if k < nsteps:
            z = base.rk4_step(z, t_float, args.dt, rhs_fn)
        if k % max(args.log_every, 1) == 0 or k == nsteps:
            print(
                f"[rollout] step={k:04d}/{nsteps:04d} t={t_float:.3f} "
                f"rel_exact={rel:.3e} rel_projected={rel_projected:.3e} "
                f"projection_floor={proj_rel:.3e} bc={bc:.3e} "
                f"change={change:.3e} max|u|={max_u:.3e}"
            )

    rel_arr = np.array(rel_hist)
    rel_projected_arr = np.array(rel_projected_hist)
    proj_arr = np.array(proj_hist)
    bc_arr = np.array(bc_hist)
    change_arr = np.array(change_hist)
    max_arr = np.array(max_hist)

    base.plot_error_curve(
        times,
        rel_arr,
        bc_arr,
        os.path.join(args.outdir, "error_curves.png"),
        "MMS-V scalar Burgers, error to analytic solution",
    )
    base.plot_curve(
        times,
        [
            ("rollout error to reduced projection", rel_projected_arr),
            ("projection floor to analytic field", proj_arr),
        ],
        os.path.join(args.outdir, "projected_diagnostics.png"),
        "Projected-space diagnostics",
        "relative error",
        semilogy=True,
    )
    base.plot_scalar_snapshots(
        grid,
        snaps,
        os.path.join(args.outdir, "snapshots_section3_vs_exact.png"),
        (xd, yd),
        "MMS-V scalar Burgers: predicted vs analytic exact",
    )
    base.plot_change_from_initial(
        grid,
        snaps,
        os.path.join(args.outdir, "change_from_initial.png"),
        (xd, yd),
        "MMS-V scalar Burgers: change from analytic initial field",
    )
    base.plot_curve(
        times,
        [("relative change from initial", change_arr)],
        os.path.join(args.outdir, "change_history.png"),
        "Change from initial condition",
        "relative change",
    )
    base.plot_curve(
        times,
        [("max |u|", max_arr)],
        os.path.join(args.outdir, "max_history.png"),
        "Maximum amplitude history",
        "max |u|",
    )

    total_wall_time_sec = time.perf_counter() - wall_time_start
    summary = {
        "problem": "mms5",
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
        "final_bc": float(bc_arr[-1]),
        "final_change_from_initial_analytic": float(change_arr[-1]),
        "max_change_from_initial_analytic": float(np.max(change_arr)),
        "max_abs_u": float(np.max(max_arr)),
        "laplace_checkpoint_diagnostics": laplace_diagnostics,
        "transport_realization": "target-domain h_theta second derivative",
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    base._write_table_metrics_json(
        args.outdir,
        summary,
        rel_arr,
        bc_arr,
        solution_sq_sum,
        solution_count,
        extra={
            "problem": "mms5",
            "final_rel_l2_projected": float(rel_projected_arr[-1]),
            "mean_rel_l2_projected": float(np.mean(rel_projected_arr)),
            "max_rel_l2_projected": float(np.max(rel_projected_arr)),
            "final_projection_error_to_analytic": float(proj_arr[-1]),
            "max_projection_error_to_analytic": float(np.max(proj_arr)),
        },
    )
    base._write_scalar_rollout_csv(
        os.path.join(args.outdir, "rollout_fields_and_relerr.csv"),
        x_in,
        y_in,
        field_records,
        times,
        rel_arr,
    )
    print(
        f"[summary] final_rel_exact={rel_arr[-1]:.3e} max_rel_exact={rel_arr.max():.3e} "
        f"final_rel_projected={rel_projected_arr[-1]:.3e} "
        f"max_projection_floor={proj_arr.max():.3e} final_bc={bc_arr[-1]:.3e} "
        f"final_change={change_arr[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
