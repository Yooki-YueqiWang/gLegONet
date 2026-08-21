#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import common as bc
import model as k22mod


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_learned_model_vs_fv(
    model: k22mod.TorchReducedVectorBurgersModel,
    grid: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = model.device
    dtype = model.dtype
    x = np.asarray(grid["x"], dtype=np.float64)
    y = np.asarray(grid["y"], dtype=np.float64)
    w_t = model.w

    u0_np, v0_np = bc.rescaled_sine_ic(x, y, args.inner_halfwidth)
    u0_t = torch.as_tensor(u0_np, dtype=dtype, device=device)
    v0_t = torch.as_tensor(v0_np, dtype=dtype, device=device)
    zu = model.P0 @ u0_t
    zv = model.P0 @ v0_t

    xv_ref, yv_ref, u_ref, v_ref, dx_ref = bc.fv_initial_from_values(args.inner_halfwidth, args.ref_N)
    nsteps = int(round(float(args.T) / float(args.dt)))

    rows: List[Dict[str, Any]] = []
    learned_rollout_time_sec = 0.0
    reference_rollout_time_sec = 0.0
    total_reference_substeps = 0
    loop_t0 = time.perf_counter()

    for step in range(nsteps + 1):
        u = model.lift(zu)
        v = model.lift(zv)
        ur_q = bc.bilinear_interpolate(u_ref, xv_ref, yv_ref, x, y, args.inner_halfwidth)
        vr_q = bc.bilinear_interpolate(v_ref, xv_ref, yv_ref, x, y, args.inner_halfwidth)
        ur_t = torch.as_tensor(ur_q, dtype=dtype, device=device)
        vr_t = torch.as_tensor(vr_q, dtype=dtype, device=device)

        row: Dict[str, Any] = {
            "time": step * float(args.dt),
            "rel_u": k22mod.weighted_rel_l2_torch(u, ur_t, w_t),
            "rel_v": k22mod.weighted_rel_l2_torch(v, vr_t, w_t),
            "rel_combined": k22mod.combined_rel_l2_torch(u, v, ur_t, vr_t, w_t),
            "bc_u": model.boundary_residual(zu),
            "bc_v": model.boundary_residual(zv),
            "energy_learned": model.kinetic_energy(zu, zv),
            "energy_reference": bc.fv_energy(u_ref, v_ref, dx_ref),
            "learned_step_sec": 0.0,
            "reference_step_sec": 0.0,
            "reference_substeps": 0,
        }
        rows.append(row)

        if step == nsteps:
            break

        sync_device(device)
        t_model0 = time.perf_counter()
        zu, zv = model.strang_step(zu, zv, args.dt, args.nonlinear_substeps)
        sync_device(device)
        learned_step_sec = time.perf_counter() - t_model0

        t_ref0 = time.perf_counter()
        u_ref, v_ref, ns = bc.fv_advance(
            u_ref,
            v_ref,
            args.dt,
            dx_ref,
            args.nu,
            args.ref_cfl,
            args.ref_diff_cfl,
        )
        reference_step_sec = time.perf_counter() - t_ref0

        row["learned_step_sec"] = float(learned_step_sec)
        row["reference_step_sec"] = float(reference_step_sec)
        row["reference_substeps"] = int(ns)
        learned_rollout_time_sec += float(learned_step_sec)
        reference_rollout_time_sec += float(reference_step_sec)
        total_reference_substeps += int(ns)

        if args.log_every > 0 and (((step + 1) % args.log_every == 0) or (step + 1 == nsteps)):
            print(
                f"[learned-vs-fv] step={step+1:05d}/{nsteps:05d} "
                f"t={(step+1)*args.dt:.3f} rel={rows[-1]['rel_combined']:.3e} "
                f"learned_step={learned_step_sec:.4e}s ref_step={reference_step_sec:.4e}s",
                flush=True,
            )

    wall_loop_total_sec = time.perf_counter() - loop_t0
    final = rows[-1]
    rel_combined = [float(r["rel_combined"]) for r in rows]
    rel_u = [float(r["rel_u"]) for r in rows]
    rel_v = [float(r["rel_v"]) for r in rows]
    speed_ratio = None
    if reference_rollout_time_sec > 0.0:
        speed_ratio = learned_rollout_time_sec / reference_rollout_time_sec

    return {
        "history": rows,
        "final": final,
        "summary": {
            "max_rel_combined": float(max(rel_combined)),
            "mean_rel_combined": float(sum(rel_combined) / len(rel_combined)),
            "max_rel_u": float(max(rel_u)),
            "max_rel_v": float(max(rel_v)),
            "learned_rollout_time_sec": float(learned_rollout_time_sec),
            "reference_rollout_time_sec": float(reference_rollout_time_sec),
            "learned_div_reference_rollout_time": speed_ratio,
            "wall_loop_total_sec": float(wall_loop_total_sec),
            "total_reference_substeps": int(total_reference_substeps),
            "learned_primitive_absmax_seen": float(model.observed_absmax),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run bundle-based learned Burgers blocks on the inner-square problem and compare against the FV reference."
    )
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
    p.add_argument("--ref_N", type=int, required=True)
    p.add_argument("--ref_cfl", type=float, default=0.22)
    p.add_argument("--ref_diff_cfl", type=float, default=0.12)
    p.add_argument("--log_every", type=int, default=100)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    laplace_ckpt = Path(args.laplace_ckpt)
    burgers_ckpt = Path(args.burgers_ckpt)
    if not laplace_ckpt.exists():
        raise FileNotFoundError(f"Missing Laplace checkpoint: {laplace_ckpt}")
    if not burgers_ckpt.exists():
        raise FileNotFoundError(f"Missing transport checkpoint: {burgers_ckpt}")

    args.laplace_ckpt = str(laplace_ckpt)
    args.burgers_ckpt = str(burgers_ckpt)

    print("=" * 100, flush=True)
    print("Bundle learned-block rollout vs FV reference", flush=True)
    print("=" * 100, flush=True)
    print(
        f"[setup] K={args.K} rank={args.rank} h={args.inner_halfwidth} "
        f"tau_rel={args.tau_rel:.1e} dt={args.dt} T={args.T}",
        flush=True,
    )
    print(f"[ckpt] laplace={laplace_ckpt}", flush=True)
    print(f"[ckpt] transport={burgers_ckpt}", flush=True)

    build_t0 = time.perf_counter()
    _, learned_model, payload = k22mod.build_models(args)
    sync_device(learned_model.device)
    build_time_sec = time.perf_counter() - build_t0

    info = payload["info"]
    print(
        f"[basis] ambient_dim={info['meta_dim']} null_dim={info['null_info']['null_dim']} "
        f"rank_used={info['rank_used']}",
        flush=True,
    )
    print(
        f"[build] device={info['device']} dtype={info['dtype']} "
        f"build_time={build_time_sec:.6f}s memory={learned_model.memory_mib:.1f} MiB",
        flush=True,
    )

    result = run_learned_model_vs_fv(learned_model, payload["grid"], args)
    report = {
        "experiment": "bundle_learned_block_vs_fv_reference",
        "setup": vars(args),
        "basis_info": info,
        "operators": {
            "single_component_transport": "learned local-density primitive derivative table for uu_x and uu_y",
            "mixed_transport": "exact reduced projection for v*u_y and u*v_x",
            "laplace": "transferred learned ambient Laplace block",
            "same_outer_dt_for_learned_and_reference": True,
            "outer_nsteps": int(round(float(args.T) / float(args.dt))),
        },
        "artifacts": {
            "laplace_ckpt": laplace_ckpt,
            "burgers_ckpt": burgers_ckpt,
        },
        "learned_model": {
            "name": learned_model.name,
            "build_time_sec": float(build_time_sec),
            "memory_mib": float(learned_model.memory_mib),
            "device": str(learned_model.device),
            "dtype": str(learned_model.dtype),
        },
        "result": result,
    }

    summary_path = outdir / "summary.json"
    history_csv = outdir / "history.csv"
    summary_path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    write_csv(history_csv, result["history"])

    print("\n[final]", flush=True)
    print(json.dumps(to_jsonable(result["final"]), indent=2), flush=True)
    print("\n[summary]", flush=True)
    print(json.dumps(to_jsonable(result["summary"]), indent=2), flush=True)
    print(f"[done] wrote {summary_path} and {history_csv}", flush=True)


if __name__ == "__main__":
    main()
