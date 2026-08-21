#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume-constrained Allen--Cahn on a disk: FEM timing and Exact-S3 vs Learned-S3.

This script is intentionally a comparison driver, not a new method.  It uses the
same disk geometry, Section-3 null-space construction, initial condition, FEM
reference, and time integrator for both reduced solvers.  The only difference is
which ambient-square Laplace block is pulled back to the disk:

  Exact-S3:   analytic ambient Laplace symbol, Delta phi = -pi^2 |k|^2 phi.
  Learned-S3: pretrained positive diagonal neural Laplace block.

Metrics written to CSV/JSON:
  - FEM reference rollout time.
  - Exact-S3 rollout time.
  - Learned-S3 rollout time.
  - final and sampled-mean weighted relative L2 vs FEM.
  - final and sampled-mean weighted relative L2 between Exact-S3 and Learned-S3.
  - max mass drift, max dense Neumann residual, energy violation count.

Expected local dependency:
  vcac_disk_common.py in the same directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as cm


def parse_list_int(s: str) -> List[int]:
    return [int(x) for x in s.replace(',', ' ').split() if x.strip()]


def max_mass_drift(means: np.ndarray) -> float:
    return float(np.max(np.abs(means - means[0]))) if means.size else 0.0


def energy_violation_count(energies: np.ndarray, tol: float = 1.0e-10) -> int:
    if energies.size <= 1:
        return 0
    return int(np.sum(np.diff(energies) > tol))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def latex_sci(x: Any) -> str:
    """Format a scalar for compact LaTeX table output."""
    if x is None:
        return "--"
    try:
        xf = float(x)
    except Exception:
        return "--"
    if not np.isfinite(xf):
        return "--"
    if abs(xf) < 1.0e-300:
        return r"0"
    e = int(np.floor(np.log10(abs(xf))))
    m = xf / (10 ** e)
    if round(abs(m), 2) >= 10.0:
        m /= 10.0
        e += 1
    return rf"{m:.2f}{{\times}}10^{{{e}}}"


def mass_drift_series(means: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    if means.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.abs(means - means[0])


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Return the first available key from a possibly version-dependent info dict."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def extract_rank_info(problem: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Collect the Section-3 rank information in a stable, table-friendly format."""
    info = problem.get("infoN", {})
    if not isinstance(info, dict):
        info = {}
    N = np.asarray(problem["N"])
    basis_dim = int(len(problem.get("meta", [])))
    actual_rank = int(N.shape[1])
    requested_rank = int(args.reduced_rank)

    constraint_rank = _first_present(
        info,
        ["constraint_rank", "rank_C", "C_rank", "rank", "rank_boundary", "rank_build"],
        None,
    )
    null_dim_raw = _first_present(
        info,
        ["raw_null_dim", "null_dim_raw", "null_dim", "nullity", "full_null_dim"],
        None,
    )
    if null_dim_raw is None and constraint_rank is not None:
        try:
            null_dim_raw = basis_dim - int(constraint_rank)
        except Exception:
            null_dim_raw = None

    out = {
        "basis_dim": basis_dim,
        "requested_reduced_rank": requested_rank,
        "actual_reduced_rank": actual_rank,
        "rank": actual_rank,
        "constraint_rank": None if constraint_rank is None else int(constraint_rank),
        "raw_null_dim": None if null_dim_raw is None else int(null_dim_raw),
        "Nb_build": int(args.Nb_build),
        "Nb_dense": int(args.Nb_dense),
        "tau_rel": float(args.tau_rel),
    }
    # Preserve the original info dict so that no solver-specific diagnostics are lost.
    out["basis_info_raw"] = info
    return out


def write_rank_info(outdir: str, rank_info: Dict[str, Any]) -> None:
    path = os.path.join(outdir, "rank_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cm.to_jsonable(rank_info), f, indent=2)
    print("\n[rank_info]")
    print(json.dumps(cm.to_jsonable(rank_info), indent=2))
    print(f"[rank output] wrote {path}")


def build_table_metrics(
    rows: List[Dict[str, Any]],
    histories: Dict[str, Dict[str, np.ndarray]],
    fem_rollout_time: float,
    fem_setup_time: float,
    fem_interp_time: float,
    setup_time: float,
    rank_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert detailed reduced-solver histories to the common table format.

    The downstream AC table uses:
      train time       -> not applicable for Exact-S3/Learned-S3 in this driver;
      rank             -> the actual reduced dimension dim(z)=N.shape[1];
      final/mean/max   -> weighted relative L2 versus FEM;
      boundary column  -> final dense Neumann RMS residual;
      auxiliary column -> maximum mass drift.
    Extra fields are retained for diagnostics and timing analysis.
    """
    table: List[Dict[str, Any]] = []
    fem_total = float(fem_setup_time + fem_rollout_time + fem_interp_time)
    for row in rows:
        name = str(row["method"])
        h = histories[name]
        rel = np.asarray(h["rel_vs_fem"], dtype=np.float64)
        bc = np.asarray(h["bc"], dtype=np.float64)
        mass_drift = mass_drift_series(np.asarray(h["mean"], dtype=np.float64))
        energy = np.asarray(h["energy"], dtype=np.float64)

        # Keep the same convention as the AC FNO/UNO/PINN scripts:
        # the reported mean error excludes t=0, while max includes all sampled times.
        mean_rel = float(np.mean(rel[1:])) if rel.size > 1 else float(np.mean(rel))
        mean_bc = float(np.mean(bc[1:])) if bc.size > 1 else float(np.mean(bc))
        mean_mass = float(np.mean(mass_drift[1:])) if mass_drift.size > 1 else float(np.mean(mass_drift))

        table.append({
            "method": name,
            "params": 0,
            "train_time_sec": None,
            "rank": int(rank_info["actual_reduced_rank"]),
            "actual_reduced_rank": int(rank_info["actual_reduced_rank"]),
            "requested_reduced_rank": int(rank_info["requested_reduced_rank"]),
            "basis_dim": int(rank_info["basis_dim"]),
            "constraint_rank": rank_info.get("constraint_rank"),
            "raw_null_dim": rank_info.get("raw_null_dim"),
            "rollout_time_sec": float(row["wall_time_rollout_sec"]),
            "total_s3_time_sec": float(setup_time + row["wall_time_rollout_sec"]),
            "fem_reference_time_sec": fem_total,
            "fem_rollout_time_sec": float(fem_rollout_time),
            "speedup_vs_fem_rollout": float(row["speedup_vs_fem_rollout"]),
            "final_rel_l2": float(rel[-1]),
            "mean_rel_l2": mean_rel,
            "max_rel_l2": float(np.max(rel)),
            "final_bc_rms": float(bc[-1]),
            "mean_bc_rms": mean_bc,
            "max_bc_rms": float(np.max(bc)),
            "final_mass_drift": float(mass_drift[-1]) if mass_drift.size else 0.0,
            "mean_mass_drift": mean_mass,
            "max_mass_drift": float(np.max(mass_drift)) if mass_drift.size else 0.0,
            "energy_violation_count": int(row["energy_violation_count"]),
            "final_energy_relative_gap_vs_fem": float(row["final_energy_relative_gap_vs_fem"]),
            "initial_energy": float(energy[0]) if energy.size else 0.0,
            "final_energy": float(energy[-1]) if energy.size else 0.0,
            "final_rel_l2_vs_exact_s3": float(row["final_relL2_vs_exact_s3"]),
            "mean_rel_l2_vs_exact_s3": float(row["mean_relL2_vs_exact_s3"]),
        })
    return table


def write_table_outputs(outdir: str, table_rows: List[Dict[str, Any]]) -> None:
    """Write table_metrics.{json,csv} and direct LaTeX row files."""
    json_path = os.path.join(outdir, "table_metrics.json")
    csv_path = os.path.join(outdir, "table_metrics.csv")
    tex_path = os.path.join(outdir, "latex_table_rows.txt")
    tex_rank_path = os.path.join(outdir, "latex_table_rows_with_rank.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cm.to_jsonable(table_rows), f, indent=2)
    write_csv(csv_path, table_rows)

    # Original downstream table layout, without a Rank column.
    with open(tex_path, "w", encoding="utf-8") as f:
        for r in table_rows:
            latex = (
                rf"& {r['method']} & -- "
                rf"& {latex_sci(r['final_rel_l2'])} "
                rf"& {latex_sci(r['mean_rel_l2'])} "
                rf"& {latex_sci(r['max_rel_l2'])} "
                rf"& {latex_sci(r['final_bc_rms'])} "
                rf"& {latex_sci(r['max_mass_drift'])} \\"
            )
            f.write(latex + "\n")

    # Extended layout for tables that include a Rank column after Method.
    with open(tex_rank_path, "w", encoding="utf-8") as f:
        for r in table_rows:
            latex = (
                rf"& {r['method']} & {int(r['rank'])} & -- "
                rf"& {latex_sci(r['final_rel_l2'])} "
                rf"& {latex_sci(r['mean_rel_l2'])} "
                rf"& {latex_sci(r['max_rel_l2'])} "
                rf"& {latex_sci(r['final_bc_rms'])} "
                rf"& {latex_sci(r['max_mass_drift'])} \\"
            )
            f.write(latex + "\n")

    print("\n[table_metrics]")
    print(json.dumps(cm.to_jsonable(table_rows), indent=2))
    print("\n[latex_table_rows]")
    with open(tex_path, "r", encoding="utf-8") as f:
        print(f.read().rstrip())
    print("\n[latex_table_rows_with_rank]")
    with open(tex_rank_path, "r", encoding="utf-8") as f:
        print(f.read().rstrip())
    print(f"[table output] wrote {json_path}, {csv_path}, {tex_path}, {tex_rank_path}")


def plot_histories(outdir: str, tvals: np.ndarray, histories: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Plot comparison curves.

    The FEM entry only contains energy/mean histories, while reduced solvers
    contain rel_vs_fem/energy/mean/bc.  Therefore the relative-error plot must
    skip entries without rel_vs_fem.
    """
    reduced_names = [name for name, h in histories.items() if 'rel_vs_fem' in h]

    plt.figure(figsize=(7.2, 4.6))
    for name in reduced_names:
        h = histories[name]
        plt.semilogy(tvals, h['rel_vs_fem'], marker='o', markersize=3, label=name)
    plt.xlabel('time')
    plt.ylabel('weighted rel-L2 vs FEM')
    plt.title('Exact-S3 vs Learned-S3')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'relL2_exact_vs_learned.png'), dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    for name in reduced_names:
        h = histories[name]
        plt.plot(tvals, h['energy'], marker='o', markersize=3, label=name)
    if 'FEM' in histories and 'energy' in histories['FEM']:
        plt.plot(tvals, histories['FEM']['energy'], marker='s', markersize=3, label='FEM')
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.title('Energy histories')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'energy_exact_vs_learned.png'), dpi=220)
    plt.close()

    if 'Learned-S3' in histories and 'exact_learned_rel' in histories['Learned-S3']:
        plt.figure(figsize=(7.2, 4.6))
        plt.semilogy(tvals, histories['Learned-S3']['exact_learned_rel'], marker='o', markersize=3)
        plt.xlabel('time')
        plt.ylabel('weighted rel-L2, Learned-S3 vs Exact-S3')
        plt.title('Learned block rollout gap')
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'learned_vs_exact_gap.png'), dpi=220)
        plt.close()

def build_problem(args: argparse.Namespace) -> Dict[str, Any]:
    radius = 0.5 * args.disk_diameter
    meta = cm.build_trig_basis_meta(args.K, radial_truncation=True)
    grid_pack = cm.build_disk_in_box_grid(args.Nx_eval, args.box_halfwidth, radius)
    x_in, y_in, w = grid_pack['x_in'], grid_pack['y_in'], grid_pack['weight']
    Phi = cm.eval_trig_basis_from_meta(x_in, y_in, meta, args.box_halfwidth, args.box_halfwidth)
    dPhix, dPhiy = cm.eval_trig_basis_grad_from_meta(x_in, y_in, meta, args.box_halfwidth, args.box_halfwidth)
    M_dom, K_dom = cm.assemble_domain_mass_stiffness(Phi, dPhix, dPhiy, w)

    xb, yb, nxb, nyb = cm.circle_boundary_points(args.Nb_build, radius)
    xd, yd, nxd, nyd = cm.circle_boundary_points(args.Nb_dense, radius)
    dphix_b, dphiy_b = cm.eval_trig_basis_grad_from_meta(xb, yb, meta, args.box_halfwidth, args.box_halfwidth)
    dphix_d, dphiy_d = cm.eval_trig_basis_grad_from_meta(xd, yd, meta, args.box_halfwidth, args.box_halfwidth)
    C_build = nxb[:, None] * dphix_b + nyb[:, None] * dphiy_b
    C_dense = nxd[:, None] * dphix_d + nyd[:, None] * dphiy_d
    N, infoN = cm.build_section3_basis(C_build, M_dom, args.tau_rel, args.reduced_rank)

    Phi_r = Phi @ N
    Mr = N.T @ M_dom @ N
    Kr = N.T @ K_dom @ N
    z0, u0_cont, z0_info = cm.construct_reduced_space_interface_z0(
        Phi_r, Mr, Kr, grid_pack['X'], grid_pack['Y'], grid_pack['mask'], w,
        ic_amp=args.ic_amp, bend=args.interface_bend, nmodes=args.smooth_mode_count, seed=args.seed,
    )
    return dict(radius=radius, meta=meta, grid_pack=grid_pack, x_in=x_in, y_in=y_in, w=w,
                Phi=Phi, M_dom=M_dom, K_dom=K_dom, C_dense=C_dense,
                N=N, infoN=infoN, z0=z0, u0_cont=u0_cont, z0_info=z0_info)


def run_reduced(name: str, problem: Dict[str, Any], args: argparse.Namespace,
                fem_grid_samples: Dict[int, np.ndarray], sample_steps: set[int],
                laplace_mass_op_r: np.ndarray | None, laplace_label: str) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    res = cm.rollout_reduced_method_sampled(
        problem['N'], problem['Phi'], problem['K_dom'], problem['M_dom'], problem['C_dense'], problem['grid_pack'], problem['z0'],
        args.eps2, args.dt, args.T, sample_steps, args.log_every, args.react_substeps,
        fem_grid_samples, laplace_mass_op_r=laplace_mass_op_r, laplace_label=laplace_label,
    )
    elapsed = time.perf_counter() - t0
    print(f"[{name}] rollout wall time = {elapsed:.6f} s")
    return res, elapsed


def collect_history(res: Dict[str, Any], fem_ref: Dict[str, Any], times_sampled: np.ndarray, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    rel = np.array([res['rel_samples'][int(s)] for s in times_sampled], dtype=float)
    energy = np.array([res['energy_samples'][int(s)] for s in times_sampled], dtype=float)
    mean = np.array([res['mean_samples'][int(s)] for s in times_sampled], dtype=float)
    bc = np.array([res['bc_samples'][int(s)] for s in times_sampled], dtype=float)
    return {'rel_vs_fem': rel, 'energy': energy, 'mean': mean, 'bc': bc}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, required=True)
    p.add_argument('--box_halfwidth', type=float, required=True)
    p.add_argument('--disk_diameter', type=float, required=True)
    p.add_argument('--K', type=int, required=True)
    p.add_argument('--Nb_build', type=int, required=True)
    p.add_argument('--Nb_dense', type=int, required=True)
    p.add_argument('--tau_rel', type=float, default=1e-10)
    p.add_argument('--reduced_rank', type=int, required=True)
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
    p.add_argument('--laplace_block_path', type=str, required=True)
    p.add_argument('--laplace_block_sign', type=str, default='auto', choices=['auto', 'positive_stiffness', 'laplace_symbol'])
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cm.set_seed(args.seed)

    setup_t0 = time.perf_counter()
    problem = build_problem(args)
    setup_time = time.perf_counter() - setup_t0
    rank_info = extract_rank_info(problem, args)
    write_rank_info(args.outdir, rank_info)
    cm.save_disk_geometry(problem['grid_pack'], args.Nb_build, os.path.join(args.outdir, 'disk_geometry.png'))

    nsteps = int(round(args.T / args.dt))
    snap_steps = sorted(set([0, nsteps // 4, nsteps // 2, (3 * nsteps) // 4, nsteps]))
    log_steps = set(range(0, nsteps + 1, args.log_every)); log_steps.add(nsteps)
    sample_steps = set(snap_steps) | log_steps | {0}
    times_sampled = np.array(sorted(sample_steps), dtype=np.int64)
    tvals = times_sampled.astype(float) * args.dt

    # FEM reference, with timing split into setup/projection/rollout/interpolation.
    fem_setup_t0 = time.perf_counter()
    mesh = cm.generate_circle_mesh(problem['radius'], args.fem_boundary_nodes, args.fem_interior_nodes, args.seed)
    u0_nodes = cm.fem_l2_project_initial_condition(mesh, problem['u0_cont'], problem['grid_pack'])
    fem_setup_time = time.perf_counter() - fem_setup_t0
    fem_roll_t0 = time.perf_counter()
    fem_ref = cm.fem_strang_rollout_sampled(mesh, u0_nodes, args.eps2, args.dt, args.T, sample_steps)
    fem_rollout_time = time.perf_counter() - fem_roll_t0
    interp_t0 = time.perf_counter()
    fem_grid_samples = {s: cm.fem_interpolate_to_grid(fem_ref['points'], fem_ref['tris'], u, problem['x_in'], problem['y_in'])
                        for s, u in fem_ref['node_samples'].items()}
    fem_interp_time = time.perf_counter() - interp_t0
    print(f"[FEM timing] setup/projection={fem_setup_time:.6f}s | rollout={fem_rollout_time:.6f}s | interp={fem_interp_time:.6f}s")

    # Exact ambient-square analytic Laplace block pulled back to disk.
    lam_exact = cm.exact_laplace_symbol_from_meta(problem['meta'], args.box_halfwidth, args.box_halfwidth)
    Lr_exact_symbol = cm.build_reduced_laplace_mass_from_symbol(problem['N'], problem['M_dom'], lam_exact)
    exact_res, exact_time = run_reduced('Exact-S3', problem, args, fem_grid_samples, sample_steps,
                                        Lr_exact_symbol, 'analytic ambient-square Laplace symbol')

    lam_learn, learn_info = cm.load_learned_laplace_symbol(
        args.laplace_block_path, problem['meta'], args.box_halfwidth, args.box_halfwidth,
        sign_convention=args.laplace_block_sign,
    )
    Lr_learn = cm.build_reduced_laplace_mass_from_symbol(problem['N'], problem['M_dom'], lam_learn)
    learned_res, learned_time = run_reduced('Learned-S3', problem, args, fem_grid_samples, sample_steps,
                                            Lr_learn, f'pretrained neural Laplace block: {args.laplace_block_path}')

    histories: Dict[str, Dict[str, np.ndarray]] = {}
    histories['Exact-S3'] = collect_history(exact_res, fem_ref, times_sampled, args)
    histories['Learned-S3'] = collect_history(learned_res, fem_ref, times_sampled, args)
    histories['FEM'] = {'energy': np.array([fem_ref['energy_samples'][int(s)] for s in times_sampled]),
                        'mean': np.array([fem_ref['mean_samples'][int(s)] for s in times_sampled])}
    exact_learned_rel = []
    for s in times_sampled:
        s_int = int(s)
        exact_learned_rel.append(cm.weighted_rel_l2(learned_res['field_samples'][s_int], exact_res['field_samples'][s_int], problem['w']))
    histories['Learned-S3']['exact_learned_rel'] = np.array(exact_learned_rel)

    plot_histories(args.outdir, tvals, histories)
    snap_times = {k: k * args.dt for k in snap_steps}
    cm.save_snapshot_compare(problem['grid_pack'], {k: exact_res['field_samples'][k] for k in snap_steps},
                             {k: fem_grid_samples[k] for k in snap_steps}, 'Exact-S3', 'FEM', snap_times,
                             os.path.join(args.outdir, 'snapshots_exact_vs_fem.png'))
    cm.save_snapshot_compare(problem['grid_pack'], {k: learned_res['field_samples'][k] for k in snap_steps},
                             {k: fem_grid_samples[k] for k in snap_steps}, 'Learned-S3', 'FEM', snap_times,
                             os.path.join(args.outdir, 'snapshots_learned_vs_fem.png'))

    rows = []
    for name, res, wall in [('Exact-S3', exact_res, exact_time), ('Learned-S3', learned_res, learned_time)]:
        h = histories[name]
        rel = np.asarray(h['rel_vs_fem'], dtype=float)
        bc = np.asarray(h['bc'], dtype=float)
        mass_drift = mass_drift_series(np.asarray(h['mean'], dtype=float))
        final_energy_gap = abs(h['energy'][-1] - histories['FEM']['energy'][-1]) / (abs(histories['FEM']['energy'][-1]) + 1e-14)
        rows.append({
            'method': name,
            'wall_time_rollout_sec': wall,
            'speedup_vs_fem_rollout': fem_rollout_time / max(wall, 1e-14),
            'final_relL2_vs_fem': float(rel[-1]),
            'sampled_mean_relL2_vs_fem_excluding_t0': float(np.mean(rel[1:])) if rel.size > 1 else float(np.mean(rel)),
            'max_relL2_vs_fem': float(np.max(rel)),
            'final_mass_drift': float(mass_drift[-1]) if mass_drift.size else 0.0,
            'mean_mass_drift_excluding_t0': float(np.mean(mass_drift[1:])) if mass_drift.size > 1 else float(np.mean(mass_drift)),
            'max_mass_drift': float(np.max(mass_drift)) if mass_drift.size else 0.0,
            'final_dense_neumann_residual': float(bc[-1]),
            'mean_dense_neumann_residual_excluding_t0': float(np.mean(bc[1:])) if bc.size > 1 else float(np.mean(bc)),
            'max_dense_neumann_residual': float(np.max(bc)),
            'energy_violation_count': energy_violation_count(h['energy']),
            'final_energy_relative_gap_vs_fem': float(final_energy_gap),
            'final_relL2_vs_exact_s3': 0.0 if name == 'Exact-S3' else float(histories['Learned-S3']['exact_learned_rel'][-1]),
            'mean_relL2_vs_exact_s3': 0.0 if name == 'Exact-S3' else float(np.mean(histories['Learned-S3']['exact_learned_rel'][1:])),
        })
    write_csv(os.path.join(args.outdir, 'summary_metrics.csv'), rows)

    table_rows = build_table_metrics(rows, histories, fem_rollout_time, fem_setup_time, fem_interp_time, setup_time, rank_info)
    write_table_outputs(args.outdir, table_rows)

    history_rows = []
    for i, s in enumerate(times_sampled):
        row = {'step': int(s), 'time': float(tvals[i])}
        row['fem_energy'] = float(histories['FEM']['energy'][i])
        row['fem_mean'] = float(histories['FEM']['mean'][i])
        for name in ['Exact-S3', 'Learned-S3']:
            row[f'{name}_relL2_vs_fem'] = float(histories[name]['rel_vs_fem'][i])
            row[f'{name}_energy'] = float(histories[name]['energy'][i])
            row[f'{name}_mean'] = float(histories[name]['mean'][i])
            row[f'{name}_bc'] = float(histories[name]['bc'][i])
        row['Learned_vs_Exact_relL2'] = float(histories['Learned-S3']['exact_learned_rel'][i])
        history_rows.append(row)
    write_csv(os.path.join(args.outdir, 'sampled_histories.csv'), history_rows)

    summary = {
        'args': vars(args),
        'basis_info': cm.to_jsonable(problem['infoN']),
        'rank_info': cm.to_jsonable(rank_info),
        'z0_info': cm.to_jsonable(problem['z0_info']),
        'learned_block_info': cm.to_jsonable(learn_info),
        'timing': {
            'problem_setup_sec': setup_time,
            'fem_setup_projection_sec': fem_setup_time,
            'fem_rollout_sec': fem_rollout_time,
            'fem_interpolation_sec': fem_interp_time,
            'exact_s3_rollout_sec': exact_time,
            'learned_s3_rollout_sec': learned_time,
        },
        'metrics': rows,
        'table_metrics': table_rows,
    }
    with open(os.path.join(args.outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(cm.to_jsonable(summary), f, indent=2)
    print(json.dumps(cm.to_jsonable(summary['timing']), indent=2))
    print('[done] wrote', args.outdir)


if __name__ == '__main__':
    main()

# python run.py --K 22 --Nx_eval 301 --Nb_build 1600 --Nb_dense 6400 --reduced_rank 412 --eps2 5e-3 --T 4.0 --dt 5e-4 --react_substeps 2 --fem_boundary_nodes 900 --fem_interior_nodes 20000
