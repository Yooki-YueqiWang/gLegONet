# Complete MMS-IV workflow

This folder documents an end-to-end, paper-scale reproduction of **MMS-IV: annular-star coupled reaction--diffusion**. It starts from ambient Laplace-block training, constructs the mixed-boundary matrices $C$ and $N_\Omega$, advances the reduced system with RK4, and checks the resulting dimensions, boundary residuals, and errors.

Run every command from the repository root. No configuration file is used: every numerical choice is visible in the command line.

## Problem

MMS-IV solves

```math
\begin{aligned}
u_t &= D_u\Delta u+a-u+u^2v+f_u^\Omega,\\
v_t &= D_v\Delta v+b-u^2v+f_v^\Omega,
\end{aligned}
```

on the annular-star domain

```math
R_{\mathrm{in}}(\theta)\leq \rho\leq R_{\mathrm{out}}(\theta),
```

with

```math
\begin{aligned}
R_{\mathrm{out}}(\theta)&=0.70+0.105\cos(5\theta)+0.040\sin(2\theta),\\
R_{\mathrm{in}}(\theta)&=0.23+0.055\cos(3\theta+\pi/5).
\end{aligned}
```

Both components satisfy homogeneous Dirichlet conditions on the outer boundary and homogeneous Neumann conditions on the inner boundary.

## Paper-scale settings

| Quantity | Value |
|---|---:|
| Fourier cutoff $K$ | 22 |
| Ambient dimension $M$ | 1517 |
| Outer Dirichlet samples | 420 |
| Inner Neumann samples | 300 |
| Total boundary samples $N_b$ | 720 |
| Requested maximum reduced rank | 1000 |
| Retained reduced rank $r$ | 832 |
| $D_u,D_v$ | 0.02, 0.012 |
| $a,b$ | 0.8, 0.6 |
| $A_u,A_v,\omega$ | 4.0, 3.2, 4.6 |
| Final time $T$ | 0.90 |
| RK4 step $\Delta t$ | $1.5\times10^{-3}$ |

The manuscript specifies $K$, the training-set sizes, optimizer learning rate and batch size, the boundary counts and retained dimension, the PDE coefficients, and the time interval. The remaining numerical controls shown below are the reference implementation controls used for this reproduction.

The distinction between `--reduced_rank 1000` and $r=832$ is intentional. The command-line value is an upper bound applied after the boundary null space is formed. Mass-matrix eigenvalue truncation removes unresolved directions, leaving the Figure 3 dimension $r=832$.

## 1. Install the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## 2. Train the ambient $K=22$ Laplace block

```bash
python training/laplace/train.py \
  --K 22 \
  --outdir artifacts/mms_iv/laplace_K22 \
  --epochs 80 \
  --n-train 20000 \
  --n-test 4000 \
  --batch-size 16 \
  --lr 0.0005 \
  --weight-decay 0.0 \
  --step-lr 40 \
  --gamma 0.3 \
  --init-scaled-diag 0.1 \
  --loss-mode sample_relative_mse \
  --loss-eps 1e-14 \
  --seed 123 \
  --device cuda \
  --dtype float64
```

This command uses the manuscript protocol of 20,000 independent training samples, 4,000 held-out samples, the per-sample relative supervised loss with $\varepsilon_{\mathrm{den}}=10^{-14}$, and Adam with learning rate $5\times10^{-4}$ and minibatches of 16. Float64 is used to resolve the diagonal spectrum accurately enough for the 600-step rollout. The trained checkpoint is written to `artifacts/mms_iv/laplace_K22/model_state.pt`.

Use `--device cpu` on a machine without CUDA; this changes the execution device, not the mathematical experiment.

## 3. Construct $C$ and $N_\Omega$, then roll out

```bash
python experiments/manufactured_solutions/mms_04_annular_star.py \
  --outdir artifacts/mms_iv/paper_run \
  --K 22 \
  --Nx_eval 120 \
  --Nb_outer 420 \
  --Nb_inner 300 \
  --Nb_dense 1600 \
  --reduced_rank 1000 \
  --tau_rel 1e-10 \
  --T 0.90 \
  --dt 0.0015 \
  --Du 0.02 \
  --Dv 0.012 \
  --a_param 0.8 \
  --b_param 0.6 \
  --amp_u 4.0 \
  --amp_v 3.2 \
  --omega 4.6 \
  --fit_lam 1e-11 \
  --seed 1234 \
  --device cuda \
  --log_every 50 \
  --field_time_stride 1 \
  --save_operators \
  --laplace_checkpoint artifacts/mms_iv/laplace_K22/model_state.pt
```

`--field_time_stride 1` writes every saved field to CSV. Set it to `0` when only the trajectory metrics and figures are needed; this reduces disk use and does not change the rollout.

### Boundary realization performed by the driver

Let $\Phi$ be the ambient real Fourier basis. The 420 outer value traces and 300 inner normal traces are stacked as

```math
C=
\begin{bmatrix}
\Phi(\mathbf{x}^{\mathrm{out}})\\
\mathbf{n}_{\mathrm{in}}^\mathsf{T}\nabla\Phi(\mathbf{x}^{\mathrm{in}})
\end{bmatrix}.
```

The driver computes a full SVD $C=U\Sigma V^\mathsf{T}$. With relative threshold $\tau_C=10^{-10}$, the numerical rank is 327 and the raw null dimension is 1190. If $Z$ contains the null-space columns of $V$, the domain mass matrix and null-space Gram matrix are

```math
M_\Omega=\Phi_\Omega^\mathsf{T}W_\Omega\Phi_\Omega,
\qquad
G=Z^\mathsf{T}M_\Omega Z.
```

The eigendecomposition of $G$, followed by the reference mass cutoff and the requested cap of 1000, gives

```math
N_\Omega=ZP\Lambda^{-1/2},
\qquad
CN_\Omega\approx0,
\qquad
N_\Omega^\mathsf{T}M_\Omega N_\Omega\approx I,
```

with 832 retained columns. Because the boundary data are homogeneous, the two coefficient vectors are simply

```math
\mathbf{a}_u=N_\Omega\mathbf{z}_u,
\qquad
\mathbf{a}_v=N_\Omega\mathbf{z}_v.
```

The initial exact fields are fitted in this admissible space. At each RK4 stage, the frozen Laplace block is evaluated in ambient coefficients, reaction and manufactured-forcing terms are assembled on the target quadrature, and the two right-hand sides are projected back into the same 832-dimensional coordinates.

With `--save_operators`, `boundary_operators.npz` contains `C`, `M_omega`, `N_omega`, the sampled boundary points, and inner-boundary normals for direct inspection.

## 4. Verify the completed run

```bash
python workflows/mms_iv/verify_run.py \
  --results artifacts/mms_iv/paper_run
```

The verifier checks the complete paper-scale parameter signature, the training metadata stored in the checkpoint, held-out block accuracy, finite output, $M=1517$, `rank(C)=327`, raw null dimension 1190, retained rank 832, the null-space and mass-orthonormality residuals, checkpoint compatibility, mixed-boundary residual, and agreement with the reported MMS-IV error scale.

The paper reports the following gLegONet values:

| Metric | Paper value |
|---|---:|
| Final relative $L^2$ error | $5.77\times10^{-3}$ |
| Mean relative $L^2$ error | $6.12\times10^{-3}$ |
| Maximum relative $L^2$ error | $7.70\times10^{-3}$ |
| RMS boundary residual | $4.51\times10^{-10}$ |
| Solution-scaled boundary residual | $1.93\times10^{-8}$ |

Small floating-point and hardware-dependent variation is expected. The verifier uses narrow numerical tolerances around the reported field errors and direct upper bounds for algebraic and boundary residuals.

## Outputs

The paper-scale run writes:

- `summary.json`: parameters, matrix dimensions, algebraic residuals, checkpoint diagnostics, and rollout errors;
- `table_metrics.json`: the metrics reported in the paper table;
- `boundary_operators.npz`: saved $C$, $M_\Omega$, and $N_\Omega$ when `--save_operators` is present;
- `error_curves.png`: relative field error and mixed-boundary residual;
- `snapshots_section3_vs_exact_u.png` and `snapshots_section3_vs_exact_v.png`: reference/prediction comparisons;
- `rollout_fields_and_relerr.csv`: time history and, when requested, pointwise fields.
