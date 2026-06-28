# Geometry-aware LegONet

This repository accompanies the paper **Geometry-aware LegONet for PDE Learning on Arbitrary Domains**.

Geometry-aware LegONet extends Lego-like operator learning from fixed-domain block composition to PDE learning on embedded target geometries. The method separates **reusable physical mechanisms** from **geometry-specific realization**. Neural mechanism blocks are pretrained once on a fixed ambient domain and then frozen. For each target domain, sampled boundary constraints define boundary-adapted coordinates, so changing the geometry or boundary operator changes only the algebraic realization layer, not the learned block parameters.

```math
\mathbf a(t)=\mathbf a_{\rm bc}(t)+N_\Omega \mathbf z(t).
```

Here `a_bc` is an affine boundary lift, `N_Omega` spans the admissible tangent space induced by the sampled boundary constraints, and `z` contains the reduced boundary-adapted coordinates. 

## Current code release

This release contains three parts:

1. `ambient_block_training/`  
   Scripts for pretraining and verifying the reusable ambient mechanism blocks.

2. `mms_benchmarks/`  
   Code for the five manufactured-solution tests used to validate boundary transfer across complex geometries and boundary operators.

3. `physical_law_discovery/`  
   Code for the boundary-adapted sparse physical-law discovery experiments on unseen domains.

Other numerical experiments in the paper, including the additional physical dynamics and stress-test solvers, will be released after the paper is accepted.

## Repository structure

```text
gLegONet/
├── README.md
├── requirements.txt
├── ambient_block_training/
│   ├── README.md
│   ├── train_laplace_block.py
│   ├── train_burgers_local_density_K22_plain_mlp_fixed_v2.py
│   ├── verify_trained_ambient_blocks.py
│   ├── verify_local_density_burgers_block_bins.py
│   ├── run_laplace_k22_style_extension.py
│   ├── run_laplace_corrected_k_ablation.py
│   ├── run_local_density_k22_style_extension.py
│   ├── checkpoints/
│   └── outputs/
├── mms_benchmarks/
│   ├── README.md
│   ├── configs/
│   ├── run_mms1_rosette_dirichlet.py
│   ├── run_mms2_crescent_neumann.py
│   ├── run_mms3_bunny_robin.py
│   ├── run_mms4_annular_star_mixed.py
│   ├── run_mms5_pinwheel_burgers.py
│   ├── run_all_mms.py
│   └── outputs/
└── physical_law_discovery/
    ├── README.md
    ├── configs/
    ├── run_peanut_law_discovery.py
    ├── run_channel_law_discovery.py
    ├── run_all_law_discovery.py
    └── outputs/
```


## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

Core dependencies are:

- `torch`;
- `numpy`;
- `scipy`;
- `matplotlib`.

A CUDA-capable GPU is recommended for block pretraining and large diagnostic batches. The verification scripts can also be run on CPU with smaller test sizes.

## Method overview

All target domains are embedded into the ambient square `Q = [-1,1]^2`. The ambient coefficient space is a truncated real Fourier space. For a cutoff `K`, the basis contains the constant mode and paired sine/cosine modes selected by a radial frequency cutoff. For `K = 22`, the ambient coefficient dimension is `M = 1517`.

A target domain is handled by a deterministic boundary realization step. Boundary samples produce a linear constraint system,

```math
C\mathbf a=\mathbf d(t),
```

and the admissible coefficient vector is represented as

```math
\mathbf a(t)=\mathbf a_{\rm bc}(t)+N_\Omega\mathbf z(t).
```

The frozen mechanism blocks are then evaluated at the admissible state and projected into the target-domain Galerkin space. Thus the neural parameters remain fixed, while the matrices defining the boundary condition, quadrature, mass metric, and reduced coordinates are rebuilt for each domain.

## Part I: ambient block training

The folder `ambient_block_training/` contains the code used to train and verify the reusable ambient block library.

The released scripts cover:

- a dissipative diffusion block trained by instantaneous ambient-response matching;
- a local-density transport block used to realize the directional quadratic transport mechanisms;
- held-out diagnostics for pretrained checkpoints;
- optional resolution studies for changing the ambient cutoff `K`.

The README in `ambient_block_training/` gives the detailed command-line usage. The main scripts are:

```text
ambient_block_training/train_laplace_block.py
ambient_block_training/train_burgers_local_density_K22_plain_mlp_fixed_v2.py
ambient_block_training/verify_trained_ambient_blocks.py
ambient_block_training/verify_local_density_burgers_block_bins.py
```

Example diffusion-block training command:

```bash
python ambient_block_training/train_laplace_block.py \
  --K 22 \
  --epochs 80 \
  --n-train 20000 \
  --n-test 4000 \
  --batch-size 128 \
  --device cuda \
  --outdir ambient_block_training/outputs/runs_ambient_laplace_K22
```

Example transport-density training command:

```bash
python ambient_block_training/train_burgers_local_density_K22_plain_mlp_fixed_v2.py \
  --K 22 \
  --n-grid 96 \
  --epochs 1500 \
  --steps-per-epoch 200 \
  --batch-size 8192 \
  --device cuda \
  --eval-every 5 \
  --outdir ambient_block_training/outputs/runs_ambient_burgers_K22_local_density
```

The training scripts internally construct the analytic ambient reference responses needed for supervision. 

### Checkpoint layout

Pretrained checkpoints should be placed under:

```text
ambient_block_training/checkpoints/
├── ambient_laplace_K22/
│   └── model_state.pt
└── ambient_burgers_K22_local_density/
    └── model_state.pt
```

Combined held-out diagnostics can be run by:

```bash
python ambient_block_training/verify_trained_ambient_blocks.py \
  --laplace-ckpt ambient_block_training/checkpoints/ambient_laplace_K22/model_state.pt \
  --burgers-ckpt ambient_block_training/checkpoints/ambient_burgers_K22_local_density/model_state.pt \
  --outdir ambient_block_training/outputs/block_test_diagnostics \
  --device cuda \
  --n-test 4096
```


## Part II: five MMS benchmarks

The folder `mms_benchmarks/` contains the five manufactured-solution benchmarks used to validate boundary transfer independently of reference-solver error.

The five tests are:

| Test | Domain | Mechanism class | Boundary condition |
|---|---|---|---|
| MMS-I | Rosette | scalar reaction--diffusion | homogeneous Dirichlet |
| MMS-II | Crescent | scalar reaction--diffusion | homogeneous Neumann |
| MMS-III | Bunny | logistic reaction--diffusion | nonhomogeneous Robin |
| MMS-IV | Annular star | coupled reaction--diffusion | mixed Dirichlet--Neumann |
| MMS-V | Pinwheel shell | scalar Burgers-type transport--diffusion | homogeneous Dirichlet |

These tests use the same frozen ambient library and rebuild only the target-domain objects: boundary constraint matrix, affine lift, mass matrix, quadrature, and boundary-adapted basis. They are designed to test different boundary operators, nonconvex or multiply structured geometries, scalar and coupled systems, and nonlinear transport.

Suggested commands:

```bash
python mms_benchmarks/run_mms1_rosette_dirichlet.py
python mms_benchmarks/run_mms2_crescent_neumann.py
python mms_benchmarks/run_mms3_bunny_robin.py
python mms_benchmarks/run_mms4_annular_star_mixed.py
python mms_benchmarks/run_mms5_pinwheel_burgers.py
```

or run all five benchmarks:

```bash
python mms_benchmarks/run_all_mms.py
```

Each benchmark writes its figures, error histories, boundary residuals, and configuration files to `mms_benchmarks/outputs/`.

## Part III: boundary-adapted physical-law discovery

The folder `physical_law_discovery/` contains the inverse-problem experiments from the paper. The goal is to identify an unknown interior law on a known target geometry using sparse interior observations over a short identification window.

The geometry and boundary condition are encoded first through the same boundary-adapted coordinates used in forward simulation. Candidate mechanisms are then realized as reduced responses in the admissible space. A sparse regression step selects active mechanisms and estimates their coefficients; the recovered law is rolled out in the same boundary-constrained coordinates.

The released discovery experiments include:

- a peanut-shaped dumbbell domain, which tests reaction--diffusion through a narrow bottleneck;
- an oblique sinusoidal channel, which tests nonlinear transport in a thin anisotropic geometry.

Suggested commands:

```bash
python physical_law_discovery/run_peanut_law_discovery.py
python physical_law_discovery/run_channel_law_discovery.py
```

or run both inverse problems:

```bash
python physical_law_discovery/run_all_law_discovery.py
```

The outputs include recovered coefficients, selected supports, rollout errors, diagnostic plots, and configuration files.

## Reproducibility notes

The initial public release is intended to make the frozen ambient library, MMS validation, and physical-law discovery pipeline inspectable and reproducible. To reproduce a result, keep the following files together whenever possible:

```text
model_state.pt
config.json
history.csv
final_metrics.json
```

Intermediate run folders, temporary figures, and large raw arrays should not be committed unless they are part of a documented release artifact.

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{zhang2026geometryawarelegonet,
  title  = {Geometry-aware LegONet for PDE Learning on Arbitrary Domains},
  author = {Zhang, Jiahao and Wang, Yueqi and Lin, Guang},
  year   = {2026}
}
```

## License

Add the project license before making the repository public.
