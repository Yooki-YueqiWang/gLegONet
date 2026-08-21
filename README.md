# Geometry-aware LegONet (gLegONet)

Official research code for **Geometry-aware LegONet for PDE Learning on Arbitrary Domains** by Jiahao Zhang, Yueqi Wang, and Guang Lin.

gLegONet separates reusable physical mechanisms from target geometry. Diffusion and directional transport blocks are pretrained once on the ambient square (Q=[-1,1]^2). A deterministic boundary-adapted Galerkin interface then realizes those frozen blocks on an unseen embedded domain without geometry-specific neural retraining.

> **Release status.** This repository is a pre-acceptance code release. It contains block training, the five manufactured-solution benchmarks, volume-constrained Allen--Cahn, vector Burgers, and sparse physical-law discovery. Checkpoints, generated run artifacts, and external datasets are not distributed. Cylinder-wake and clamped Swift--Hohenberg code will be released after paper acceptance.

## Method

The target PDE is decomposed into reusable mechanisms,

$$
\partial_t u(\mathbf{x},t)
=
\sum_{i=1}^{N_{\mathrm{blk}}} c_i L_i^\Omega(u)(\mathbf{x},t).
$$

The ambient representation is the normalized real Fourier space

$$
\mathcal{V}_K(Q)=\operatorname{span}\!\left(
\{1\}\cup
\left\{\sqrt{2}\cos\bigl(\pi(kx+\ell y)\bigr),
\sqrt{2}\sin\bigl(\pi(kx+\ell y)\bigr):(k,\ell)\in\mathcal{I}_K^+\right\}
\right),
$$

where \(\mathcal{I}_K=\{(k,\ell)\in\mathbb{Z}^2:k^2+\ell^2\le K^2\}\). For (K=22), the ambient coefficient dimension is (M=1517).

For each new domain, sampled boundary conditions define

$$
C\mathbf{a}(t)=\mathbf{d}(t),
\qquad
\mathbf{a}(t)=\mathbf{a}_{\mathrm{bc}}(t)+N_\Omega\mathbf{z}(t),
$$

with (CN_\Omega\approx0) and (N_\Omega^\top M_\Omega N_\Omega=I). The reduced dynamics are

$$
\dot{\mathbf{z}}(t)
=
\sum_{i=1}^{N_{\mathrm{blk}}}c_i\,\mathbf{q}_i^\Omega(\mathbf{z},t)
-N_\Omega^\top M_\Omega\dot{\mathbf{a}}_{\mathrm{bc}}(t).
$$

The complete workflow is:

1. **Pretrain:** learn the diffusion response and the local transport density on (Q).
2. **Freeze:** save one resolution-matched mechanism library for each Fourier cutoff (K).
3. **Realize:** sample the target boundary, construct the affine lift and numerical null space, and mass-orthonormalize the admissible coordinates.
4. **Assemble:** project each frozen mechanism through the same target-domain Galerkin interface.
5. **Roll out or identify:** integrate directly in (mathbf{z}), or regress candidate mechanism coefficients from short-time sensor observations.

Boundary conditions are therefore part of the coordinates, not a penalty term or a post-step correction.

## Released experiments

| Experiment | Geometry / boundary condition | Manuscript configuration |
|---|---|---|
| MMS-I | Rosette / homogeneous Dirichlet | (K=22), (N_b=420), (r=207), (T=1.0), (Delta t=1.5\times10^{-3}) |
| MMS-II | Crescent / homogeneous Neumann | (K=22), (N_b=620), (r=220), (T=1.0), (Delta t=2\times10^{-3}) |
| MMS-III | Bunny / nonhomogeneous Robin | (K=22), (N_b=420), (r=843), (T=0.9), (Delta t=7.5\times10^{-4}) |
| MMS-IV | Annular star / mixed Dirichlet--Neumann | (K=22), (N_b=720), (r=832), (T=0.9), (Delta t=1.5\times10^{-3}) |
| MMS-V | Pinwheel shell / homogeneous Dirichlet | (K=22), (N_b=420), (r=644), (T=0.9), (Delta t=1.2\times10^{-3}) |
| Allen--Cahn | Disk / homogeneous Neumann | (K=22), (N_b=1600), (r=412), (T=4.0), (Delta t=5\times10^{-4}) |
| Vector Burgers | Inner square / homogeneous Dirichlet | (K=22), (N_b=1000), (r=833), (T=0.5), (Delta t=5\times10^{-4}) |
| Inverse discovery | Peanut and oblique channel / homogeneous Dirichlet | (K=22), (r=80), (N_{\mathrm{traj}}=16), (T_{\mathrm{id}}=0.012), (Delta t_{\mathrm{obs}}=5\times10^{-5}), (N_{\mathrm{obs}}\in\{240,1000\}) |

The values above are passed from JSON files under `configs/paper/`; experiment scripts do not silently select them. Every scientific parameter can also be supplied directly through the command line. Configuration metadata distinguishes manuscript-stated values from implementation controls that the manuscript does not specify, such as the number of optimizer epochs.

## Repository layout

```text
gLegONet/
├── configs/paper/                 # Explicit manuscript configurations
├── experiments/
│   ├── manufactured_solutions/    # MMS-I through MMS-V
│   ├── allen_cahn/                 # Volume-constrained Allen--Cahn
│   ├── burgers/                    # Two-component Burgers comparison
│   ├── inverse_discovery/          # Peanut/channel law identification
│   ├── cylinder_wake/              # Post-acceptance placeholder
│   └── swift_hohenberg/            # Post-acceptance placeholder
├── scripts/run_config.py           # JSON-to-CLI launcher with overrides
├── training/
│   ├── laplace/                    # Dissipative diagonal block
│   └── transport/                  # Shared local density for x/y transport
└── tests/                           # Configuration and smoke tests
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `--set device=cpu` when CUDA is unavailable.

## Reproduction workflow

Train the two (K=22) frozen blocks first:

```bash
python scripts/run_config.py --config configs/paper/train_laplace_k22.json
python scripts/run_config.py --config configs/paper/train_transport_k22.json
```

The commands write unpublished artifacts to `artifacts/laplace/K22/` and `artifacts/transport/K22/`. That directory is ignored by Git.

Run a forward benchmark:

```bash
python scripts/run_config.py --config configs/paper/mms_01_rosette.json
python scripts/run_config.py --config configs/paper/allen_cahn.json
python scripts/run_config.py --config configs/paper/burgers.json
```

Run sparse law identification:

```bash
python scripts/run_config.py --config configs/paper/inverse_peanut.json
python scripts/run_config.py --config configs/paper/inverse_channel.json
```

Override any input without editing code or the checked-in configuration:

```bash
python scripts/run_config.py \
  --config configs/paper/mms_01_rosette.json \
  --set K=24 \
  --set reduced_rank=240 \
  --set laplace_checkpoint="artifacts/laplace/K24/model_state.pt"
```

Use `--dry-run` to inspect the generated command. Each experiment also supports `python <entrypoint> --help`.

## Data and checkpoints

The manufactured fields, geometries, quadrature nodes, initial conditions, and inverse-discovery observations are generated at runtime, so no external dataset is required for the released experiments. Pretrained checkpoints are intentionally excluded from this release. A checkpoint must be trained with the same (K) used by the target experiment; loaders reject a resolution mismatch.

Because neural training and hardware-dependent linear algebra are stochastic, exact last digits can vary. Reproduction should use the stated seed and compare the saved relative-error and boundary-residual summaries. The code never substitutes an analytic block when a required learned checkpoint is missing.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@article{zhang2026geometryawarelegonet,
  title   = {Geometry-aware LegONet for PDE Learning on Arbitrary Domains},
  author  = {Zhang, Jiahao and Wang, Yueqi and Lin, Guang},
  journal = {Manuscript under review},
  year    = {2026}
}
```

Citation metadata are also available in [`CITATION.cff`](CITATION.cff).

## License

A software license has not yet been selected by the authors. Until a license file is added, copyright remains with the authors and reuse requires permission.
