# Geometry-aware LegONet (gLegONet)

Official research code for **Geometry-aware LegONet for PDE Learning on Arbitrary Domains** by Jiahao Zhang, Yueqi Wang, and Guang Lin.

gLegONet separates reusable physical mechanisms from target geometry. Diffusion and directional-transport blocks are pretrained on the ambient square $Q=[-1,1]^2$. A deterministic boundary-adapted Galerkin interface then realizes those frozen blocks on an unseen embedded domain without geometry-specific neural retraining.

## Method

The target PDE is decomposed into reusable mechanisms,

```math
\partial_t u(\mathbf{x},t)
=
\sum_{i=1}^{N_{\mathrm{blk}}} c_i L_i^\Omega(u)(\mathbf{x},t).
```

The ambient representation is the normalized real Fourier space

```math
\mathcal{V}_K(Q)=\mathrm{span}\left(
\{1\}\cup
\left\{\sqrt{2}\cos\bigl(\pi(kx+\ell y)\bigr),
\sqrt{2}\sin\bigl(\pi(kx+\ell y)\bigr):(k,\ell)\in\mathcal{I}_K^+\right\}
\right)
```

where

```math
\mathcal{I}_K=\{(k,\ell)\in\mathbb{Z}^2:k^2+\ell^2\le K^2\}.
```

For each new domain, sampled boundary conditions define

```math
C\mathbf{a}(t)=\mathbf{d}(t),
\qquad
\mathbf{a}(t)=\mathbf{a}_{\mathrm{bc}}(t)+N_\Omega\mathbf{z}(t),
```

with $CN_\Omega\approx0$ and $N_\Omega^\top M_\Omega N_\Omega=I$. The reduced dynamics are

```math
\dot{\mathbf{z}}(t)
=
\sum_{i=1}^{N_{\mathrm{blk}}}c_i\,\mathbf{q}_i^\Omega(\mathbf{z},t)
-N_\Omega^\top M_\Omega\dot{\mathbf{a}}_{\mathrm{bc}}(t).
```

The workflow is:

1. **Pretrain:** learn diffusion and local transport responses on $Q$.
2. **Freeze:** save one resolution-matched mechanism library for each Fourier cutoff $K$.
3. **Realize:** sample the target boundary, construct the affine lift and numerical null space, and mass-orthonormalize the admissible coordinates.
4. **Assemble:** project each frozen mechanism through the same target-domain Galerkin interface.
5. **Roll out or identify:** integrate in $\mathbf{z}$, or regress candidate mechanism coefficients from short-time sensor observations.

Boundary conditions are therefore part of the coordinates, not a penalty term or a post-step correction.

## Released experiments

| Experiment | Geometry / boundary condition |
|---|---|
| MMS-I | Rosette / homogeneous Dirichlet |
| MMS-II | Crescent / homogeneous Neumann |
| MMS-III | Bunny / nonhomogeneous Robin |
| MMS-IV | Annular star / mixed Dirichlet--Neumann |
| MMS-V | Pinwheel shell / homogeneous Dirichlet |
| Allen--Cahn | Disk / homogeneous Neumann |
| Vector Burgers | Inner square / homogeneous Dirichlet |
| Inverse discovery | Peanut and oblique channel / homogeneous Dirichlet |

## Repository layout

```text
gLegONet/
|-- experiments/
|   |-- manufactured_solutions/   # MMS-I through MMS-V
|   |-- allen_cahn/                # Volume-constrained Allen--Cahn
|   |-- burgers/                   # Two-component Burgers comparison
|   |-- inverse_discovery/         # Peanut/channel law identification
|   |-- cylinder_wake/             # Cylinder-wake study
|   `-- swift_hohenberg/           # Swift--Hohenberg study
|-- training/
|   |-- laplace/                   # Dissipative diagonal block
|   `-- transport/                 # Shared local density for x/y transport
`-- workflows/
    `-- mms_iv/                    # Complete paper-scale MMS-IV reproduction
```

## Complete end-to-end workflow

The [`workflows/mms_iv`](workflows/mms_iv/README.md) guide reproduces MMS-IV from ambient $K=22$ Laplace-block training through mixed-boundary matrix construction and RK4 rollout. It gives the complete command-line parameter signature, explains how $C$ and $N_\Omega$ are assembled, and includes a result verifier for the dimensions, residuals, and paper-scale errors.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Command-line interfaces

Every public driver exposes its inputs through `argparse`. Inspect the interface before constructing a run:

```bash
python training/laplace/train.py --help
python training/transport/train.py --help

python experiments/manufactured_solutions/mms_01_rosette.py --help
python experiments/manufactured_solutions/mms_02_crescent.py --help
python experiments/manufactured_solutions/mms_03_bunny.py --help
python experiments/manufactured_solutions/mms_04_annular_star.py --help
python experiments/manufactured_solutions/mms_05_pinwheel.py --help

python experiments/allen_cahn/run.py --help
python experiments/burgers/run.py --help
python experiments/burgers/model.py --help
python experiments/inverse_discovery/run.py --help
```

Required options are marked in each help page. Training and experiment scripts reject incompatible checkpoint resolutions. The source code contains the PDE definitions, geometry construction, discretization, and solver logic needed to execute the workflow; users choose run parameters explicitly at the command line.

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
