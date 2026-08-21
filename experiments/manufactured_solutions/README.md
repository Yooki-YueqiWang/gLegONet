# Manufactured-solution benchmarks

This directory contains the five benchmarks in Figure 3. Each script constructs the target geometry, samples its boundary operator, forms an affine lift when needed, builds a mass-orthonormal null-space basis, realizes the frozen mechanism blocks, advances the reduced system with RK4, and evaluates the analytic manufactured solution.

| Script | PDE | Boundary condition |
|---|---|---|
| `mms_01_rosette.py` | cubic reaction--diffusion | homogeneous Dirichlet |
| `mms_02_crescent.py` | cubic reaction--diffusion | homogeneous Neumann |
| `mms_03_bunny.py` | logistic reaction--diffusion | nonhomogeneous Robin |
| `mms_04_annular_star.py` | coupled reaction--diffusion | outer Dirichlet, inner Neumann |
| `mms_05_pinwheel.py` | scalar Burgers transport | homogeneous Dirichlet |

Run any case through its JSON file, for example:

```bash
python scripts/run_config.py --config configs/paper/mms_03_bunny.json
```

The learned block checkpoint is mandatory and must match the requested (K).
