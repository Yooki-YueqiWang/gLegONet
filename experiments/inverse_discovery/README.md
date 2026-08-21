# Boundary-adapted physical-law discovery

The candidate dictionary is

$$
\mathcal D_\theta^\Omega=
\{\mathbf q_\Delta^\Omega,\mathbf q_x^\Omega,\mathbf q_y^\Omega,
\mathbf q_u^\Omega,\mathbf q_{u^2}^\Omega,\mathbf q_{u^3}^\Omega\}.
$$

The first three responses use frozen diffusion and transport checkpoints; the polynomial responses are analytic Galerkin projections. Sparse sensor values are fitted to boundary-adapted coordinates, temporal derivatives are estimated from short trajectories, and sequential thresholded ridge regression identifies the six coefficients. The recovered law is then assessed on held-out rollouts.

The same entry point supports both released geometries. Inspect its parameter interface with:

```bash
python experiments/inverse_discovery/run.py --help
```
