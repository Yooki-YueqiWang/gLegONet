# Boundary-adapted physical-law discovery

The candidate dictionary is

$$
\mathcal D_\theta^\Omega=
\{\mathbf q_\Delta^\Omega,\mathbf q_x^\Omega,\mathbf q_y^\Omega,
\mathbf q_u^\Omega,\mathbf q_{u^2}^\Omega,\mathbf q_{u^3}^\Omega\}.
$$

The first three responses use the frozen (K=22) checkpoints; the polynomial responses are analytic Galerkin projections. Sparse sensor values are fitted to the rank-80 boundary-adapted coordinates, temporal derivatives are estimated from the short trajectories, and sequential thresholded ridge regression identifies the six coefficients. The recovered law is validated to (T_{\mathrm{roll}}=1.0).

```bash
python scripts/run_config.py --config configs/paper/inverse_peanut.json
python scripts/run_config.py --config configs/paper/inverse_channel.json
```

Use `--set obs_noise_rel=0.001` to run a relative-noise condition without editing the source or checked configuration.
