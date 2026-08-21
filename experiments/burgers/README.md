# Vector Burgers on an embedded square

The two velocity components share one Dirichlet-compatible matrix (N_\Omega). Frozen diffusion and self-advection blocks are reused, while cross-advection is assembled by analytic weak projection. Nonlinear half-steps use six RK4 substeps and diffusion uses Crank--Nicolson.

```bash
python scripts/run_config.py --config configs/paper/burgers.json
```

The run also constructs the independent (201\times201) upwind finite-volume reference specified in the manuscript configuration.
