# Vector Burgers on an embedded square

The two velocity components share one Dirichlet-compatible matrix (N_\Omega). Frozen diffusion and self-advection blocks are reused, while cross-advection is assembled by analytic weak projection. Nonlinear half-steps use six RK4 substeps and diffusion uses Crank--Nicolson.

```bash
python experiments/burgers/run.py --help
```

The run also constructs an independent upwind finite-volume reference. Its resolution and integration parameters are exposed through the command-line interface.

Use `python experiments/burgers/model.py --help` to inspect the learned-block versus exact-operator comparison interface.
