# Volume-constrained Allen--Cahn

This experiment solves

$$
u_t=\varepsilon^2\Delta u+u-u^3-\lambda(t),
\qquad
\lambda(t)=\frac{1}{|\Omega|}\int_\Omega(u-u^3)\,d\mathbf{x},
$$

on a disk with homogeneous Neumann data. The reaction half-steps use RK4 and the diffusion step uses Crank--Nicolson in a symmetric Strang composition. The mean correction is recomputed at every nonlinear RK stage.

```bash
python experiments/allen_cahn/run.py --help
```
