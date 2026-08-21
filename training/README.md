# Ambient block training

The released library contains two single-purpose mechanisms trained on $Q=[-1,1]^2$: a dissipative diagonal Laplace response and a shared scalar density for the directional transport terms $u u_x$ and $u u_y$.

This release provides parameterized training entry points but no ready-to-run or paper-specific parameter set. Inspect every required and optional input from the repository root:

```bash
python training/laplace/train.py --help
python training/transport/train.py --help
```

Train a separate mechanism library for every Fourier cutoff. A checkpoint cannot be reused at a different cutoff.
