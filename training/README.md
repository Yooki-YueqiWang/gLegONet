# Ambient block training

The released library contains two single-purpose mechanisms trained on (Q=[-1,1]^2): a dissipative diagonal Laplace response and a shared scalar density for the directional transport terms (u u_x) and (u u_y).

The manuscript protocol uses 20,000 independent training samples, 4,000 held-out samples, Adam with learning rate (5\times10^{-4}), and mini-batches of 16. The transport density has four hidden layers of width 128 with GELU activations. Optimizer epoch counts are implementation controls because the manuscript does not prescribe them.

Run the checked configurations from the repository root:

```bash
python scripts/run_config.py --config configs/paper/train_laplace_k22.json
python scripts/run_config.py --config configs/paper/train_transport_k22.json
```

Train a separate library for every (K). Do not use a checkpoint at a different Fourier cutoff.
