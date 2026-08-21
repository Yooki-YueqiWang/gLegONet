# Laplace block

`train.py` learns the nonnegative diagonal energy matrix in the dissipative generator

$$
\mathbf{F}_\Delta^\theta(\mathbf{a})=-G_\Delta\nabla_{\mathbf a}E_\Delta^\theta(\mathbf a).
$$

All training and output parameters are CLI inputs. See `configs/paper/train_laplace_k22.json` for the manuscript protocol and the explicitly labeled implementation controls.
