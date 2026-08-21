# Laplace block

`train.py` learns the nonnegative diagonal energy matrix in the dissipative generator

$$
\mathbf{F}_\Delta^\theta(\mathbf{a})=-G_\Delta\nabla_{\mathbf a}E_\Delta^\theta(\mathbf a).
$$

Training and output parameters are command-line inputs. Inspect the complete interface with:

```bash
python training/laplace/train.py --help
```
