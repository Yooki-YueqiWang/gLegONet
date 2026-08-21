# Directional transport block

`train.py` learns a scalar density (h_\theta) whose derivative satisfies

$$
h_\theta'(u)\approx \frac{u^2}{2}.
$$

The geometry-dependent weak derivative is assembled only after the target domain is known. One frozen density is reused for both (x)- and (y)-directional transport.
