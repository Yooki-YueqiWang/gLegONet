"""Checkpoint loaders shared by the manufactured-solution experiments."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _load(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _bundle_metadata(K: int) -> dict[str, np.ndarray]:
    entries: list[tuple[int, int, str]] = [(0, 0, "const")]
    pairs: list[tuple[int, int]] = []
    for k in range(-K, K + 1):
        for ell in range(-K, K + 1):
            if k == 0 and ell == 0:
                continue
            if k * k + ell * ell <= K * K and (k > 0 or (k == 0 and ell > 0)):
                pairs.append((k, ell))
    pairs.sort(key=lambda pair: (pair[0] * pair[0] + pair[1] * pair[1], pair[0], pair[1]))
    for k, ell in pairs:
        entries.extend([(k, ell, "cos"), (k, ell, "sin")])
    return {
        "k": np.asarray([entry[0] for entry in entries], dtype=np.int64),
        "ell": np.asarray([entry[1] for entry in entries], dtype=np.int64),
        "kind": np.asarray([entry[2] for entry in entries]),
    }


def _bundle_to_mms_transform(meta_mms: list[tuple[str, int, int]], K: int) -> np.ndarray:
    bundle = _bundle_metadata(K)
    mms_index = {entry: index for index, entry in enumerate(meta_mms)}
    bundle_index = {
        (int(k), int(ell), str(kind)): index
        for index, (k, ell, kind) in enumerate(zip(bundle["k"], bundle["ell"], bundle["kind"]))
    }
    dimension = len(meta_mms)
    transform = np.zeros((dimension, dimension), dtype=np.float64)
    transform[bundle_index[(0, 0, "const")], mms_index[("coscos", 0, 0)]] = 1.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_2sqrt2 = 0.5 * inv_sqrt2
    for (kind, k, ell), column in mms_index.items():
        if (kind, k, ell) == ("coscos", 0, 0):
            continue
        if kind == "coscos":
            if k > 0 and ell > 0:
                transform[bundle_index[(k, ell, "cos")], column] += inv_2sqrt2
                transform[bundle_index[(k, -ell, "cos")], column] += inv_2sqrt2
            elif k > 0:
                transform[bundle_index[(k, 0, "cos")], column] += inv_sqrt2
            else:
                transform[bundle_index[(0, ell, "cos")], column] += inv_sqrt2
        elif kind == "sinsin":
            transform[bundle_index[(k, -ell, "cos")], column] += inv_2sqrt2
            transform[bundle_index[(k, ell, "cos")], column] -= inv_2sqrt2
        elif kind == "sincos":
            if ell > 0:
                transform[bundle_index[(k, ell, "sin")], column] += inv_2sqrt2
                transform[bundle_index[(k, -ell, "sin")], column] += inv_2sqrt2
            else:
                transform[bundle_index[(k, 0, "sin")], column] += inv_sqrt2
        elif kind == "cossin":
            if k > 0:
                transform[bundle_index[(k, ell, "sin")], column] += inv_2sqrt2
                transform[bundle_index[(k, -ell, "sin")], column] -= inv_2sqrt2
            else:
                transform[bundle_index[(0, ell, "sin")], column] += inv_sqrt2
        else:
            raise ValueError(f"Unknown basis kind: {kind}")
    return transform


def load_laplace_operator(
    meta_mms: list[tuple[str, int, int]], K: int, checkpoint: str | Path
) -> tuple[np.ndarray, dict[str, float]]:
    path = Path(checkpoint)
    payload = _load(path)
    checkpoint_K = int(payload.get("K", K))
    if checkpoint_K != int(K):
        raise ValueError(f"Checkpoint K={checkpoint_K} does not match experiment K={K}: {path}")
    state = payload["model_state"]
    raw = state["raw_diag"].detach().cpu().numpy().astype(np.float64)
    scale = float(state["scale"].detach().cpu().numpy().reshape(-1)[0])
    mask = state["nonzero_mask"].detach().cpu().numpy().astype(np.float64)
    learned_diagonal = -(scale * np.logaddexp(0.0, raw) * mask)
    transform = _bundle_to_mms_transform(meta_mms, K)
    operator = np.linalg.solve(transform, learned_diagonal[:, None] * transform)
    exact_diagonal = -math.pi**2 * np.asarray(
        [float(k * k + ell * ell) for _, k, ell in meta_mms], dtype=np.float64
    )
    exact_operator = np.diag(exact_diagonal)
    diagnostics = {
        "relative_error_to_exact": float(
            np.linalg.norm(operator - exact_operator) / (np.linalg.norm(exact_operator) + 1.0e-14)
        ),
        "off_diagonal_norm": float(np.linalg.norm(operator - np.diag(np.diag(operator)))),
    }
    return operator, diagnostics


class DensityNet(nn.Module):
    """Four-hidden-layer pointwise density network used by transport blocks."""

    def __init__(self, width: int = 128, depth: int = 4, act: str = "gelu") -> None:
        super().__init__()
        activations = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        if act.lower() not in activations:
            raise ValueError(f"Unsupported activation: {act}")
        layers: list[nn.Module] = []
        input_width = 1
        for _ in range(depth):
            layers.extend([nn.Linear(input_width, width), activations[act.lower()]()])
            input_width = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.unsqueeze(-1)).squeeze(-1)


def load_transport_model(
    K: int, checkpoint: str | Path, device: torch.device | str = "cpu"
) -> DensityNet:
    path = Path(checkpoint)
    payload = _load(path, device)
    checkpoint_K = int(payload.get("K", K))
    if checkpoint_K != int(K):
        raise ValueError(f"Checkpoint K={checkpoint_K} does not match experiment K={K}: {path}")
    config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
    nested = config.get("density_net", {}) if isinstance(config.get("density_net", {}), dict) else {}
    model = DensityNet(
        width=int(config.get("width", nested.get("width", 128))),
        depth=int(config.get("depth", nested.get("depth", 4))),
        act=str(config.get("act", nested.get("act", "gelu"))),
    ).to(device=device, dtype=torch.float64)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def transport_multiplier(model: DensityNet, values: np.ndarray, device: torch.device) -> np.ndarray:
    """Evaluate h_theta''(u), which converges to u for the Burgers density."""
    with torch.enable_grad():
        u = torch.as_tensor(values, dtype=torch.float64, device=device).detach().clone().requires_grad_(True)
        density = model(u)
        first = torch.autograd.grad(density.sum(), u, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), u, create_graph=False)[0]
    return second.detach().cpu().numpy().astype(np.float64, copy=False)
