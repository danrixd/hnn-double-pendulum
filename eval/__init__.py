"""Shared evaluation helpers for rolling out learned models.

Every script in this package uses the same pattern:

1. Load a checkpoint produced by ``train.py``.
2. Wrap the PyTorch model in a NumPy callable compatible with the
   integrators in :mod:`physics.integrators`.
3. Roll the learned field forward in time with the same explicit integrator
   (classical RK4) that we use for the MLP baseline — so the only thing
   distinguishing the curves in every figure is the vector field itself, not
   the integration scheme.

The ground-truth reference is always produced by the symplectic (implicit
midpoint) integrator on :func:`physics.dynamics`, for clean energy conservation
along the reference trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from models import HNN, MLPBaseline
from physics import DEFAULT_PARAMS, dynamics, hamiltonian, integrate

OdeFn = Callable[[np.ndarray], np.ndarray]


def load_checkpoint(path: Path) -> torch.nn.Module:
    """Instantiate and load a model from a ``train.py`` checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hidden = tuple(ckpt["hidden_sizes"])
    name = ckpt["model"]
    if name == "hnn":
        model: torch.nn.Module = HNN(state_dim=4, hidden_sizes=hidden)
    elif name == "mlp":
        model = MLPBaseline(state_dim=4, hidden_sizes=hidden)
    else:
        raise ValueError(f"unknown model in checkpoint: {name!r}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def model_to_ode(model: torch.nn.Module) -> OdeFn:
    """Wrap a torch model as a NumPy callable ``state -> d state/dt``.

    The HNN enables grad internally on its forward pass, so we do *not* wrap
    the call in ``torch.no_grad``; that would break autograd assembly of the
    vector field.
    """
    def ode(state: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(state, dtype=torch.float32)
        out = model(t)
        return out.detach().cpu().numpy().astype(np.float64)

    return ode


def ground_truth_ode() -> OdeFn:
    return lambda s: dynamics(s, DEFAULT_PARAMS)


def rollout(
    ode_fn: OdeFn,
    state0: np.ndarray,
    dt: float,
    n_steps: int,
    *,
    method: str = "rk4",
) -> np.ndarray:
    """Thin pass-through to :func:`physics.integrate` kept for eval readability."""
    return integrate(ode_fn, state0, dt, n_steps, method=method)


def energy_series(trajectory: np.ndarray) -> np.ndarray:
    """True Hamiltonian evaluated along every state of a rollout."""
    q = trajectory[..., :2]
    p = trajectory[..., 2:]
    return hamiltonian(q, p, DEFAULT_PARAMS)


def relative_energy_drift(trajectory: np.ndarray) -> np.ndarray:
    """``|H(t) - H(0)| / |H(0)|`` along a rollout."""
    H = energy_series(trajectory)
    return np.abs((H - H[0]) / H[0])
