r"""Explicit numerical integrators used as ground truth.

We implement two schemes by hand rather than pulling in ``scipy`` or
``torchdiffeq``:

1. **Classical fourth-order Runge–Kutta (``rk4_step``).**  A general-purpose
   non-symplectic scheme. It is fourth-order accurate in ``dt``, so at small
   step sizes the energy drift is tiny over the rollout lengths we care about
   (:math:`\mathcal{O}(10^4)` steps), which is sufficient for generating
   training data.

2. **Implicit midpoint rule (``implicit_midpoint_step``).**  The canonical
   second-order symplectic integrator that works for *any* Hamiltonian system,
   separable or not. The update is

   .. math::

       y_{n+1} = y_n + \Delta t \, f\!\Big(\tfrac12 (y_n + y_{n+1})\Big),

   which we solve by fixed-point iteration on the midpoint
   :math:`y_{1/2} = \tfrac12 (y_n + y_{n+1})`. Because the double pendulum's
   Hamiltonian is **non-separable** (``T`` depends on ``q`` through
   :math:`M^{-1}(q)`), the usual Störmer–Verlet leap-frog is *not* symplectic
   on this system; implicit midpoint is the simplest integrator that is.

Both steppers take a callable ``ode_fn(state) -> state_dot`` and a state of
shape ``(..., d)``. They never inspect the physics of the problem, so they
double as integrators for the learned models too.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]


def rk4_step(ode_fn: ArrayFn, state: np.ndarray, dt: float) -> np.ndarray:
    """Classical explicit fourth-order Runge–Kutta step."""
    k1 = ode_fn(state)
    k2 = ode_fn(state + 0.5 * dt * k1)
    k3 = ode_fn(state + 0.5 * dt * k2)
    k4 = ode_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def implicit_midpoint_step(
    ode_fn: ArrayFn,
    state: np.ndarray,
    dt: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> np.ndarray:
    r"""One step of the implicit midpoint rule.

    The nonlinear relation
    :math:`y_{n+1} = y_n + \Delta t\, f(\tfrac12 (y_n + y_{n+1}))`
    is solved by fixed-point iteration on the midpoint state. For the step
    sizes we use here (``dt ~ 1e-2`` to ``1e-3``) the map is a strong
    contraction and convergence is reached in a handful of iterations.

    Parameters
    ----------
    ode_fn : callable
        Vector field ``f(y)`` to integrate.
    state : ndarray
        Current state ``y_n``.
    dt : float
        Step size.
    tol : float
        Convergence tolerance on the infinity norm of successive midpoint
        iterates.
    max_iter : int
        Hard cap on iterations; a ``RuntimeError`` is raised if it is hit.
    """
    # Warm start the midpoint with an explicit Euler half-step.
    mid = state + 0.5 * dt * ode_fn(state)
    for _ in range(max_iter):
        f_mid = ode_fn(mid)
        new_mid = state + 0.5 * dt * f_mid
        if np.max(np.abs(new_mid - mid)) < tol:
            mid = new_mid
            break
        mid = new_mid
    else:  # pragma: no cover - only hit if dt is absurdly large
        raise RuntimeError(
            f"implicit_midpoint_step failed to converge in {max_iter} iterations"
        )
    return state + dt * ode_fn(mid)


def integrate(
    ode_fn: ArrayFn,
    state0: np.ndarray,
    dt: float,
    n_steps: int,
    *,
    method: str = "rk4",
) -> np.ndarray:
    """Roll out ``n_steps`` of ``ode_fn`` starting at ``state0``.

    Returns an array of shape ``(n_steps + 1, *state0.shape)`` containing the
    initial state followed by each integrated step.
    """
    if method == "rk4":
        step = rk4_step
    elif method in {"implicit_midpoint", "symplectic"}:
        step = implicit_midpoint_step
    else:
        raise ValueError(f"unknown integrator: {method!r}")

    traj = np.empty((n_steps + 1,) + state0.shape, dtype=np.float64)
    traj[0] = state0
    state = state0
    for i in range(n_steps):
        state = step(ode_fn, state, dt)
        traj[i + 1] = state
    return traj
