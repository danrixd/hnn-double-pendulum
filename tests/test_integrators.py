"""Ground-truth integrator sanity checks.

The headline assertion here is that the symplectic (implicit midpoint) scheme
conserves energy to within 1e-3 (relative) over 10^4 steps of the double
pendulum. This is the first sanity check the project brief calls for, and it
has to hold before we start training any neural net against this data.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics import (
    DEFAULT_PARAMS,
    dynamics,
    hamiltonian,
    implicit_midpoint_step,
    integrate,
    qp_to_qqdot,
    qqdot_to_qp,
    rk4_step,
)


@pytest.fixture
def ode_fn():
    return lambda s: dynamics(s, DEFAULT_PARAMS)


def _mild_initial_state() -> np.ndarray:
    # Moderate swing, starts from rest. Energetic enough to exercise the
    # non-linear coupling but not so chaotic that a test becomes brittle.
    return np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float64)


def test_dynamics_shape(ode_fn):
    state = _mild_initial_state()
    dstate = ode_fn(state)
    assert dstate.shape == state.shape


def test_legendre_roundtrip():
    q = np.array([1.2, -0.4])
    qdot = np.array([0.3, -0.7])
    state_qp = qqdot_to_qp(q, qdot, DEFAULT_PARAMS)
    state_qqdot = qp_to_qqdot(state_qp[..., :2], state_qp[..., 2:], DEFAULT_PARAMS)
    np.testing.assert_allclose(state_qqdot[..., :2], q, atol=1e-12)
    np.testing.assert_allclose(state_qqdot[..., 2:], qdot, atol=1e-12)


def test_hamilton_equations_at_rest():
    # From rest q̇ must vanish; ṗ is pure gravity and points toward equilibrium.
    state = np.array([0.3, -0.2, 0.0, 0.0])
    dstate = dynamics(state, DEFAULT_PARAMS)
    np.testing.assert_allclose(dstate[:2], 0.0, atol=1e-12)
    # ṗ1 = -(m1+m2) g l1 sin(theta1); ṗ2 = -m2 g l2 sin(theta2).
    p = DEFAULT_PARAMS
    expected_p1 = -(p.m1 + p.m2) * p.g * p.l1 * np.sin(0.3)
    expected_p2 = -p.m2 * p.g * p.l2 * np.sin(-0.2)
    np.testing.assert_allclose(dstate[2], expected_p1, atol=1e-12)
    np.testing.assert_allclose(dstate[3], expected_p2, atol=1e-12)


def test_implicit_midpoint_energy_conservation(ode_fn):
    """10^4 steps of the symplectic integrator, relative energy drift < 1e-3."""
    state0 = _mild_initial_state()
    dt = 0.01
    n_steps = 10_000

    traj = integrate(ode_fn, state0, dt, n_steps, method="implicit_midpoint")

    H = hamiltonian(traj[:, :2], traj[:, 2:], DEFAULT_PARAMS)
    H0 = H[0]
    rel_drift = np.max(np.abs((H - H0) / H0))
    assert rel_drift < 1e-3, f"symplectic rel energy drift = {rel_drift:.3e}"


def test_rk4_energy_conservation(ode_fn):
    """RK4 is non-symplectic but 4th order; at dt=0.01 drift is still tiny."""
    state0 = _mild_initial_state()
    dt = 0.01
    n_steps = 10_000

    traj = integrate(ode_fn, state0, dt, n_steps, method="rk4")

    H = hamiltonian(traj[:, :2], traj[:, 2:], DEFAULT_PARAMS)
    H0 = H[0]
    rel_drift = np.max(np.abs((H - H0) / H0))
    assert rel_drift < 1e-3, f"rk4 rel energy drift = {rel_drift:.3e}"


def test_rk4_matches_implicit_midpoint_short_horizon(ode_fn):
    # On short horizons the two schemes should agree pointwise to ~dt^2.
    state0 = _mild_initial_state()
    dt = 0.001
    n_steps = 200

    traj_rk4 = integrate(ode_fn, state0, dt, n_steps, method="rk4")
    traj_mid = integrate(ode_fn, state0, dt, n_steps, method="implicit_midpoint")
    err = np.max(np.abs(traj_rk4 - traj_mid))
    assert err < 1e-5, f"short-horizon integrator disagreement = {err:.3e}"


def test_single_step_apis_agree_with_integrate(ode_fn):
    # integrate() should just be a loop over the per-step functions.
    state0 = _mild_initial_state()
    dt = 0.01

    s_rk4 = rk4_step(ode_fn, state0, dt)
    s_mid = implicit_midpoint_step(ode_fn, state0, dt)

    traj_rk4 = integrate(ode_fn, state0, dt, 1, method="rk4")
    traj_mid = integrate(ode_fn, state0, dt, 1, method="implicit_midpoint")

    np.testing.assert_allclose(traj_rk4[1], s_rk4, atol=1e-14)
    np.testing.assert_allclose(traj_mid[1], s_mid, atol=1e-14)
