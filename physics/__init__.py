"""Classical mechanics primitives for the double pendulum (ground truth)."""

from physics.double_pendulum import (
    DoublePendulumParams,
    hamiltonian,
    lagrangian,
    dynamics,
    qp_to_qqdot,
    qqdot_to_qp,
    DEFAULT_PARAMS,
)
from physics.integrators import rk4_step, implicit_midpoint_step, integrate

__all__ = [
    "DoublePendulumParams",
    "hamiltonian",
    "lagrangian",
    "dynamics",
    "qp_to_qqdot",
    "qqdot_to_qp",
    "DEFAULT_PARAMS",
    "rk4_step",
    "implicit_midpoint_step",
    "integrate",
]
