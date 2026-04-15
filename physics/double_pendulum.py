r"""Double pendulum in canonical (Hamiltonian) coordinates.

Generalised coordinates are the two angles measured from the downward vertical,
``q = (theta1, theta2)``; conjugate momenta ``p = (p1, p2)`` are obtained from the
Legendre transform of the Lagrangian.

Geometry and sign conventions
-----------------------------
Let the two bobs have masses :math:`m_1, m_2` and the rigid rods length
:math:`l_1, l_2`. With the pivot at the origin and :math:`y` pointing up,

.. math::

    x_1 &=  l_1 \sin\theta_1,             & y_1 &= -l_1 \cos\theta_1, \\
    x_2 &=  l_1 \sin\theta_1 + l_2 \sin\theta_2,
    & y_2 &= -l_1 \cos\theta_1 - l_2 \cos\theta_2.

Lagrangian
----------
Kinetic and potential energies are

.. math::

    T &= \tfrac12 (m_1 + m_2) l_1^2 \dot\theta_1^2
       + \tfrac12 m_2 l_2^2 \dot\theta_2^2
       + m_2 l_1 l_2 \cos(\theta_1-\theta_2)\,\dot\theta_1 \dot\theta_2, \\
    V &= -(m_1 + m_2) g l_1 \cos\theta_1 - m_2 g l_2 \cos\theta_2,

so :math:`L = T - V`.

Legendre transform
------------------
The kinetic energy is quadratic in the generalised velocities, ``T = ½ q̇ᵀ M(q) q̇``,
with the (configuration-dependent) mass matrix

.. math::

    M(q) = \begin{pmatrix}
        (m_1+m_2) l_1^2 & m_2 l_1 l_2 \cos(\theta_1-\theta_2) \\
        m_2 l_1 l_2 \cos(\theta_1-\theta_2) & m_2 l_2^2
    \end{pmatrix}.

The canonical momenta are :math:`p = \partial L / \partial \dot q = M(q)\,\dot q`,
so :math:`\dot q = M^{-1}(q)\,p`. The Hamiltonian is then

.. math::

    H(q,p) = \tfrac12\, p^{\top} M^{-1}(q)\, p + V(q),

which is **not** separable in (q, p) because ``M`` depends on ``q``. This is the
reason ``integrators.py`` uses the implicit midpoint rule (a symplectic method
that is valid for any Hamiltonian) rather than Störmer–Verlet, which is only
symplectic for separable ``H``.

Equations of motion
-------------------
Hamilton's equations give

.. math::

    \dot q &= M^{-1}(q)\, p, \\
    \dot p_1 &= -(m_1+m_2) g l_1 \sin\theta_1 - m_2 l_1 l_2 \sin(\theta_1-\theta_2)\,
                \dot\theta_1 \dot\theta_2, \\
    \dot p_2 &= -m_2 g l_2 \sin\theta_2 + m_2 l_1 l_2 \sin(\theta_1-\theta_2)\,
                \dot\theta_1 \dot\theta_2.

The force terms follow from
:math:`\partial H/\partial q_i = -\tfrac12\, \dot q^{\top} (\partial M/\partial q_i)\, \dot q
+ \partial V/\partial q_i`, using
:math:`\partial_{q_i} M^{-1} = -M^{-1}(\partial_{q_i} M) M^{-1}`, and noting that
only the :math:`\cos(\theta_1-\theta_2)` entry of ``M`` is coordinate-dependent.

All routines in this module operate on plain :class:`numpy.ndarray` objects and
broadcast over a leading batch dimension, i.e. ``q`` and ``p`` may have shape
``(..., 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoublePendulumParams:
    """Physical parameters of a planar double pendulum.

    Attributes
    ----------
    m1, m2 : float
        Bob masses.
    l1, l2 : float
        Rod lengths.
    g : float
        Gravitational acceleration (magnitude, pointing in :math:`-\\hat y`).
    """

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81


DEFAULT_PARAMS = DoublePendulumParams()


# ---------------------------------------------------------------------------
# Mass matrix and its inverse
# ---------------------------------------------------------------------------

def _mass_matrix(q: np.ndarray, params: DoublePendulumParams):
    """Return the entries ``(M11, M12, M22)`` of the symmetric mass matrix ``M(q)``."""
    theta1 = q[..., 0]
    theta2 = q[..., 1]
    c = np.cos(theta1 - theta2)
    M11 = (params.m1 + params.m2) * params.l1 ** 2
    M12 = params.m2 * params.l1 * params.l2 * c
    M22 = params.m2 * params.l2 ** 2
    # Broadcast M11/M22 to match the shape of M12 if q is batched.
    M11 = np.broadcast_to(M11, M12.shape).copy() if np.ndim(M12) else M11
    M22 = np.broadcast_to(M22, M12.shape).copy() if np.ndim(M12) else M22
    return M11, M12, M22


def _mass_matrix_inv(q: np.ndarray, params: DoublePendulumParams):
    """Return the entries ``(I11, I12, I22)`` of ``M^{-1}(q)``."""
    M11, M12, M22 = _mass_matrix(q, params)
    det = M11 * M22 - M12 * M12
    I11 = M22 / det
    I12 = -M12 / det
    I22 = M11 / det
    return I11, I12, I22


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------

def lagrangian(
    q: np.ndarray, qdot: np.ndarray, params: DoublePendulumParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Scalar Lagrangian ``L = T - V`` evaluated at ``(q, qdot)``."""
    M11, M12, M22 = _mass_matrix(q, params)
    qd1 = qdot[..., 0]
    qd2 = qdot[..., 1]
    T = 0.5 * (M11 * qd1 * qd1 + 2.0 * M12 * qd1 * qd2 + M22 * qd2 * qd2)
    V = _potential(q, params)
    return T - V


def hamiltonian(
    q: np.ndarray, p: np.ndarray, params: DoublePendulumParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Scalar Hamiltonian ``H = ½ pᵀ M⁻¹(q) p + V(q)``."""
    I11, I12, I22 = _mass_matrix_inv(q, params)
    p1 = p[..., 0]
    p2 = p[..., 1]
    T = 0.5 * (I11 * p1 * p1 + 2.0 * I12 * p1 * p2 + I22 * p2 * p2)
    V = _potential(q, params)
    return T + V


def _potential(q: np.ndarray, params: DoublePendulumParams) -> np.ndarray:
    theta1 = q[..., 0]
    theta2 = q[..., 1]
    return -(
        (params.m1 + params.m2) * params.g * params.l1 * np.cos(theta1)
        + params.m2 * params.g * params.l2 * np.cos(theta2)
    )


# ---------------------------------------------------------------------------
# Coordinate conversions between (q, qdot) and (q, p)
# ---------------------------------------------------------------------------

def qqdot_to_qp(
    q: np.ndarray, qdot: np.ndarray, params: DoublePendulumParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Map ``(q, qdot)`` to the canonical state ``(q, p) = (q, M(q) qdot)``.

    Returns an array of shape ``(..., 4)`` laid out as ``[q1, q2, p1, p2]``.
    """
    M11, M12, M22 = _mass_matrix(q, params)
    p1 = M11 * qdot[..., 0] + M12 * qdot[..., 1]
    p2 = M12 * qdot[..., 0] + M22 * qdot[..., 1]
    return np.stack([q[..., 0], q[..., 1], p1, p2], axis=-1)


def qp_to_qqdot(
    q: np.ndarray, p: np.ndarray, params: DoublePendulumParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Inverse of :func:`qqdot_to_qp`: recover generalised velocities from momenta."""
    I11, I12, I22 = _mass_matrix_inv(q, params)
    qd1 = I11 * p[..., 0] + I12 * p[..., 1]
    qd2 = I12 * p[..., 0] + I22 * p[..., 1]
    return np.stack([q[..., 0], q[..., 1], qd1, qd2], axis=-1)


# ---------------------------------------------------------------------------
# Vector field: ground-truth Hamilton's equations
# ---------------------------------------------------------------------------

def dynamics(
    state: np.ndarray, params: DoublePendulumParams = DEFAULT_PARAMS
) -> np.ndarray:
    r"""Analytical vector field ``(q̇, ṗ) = (∂H/∂p, −∂H/∂q)``.

    Parameters
    ----------
    state : ndarray, shape ``(..., 4)``
        Canonical state ``[theta1, theta2, p1, p2]``.
    params : DoublePendulumParams
        Physical parameters.

    Returns
    -------
    ndarray, shape ``(..., 4)``
        Time derivative ``[theta1_dot, theta2_dot, p1_dot, p2_dot]``.

    Notes
    -----
    Closed-form derivatives of ``H`` are used here rather than autograd so that
    (i) we avoid a PyTorch dependency in the ground-truth integrator, and
    (ii) we have an analytical reference to test the learned models against.
    """
    q = state[..., :2]
    p = state[..., 2:]

    # q̇ = M⁻¹ p
    I11, I12, I22 = _mass_matrix_inv(q, params)
    qd1 = I11 * p[..., 0] + I12 * p[..., 1]
    qd2 = I12 * p[..., 0] + I22 * p[..., 1]

    # Force from the configuration-dependent kinetic term.
    # Only M12 = m2 l1 l2 cos(theta1 - theta2) depends on q; differentiating
    # ½ q̇ᵀ M q̇ w.r.t. theta1 yields +m2 l1 l2 sin(theta1-theta2) q̇1 q̇2,
    # and w.r.t. theta2 the opposite sign. The kinetic contribution to
    # ṗ = −∂H/∂q picks up an overall minus.
    theta1 = q[..., 0]
    theta2 = q[..., 1]
    delta = theta1 - theta2
    s = np.sin(delta)

    cross = params.m2 * params.l1 * params.l2 * s * qd1 * qd2

    # Gravity contribution.
    grav1 = (params.m1 + params.m2) * params.g * params.l1 * np.sin(theta1)
    grav2 = params.m2 * params.g * params.l2 * np.sin(theta2)

    pd1 = -grav1 - cross
    pd2 = -grav2 + cross

    return np.stack([qd1, qd2, pd1, pd2], axis=-1)
