# From Lagrangian to Hamiltonian: the planar double pendulum

This note derives the equations of motion used throughout the repository. It is
self-contained at the level of an undergraduate classical mechanics course, and
the numerical code in [`double_pendulum.py`](./double_pendulum.py) is a direct
translation of the formulae here.

## 1. Geometry

A planar double pendulum consists of two point masses $m_1$ and $m_2$ connected
by rigid, massless rods of length $l_1$ and $l_2$. The first rod is pinned to
the origin; the second rod hangs from the first bob. We measure the angles
$\theta_1$ and $\theta_2$ of the rods from the downward vertical, with the
$y$-axis pointing up.

$$
\begin{aligned}
x_1 &= l_1 \sin\theta_1, & y_1 &= -l_1 \cos\theta_1, \\
x_2 &= l_1 \sin\theta_1 + l_2 \sin\theta_2, & y_2 &= -l_1 \cos\theta_1 - l_2 \cos\theta_2.
\end{aligned}
$$

Both coordinates are holonomic (the rod lengths are fixed constraints already
eliminated by the choice of $\theta_i$), and the system has two degrees of
freedom.

## 2. Lagrangian

Differentiating the positions and squaring,

$$
\begin{aligned}
\dot x_1^2 + \dot y_1^2 &= l_1^2 \dot\theta_1^2, \\
\dot x_2^2 + \dot y_2^2 &= l_1^2 \dot\theta_1^2 + l_2^2 \dot\theta_2^2
                            + 2 l_1 l_2 \cos(\theta_1 - \theta_2) \dot\theta_1 \dot\theta_2.
\end{aligned}
$$

The cross term is the only place the two degrees of freedom couple at the
kinematic level; every difficulty of the double pendulum traces back to it.
The kinetic energy is therefore

$$
T = \tfrac12 (m_1 + m_2) l_1^2 \dot\theta_1^2
  + \tfrac12 m_2 l_2^2 \dot\theta_2^2
  + m_2 l_1 l_2 \cos(\theta_1 - \theta_2) \dot\theta_1 \dot\theta_2,
$$

and the gravitational potential energy, taking the pivot as the reference, is

$$
V = -(m_1 + m_2) g l_1 \cos\theta_1 - m_2 g l_2 \cos\theta_2.
$$

The Lagrangian is $L = T - V$.

## 3. Mass matrix form

Because $T$ is a homogeneous quadratic in the generalised velocities, we can
write it compactly as

$$
T = \tfrac12\, \dot q^{\top} M(q)\, \dot q,
\qquad q = \begin{pmatrix}\theta_1\\\theta_2\end{pmatrix},
$$

with the symmetric configuration-dependent mass matrix

$$
M(q) = \begin{pmatrix}
(m_1+m_2)\,l_1^2 & m_2\, l_1 l_2 \cos(\theta_1-\theta_2) \\
m_2\, l_1 l_2 \cos(\theta_1-\theta_2) & m_2\, l_2^2
\end{pmatrix}.
$$

The dependence of $M$ on $q$ is what makes the Hamiltonian *non-separable* —
the kinetic term $\tfrac12 p^{\top} M^{-1}(q) p$ couples positions and momenta.
Consequences for numerical integration are discussed in
[`integrators.py`](./integrators.py).

## 4. Legendre transform

The canonical momenta are defined by

$$
p_i = \frac{\partial L}{\partial \dot q_i},
$$

so that

$$
p = \frac{\partial L}{\partial \dot q} = M(q)\,\dot q
\quad\Longleftrightarrow\quad
\dot q = M^{-1}(q)\, p.
$$

Writing this out,

$$
\begin{aligned}
p_1 &= (m_1 + m_2) l_1^2 \dot\theta_1 + m_2 l_1 l_2 \cos(\theta_1 - \theta_2) \dot\theta_2, \\
p_2 &= m_2 l_2^2 \dot\theta_2 + m_2 l_1 l_2 \cos(\theta_1 - \theta_2) \dot\theta_1.
\end{aligned}
$$

The Hamiltonian is the Legendre transform of the Lagrangian,

$$
H(q, p) = p^{\top} \dot q - L(q, \dot q)
       = \tfrac12\, p^{\top} M^{-1}(q)\, p + V(q),
$$

which, for a scleronomic system whose kinetic energy is quadratic in
velocities, equals the total mechanical energy $T + V$.

Introduce the shorthand $\Delta = \theta_1 - \theta_2$ and
$D(q) = m_1 + m_2 \sin^2 \Delta$. The determinant of $M$ is

$$
\det M = (m_1+m_2) l_1^2 \cdot m_2 l_2^2 - m_2^2 l_1^2 l_2^2 \cos^2\Delta
       = m_2\, l_1^2 l_2^2\, D(q),
$$

and

$$
M^{-1}(q) = \frac{1}{m_2 l_1^2 l_2^2 D(q)}
\begin{pmatrix}
m_2 l_2^2 & -m_2 l_1 l_2 \cos\Delta \\
-m_2 l_1 l_2 \cos\Delta & (m_1+m_2) l_1^2
\end{pmatrix}.
$$

## 5. Hamilton's equations

Hamilton's equations are

$$
\dot q = \frac{\partial H}{\partial p}, \qquad
\dot p = -\frac{\partial H}{\partial q}.
$$

The first is immediate: $\dot q = M^{-1}(q) p$. The second requires
differentiating the kinetic term with respect to $q$, using the matrix identity

$$
\frac{\partial M^{-1}}{\partial q_i}
= -M^{-1} \frac{\partial M}{\partial q_i} M^{-1}.
$$

Only the off-diagonal entry of $M$ depends on $q$, and only through
$\cos\Delta$, so

$$
\frac{\partial M}{\partial \theta_1}
= -\frac{\partial M}{\partial \theta_2}
= \begin{pmatrix} 0 & -m_2 l_1 l_2 \sin\Delta \\
                  -m_2 l_1 l_2 \sin\Delta & 0 \end{pmatrix}.
$$

Let $u = M^{-1} p = \dot q$. Then

$$
\frac{\partial}{\partial \theta_1}
\left[\tfrac12 p^{\top} M^{-1} p\right]
= -\tfrac12\, u^{\top} \frac{\partial M}{\partial \theta_1}\, u
= m_2 l_1 l_2 \sin\Delta\; \dot\theta_1 \dot\theta_2,
$$

and the $\theta_2$ derivative flips sign. Combining with the gravity terms,

$$
\boxed{
\begin{aligned}
\dot\theta_i &= \bigl[M^{-1}(q)\, p\bigr]_i, \\
\dot p_1 &= -(m_1 + m_2)\, g\, l_1 \sin\theta_1
           - m_2 l_1 l_2 \sin(\theta_1 - \theta_2)\, \dot\theta_1 \dot\theta_2, \\
\dot p_2 &= -m_2\, g\, l_2 \sin\theta_2
           + m_2 l_1 l_2 \sin(\theta_1 - \theta_2)\, \dot\theta_1 \dot\theta_2.
\end{aligned}
}
$$

These are the equations implemented in
[`double_pendulum.dynamics`](./double_pendulum.py). The HNN will learn the
scalar $H(q, p)$ directly; its vector field is then assembled by autograd as
$(\partial_p H,\, -\partial_q H)$, which by construction has the same symplectic
structure as the equations above.

## 6. Energy and chaos

Because $L$ has no explicit time dependence, Noether's theorem gives
conservation of the Hamiltonian $H$: $\dot H = \{H, H\} = 0$ along trajectories.
This is the single conserved quantity the HNN is asked to respect, and the
energy-drift plot in the README is the direct experimental check.

Conservation of $H$ does **not** imply integrability. The double pendulum has
two degrees of freedom but only one conserved quantity in involution; outside
the small-angle regime its motion is chaotic, exhibiting positive Lyapunov
exponents and exponential sensitivity to initial conditions. This is why it is
the hardest of the three systems studied in Greydanus et al. (2019) and the
natural stress test for a structure-preserving learned dynamics model.

## References

1. Goldstein, Poole, Safko. *Classical Mechanics*, 3rd ed., §§ 1.4, 8.1.
2. José, Saletan. *Classical Dynamics: A Contemporary Approach*, §§ 2.2, 5.1.
3. Greydanus, Dzamba, Yosinski. *Hamiltonian Neural Networks*. NeurIPS 2019.
   [arXiv:1906.01563](https://arxiv.org/abs/1906.01563).
