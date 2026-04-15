r"""Hamiltonian Neural Network for the double pendulum.

Following Greydanus, Dzamba and Yosinski (NeurIPS 2019), the network
parameterises the *scalar* Hamiltonian :math:`H_\theta(q, p)` rather than the
vector field directly. The dynamics are then recovered by autograd:

.. math::

    \frac{dq}{dt} = \frac{\partial H_\theta}{\partial p}, \qquad
    \frac{dp}{dt} = -\frac{\partial H_\theta}{\partial q}.

Assembling the vector field from the gradients of a scalar guarantees the
symplectic structure of Hamilton's equations holds exactly by construction,
irrespective of the values of the network parameters. Training only has to
shape :math:`H_\theta` so that its gradients match the observed derivatives;
it never has to *learn* energy conservation — that comes for free from the
architecture.

Training loss
-------------
Given training pairs ``(state_i, dstate_i)``, the loss is a plain L2 between
the autograd-derived vector field and the analytical derivative:

.. math::

    \mathcal{L}(\theta) = \frac{1}{N} \sum_i
        \bigl\| (\partial_p H_\theta,\ -\partial_q H_\theta)(state_i)
                - dstate_i \bigr\|^2.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class HNN(nn.Module):
    """Hamiltonian neural network that learns a scalar ``H(q, p)``.

    The forward pass returns the symplectic vector field
    ``(∂H/∂p, −∂H/∂q)`` obtained by autograd of the scalar output. The
    underlying scalar network is available via :meth:`hamiltonian`.

    Parameters
    ----------
    state_dim : int
        Full canonical state dimension (must be even). The first half is
        treated as positions, the second half as conjugate momenta.
    hidden_sizes : sequence of int
        Hidden-layer widths of the scalar network.
    activation : type[nn.Module]
        Smooth activation — autograd through the scalar network requires this
        to be twice differentiable in practice, so ``Tanh`` is the canonical
        choice.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_sizes: Sequence[int] = (200, 200),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError(f"state_dim must be even, got {state_dim}")

        dims = (state_dim, *hidden_sizes, 1)
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(d_in, d_out)
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(activation())
        self.scalar_net = nn.Sequential(*layers)
        self.state_dim = state_dim
        self._half = state_dim // 2

    def hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """Scalar Hamiltonian value for each row of ``state``.

        Returns a tensor of shape ``state.shape[:-1]``.
        """
        return self.scalar_net(state).squeeze(-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Symplectic vector field ``(∂H/∂p, −∂H/∂q)`` at ``state``.

        ``state`` can be any shape ``(..., state_dim)``. The output has the
        same shape. Works with or without an outer ``torch.no_grad`` context
        because we locally re-enable grad on the input.
        """
        with torch.enable_grad():
            state = state.requires_grad_(True)
            H = self.hamiltonian(state).sum()
            grad_H = torch.autograd.grad(
                H,
                state,
                create_graph=self.training,
            )[0]

        dH_dq = grad_H[..., : self._half]
        dH_dp = grad_H[..., self._half :]
        # Symplectic assembly: q̇ = +∂H/∂p, ṗ = -∂H/∂q.
        return torch.cat([dH_dp, -dH_dq], dim=-1)
