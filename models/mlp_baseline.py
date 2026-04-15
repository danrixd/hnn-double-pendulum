"""Vanilla MLP baseline for learning the double pendulum vector field.

Input: canonical state ``(q, p)`` of dimension 4.
Output: time derivative ``d(q, p)/dt`` of the same dimension.

This is the "no structural prior" comparator. It is trained with an L2 loss
against the analytical derivative at each training state, so the only thing
distinguishing it from the HNN is the absence of Hamilton's symplectic
structure — which is exactly what we want the headline energy-drift figure to
highlight.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Plain feed-forward network mapping ``(q, p) -> d(q, p)/dt``.

    Parameters
    ----------
    state_dim : int
        Dimension of the canonical state. Defaults to 4 for the double
        pendulum.
    hidden_sizes : sequence of int
        Sizes of the hidden layers. Defaults to two hidden layers of 200 units,
        matching the MLP baseline used in the HNN paper.
    activation : type[nn.Module]
        Activation class to instantiate between layers. ``nn.Tanh`` is chosen
        for smoothness so the two baselines are compared on equal footing
        (the HNN also uses ``Tanh``).
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_sizes: Sequence[int] = (200, 200),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        dims = (state_dim, *hidden_sizes, state_dim)
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(d_in, d_out)
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:  # no activation on the output layer
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
