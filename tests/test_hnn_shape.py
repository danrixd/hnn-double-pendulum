"""Architectural checks for the HNN and MLP baseline.

We verify the forward-pass contract (shape, symplectic assembly) and a
minimal correctness property of the HNN: its vector field really is the
symplectic gradient of its scalar output. This catches sign bugs in the
autograd assembly — the single most likely place for the HNN to silently
degrade into "just another MLP".
"""

from __future__ import annotations

import torch

from models import HNN, MLPBaseline


def test_mlp_output_shape():
    net = MLPBaseline(state_dim=4, hidden_sizes=(32, 32))
    state = torch.randn(17, 4)
    out = net(state)
    assert out.shape == state.shape


def test_hnn_output_shape_batched():
    net = HNN(state_dim=4, hidden_sizes=(32, 32))
    # Arbitrary leading dimensions — HNN should broadcast over all of them.
    state = torch.randn(5, 7, 4)
    out = net(state)
    assert out.shape == state.shape


def test_hnn_vector_field_matches_scalar_gradient():
    """HNN forward output must equal ``(∂H/∂p, −∂H/∂q)`` of its scalar net."""
    torch.manual_seed(0)
    net = HNN(state_dim=4, hidden_sizes=(32, 32)).eval()

    state = torch.randn(8, 4, requires_grad=True)
    H = net.hamiltonian(state).sum()
    grad_H = torch.autograd.grad(H, state)[0]
    expected = torch.cat([grad_H[:, 2:], -grad_H[:, :2]], dim=-1)

    got = net(state.detach())
    torch.testing.assert_close(got, expected.detach(), atol=1e-6, rtol=1e-6)


def test_hnn_conserves_its_own_hamiltonian_along_rollout():
    """Rolling out the HNN's own field with small dt should conserve its H.

    This is a free property of the architecture: since ṗ = -∂H/∂q and
    q̇ = ∂H/∂p, the directional derivative of H along the field is
    identically zero. Any drift in a short Euler rollout therefore comes from
    discretisation error alone — not from a broken symplectic assembly.
    """
    torch.manual_seed(0)
    net = HNN(state_dim=4, hidden_sizes=(32, 32)).eval()

    state = torch.randn(1, 4)
    dt = 1e-3
    H0 = net.hamiltonian(state).item()
    for _ in range(200):
        with torch.no_grad():
            pass  # field call below re-enables grad internally
        field = net(state.detach())
        state = state.detach() + dt * field
    H_end = net.hamiltonian(state).item()

    assert abs(H_end - H0) < 1e-3, f"|ΔH| = {abs(H_end - H0):.3e}"
