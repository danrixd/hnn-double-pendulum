"""Sensitivity to initial conditions.

The double pendulum is chaotic in the high-energy regime: two trajectories
seeded from states that differ by :math:`\\epsilon` diverge roughly as
:math:`\\epsilon \\cdot e^{\\lambda t}` with a positive largest Lyapunov
exponent. A faithful learned dynamics model should reproduce that behaviour,
not damp or explode it.

We pick a single initial condition and a tiny perturbation ``epsilon * u``
along a random direction, roll out both seeds under each of (truth, MLP,
HNN), and plot the separation ``||y_1(t) - y_2(t)||_2`` over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval import ground_truth_ode, load_checkpoint, model_to_ode, rollout


def compute_separations(
    model_ckpts: dict[str, Path],
    *,
    state0: np.ndarray,
    epsilon: float,
    n_steps: int,
    dt: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=state0.shape)
    direction /= np.linalg.norm(direction)
    state0_b = state0 + epsilon * direction

    def _pair(ode, method):
        a = rollout(ode, state0, dt, n_steps, method=method)
        b = rollout(ode, state0_b, dt, n_steps, method=method)
        return np.linalg.norm(a - b, axis=-1)

    out = {"truth (symplectic)": _pair(ground_truth_ode(), "implicit_midpoint")}
    for label, ckpt in model_ckpts.items():
        model = load_checkpoint(ckpt)
        out[label] = _pair(model_to_ode(model), "rk4")
    return out


def plot(series: dict[str, np.ndarray], dt: float, epsilon: float, out_path: Path) -> None:
    n_steps = next(iter(series.values())).shape[0]
    t = np.arange(n_steps) * dt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    palette = {
        "truth (symplectic)": "#666666",
        "MLP baseline": "#d62728",
        "HNN": "#1f77b4",
    }
    for label, curve in series.items():
        ax.plot(t, curve, label=label, color=palette.get(label), linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel(r"separation $\|y_1(t) - y_2(t)\|_2$")
    ax.set_title(
        f"Sensitivity to initial conditions "
        f"(initial separation $\\epsilon = {epsilon:g}$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mlp-ckpt", type=Path, default=Path("checkpoints/mlp.pt"))
    p.add_argument("--hnn-ckpt", type=Path, default=Path("checkpoints/hnn.pt"))
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--out", type=Path, default=Path("docs/screenshots/sensitivity.png")
    )
    args = p.parse_args()

    state0 = np.array([1.2, -0.5, 0.3, 0.2])
    series = compute_separations(
        {"MLP baseline": args.mlp_ckpt, "HNN": args.hnn_ckpt},
        state0=state0,
        epsilon=args.epsilon,
        n_steps=args.n_steps,
        dt=args.dt,
        seed=args.seed,
    )
    for label, curve in series.items():
        print(f"  {label:22s}  final separation = {curve[-1]:.3e}")
    plot(series, args.dt, args.epsilon, args.out)
    print(f"wrote figure -> {args.out}")


if __name__ == "__main__":
    main()
