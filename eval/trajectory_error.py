"""Trajectory error vs rollout length.

Complements the headline energy-drift figure. Even a model that conserves
energy can drift along the energy surface (rotating around the torus the
motion actually lives on), so state-space L2 error is a second, independent
axis of comparison.

For each of a batch of held-out initial conditions we compute the
per-timestep L2 distance between the learned rollout and the symplectic
ground truth, then plot the median / IQR across the batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval import ground_truth_ode, load_checkpoint, model_to_ode, rollout


def _sample_initial_states(n: int, *, q_max: float, p_max: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.uniform(-q_max, q_max, size=(n, 2))
    p = rng.uniform(-p_max, p_max, size=(n, 2))
    return np.concatenate([q, p], axis=-1)


def compute_errors(
    model_ckpts: dict[str, Path],
    *,
    n_states: int,
    n_steps: int,
    dt: float,
    q_max: float,
    p_max: float,
    seed: int,
) -> dict[str, np.ndarray]:
    states0 = _sample_initial_states(n_states, q_max=q_max, p_max=p_max, seed=seed)

    gt_ode = ground_truth_ode()
    truth_trajs = np.empty((n_states, n_steps + 1, 4))
    for i, s0 in enumerate(states0):
        truth_trajs[i] = rollout(gt_ode, s0, dt, n_steps, method="implicit_midpoint")

    out: dict[str, np.ndarray] = {}
    for label, ckpt in model_ckpts.items():
        model = load_checkpoint(ckpt)
        ode = model_to_ode(model)
        err = np.empty((n_states, n_steps + 1))
        for i, s0 in enumerate(states0):
            learned = rollout(ode, s0, dt, n_steps, method="rk4")
            err[i] = np.linalg.norm(learned - truth_trajs[i], axis=-1)
        out[label] = err

    return out


def plot(errors: dict[str, np.ndarray], dt: float, out_path: Path) -> None:
    n_steps = next(iter(errors.values())).shape[1]
    t = np.arange(n_steps) * dt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    palette = {"MLP baseline": "#d62728", "HNN": "#1f77b4"}
    for label, curves in errors.items():
        color = palette.get(label)
        median = np.median(curves, axis=0)
        q1 = np.quantile(curves, 0.25, axis=0)
        q3 = np.quantile(curves, 0.75, axis=0)
        ax.plot(t, median, label=label, color=color, linewidth=2)
        ax.fill_between(t, q1, q3, color=color, alpha=0.18, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel(r"state error $\|\hat y(t) - y(t)\|_2$")
    ax.set_title("Double pendulum: state-space error of learned rollouts")
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
    p.add_argument("--n-states", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--q-max", type=float, default=1.5)
    p.add_argument("--p-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=Path("docs/screenshots/trajectory_error.png"))
    args = p.parse_args()

    errors = compute_errors(
        {"MLP baseline": args.mlp_ckpt, "HNN": args.hnn_ckpt},
        n_states=args.n_states,
        n_steps=args.n_steps,
        dt=args.dt,
        q_max=args.q_max,
        p_max=args.p_max,
        seed=args.seed,
    )
    for label, curves in errors.items():
        final = np.median(curves[:, -1])
        print(f"  {label:22s}  median final state error = {final:.3e}")
    plot(errors, args.dt, args.out)
    print(f"wrote figure -> {args.out}")


if __name__ == "__main__":
    main()
