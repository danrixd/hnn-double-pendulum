r"""Headline figure: energy drift vs rollout length.

For each of a batch of held-out initial conditions we roll the same state
forward in time under (a) the analytical Hamiltonian vector field, (b) the
vanilla MLP baseline, and (c) the HNN. All three learned rollouts use the
same RK4 step size — the only thing that changes between (b) and (c) is the
vector field. For each rollout we then plot the *true* Hamiltonian evaluated
along the predicted trajectory, normalised to its initial value.

The expected result (reproducing Greydanus et al. 2019 on this system) is
that the MLP baseline's energy drifts by tens of percent over 10^3–10^4 steps,
while the HNN stays within a couple of percent — the headline win of
imposing symplectic structure as an architectural prior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval import (
    ground_truth_ode,
    load_checkpoint,
    model_to_ode,
    relative_energy_drift,
    rollout,
)


def _sample_initial_states(n: int, *, q_max: float, p_max: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.uniform(-q_max, q_max, size=(n, 2))
    p = rng.uniform(-p_max, p_max, size=(n, 2))
    return np.concatenate([q, p], axis=-1)


def compute_drifts(
    model_ckpts: dict[str, Path],
    *,
    n_states: int,
    n_steps: int,
    dt: float,
    q_max: float,
    p_max: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return ``{label: drift_curves}`` with shape ``(n_states, n_steps + 1)``."""
    states0 = _sample_initial_states(n_states, q_max=q_max, p_max=p_max, seed=seed)

    # Ground truth: implicit midpoint for clean reference energy.
    gt_ode = ground_truth_ode()
    truth_drift = np.empty((n_states, n_steps + 1))
    for i, s0 in enumerate(states0):
        traj = rollout(gt_ode, s0, dt, n_steps, method="implicit_midpoint")
        truth_drift[i] = relative_energy_drift(traj)

    out: dict[str, np.ndarray] = {"truth (symplectic)": truth_drift}
    for label, ckpt_path in model_ckpts.items():
        model = load_checkpoint(ckpt_path)
        ode = model_to_ode(model)
        drift = np.empty((n_states, n_steps + 1))
        for i, s0 in enumerate(states0):
            traj = rollout(ode, s0, dt, n_steps, method="rk4")
            drift[i] = relative_energy_drift(traj)
        out[label] = drift

    return out


def plot(
    drifts: dict[str, np.ndarray],
    dt: float,
    out_path: Path,
) -> None:
    steps = next(iter(drifts.values())).shape[1]
    t = np.arange(steps) * dt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    # Deterministic colour per label so MLP/HNN keep their identities across
    # the repo's figures.
    palette = {
        "truth (symplectic)": "#666666",
        "MLP baseline": "#d62728",
        "HNN": "#1f77b4",
    }
    for label, curves in drifts.items():
        color = palette.get(label, None)
        median = np.median(curves, axis=0)
        q1 = np.quantile(curves, 0.25, axis=0)
        q3 = np.quantile(curves, 0.75, axis=0)
        ax.plot(t, median, label=label, color=color, linewidth=2)
        ax.fill_between(t, q1, q3, color=color, alpha=0.18, linewidth=0)

    ax.set_yscale("log")
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel(r"relative energy drift $|H(t) - H_0| / |H_0|$")
    ax.set_title("Double pendulum: energy conservation under learned dynamics")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlp-ckpt", type=Path, default=Path("checkpoints/mlp.pt")
    )
    parser.add_argument(
        "--hnn-ckpt", type=Path, default=Path("checkpoints/hnn.pt")
    )
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--q-max", type=float, default=1.5)
    parser.add_argument("--p-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/screenshots/energy_drift.png"),
    )
    parser.add_argument(
        "--out-data",
        type=Path,
        default=Path("docs/screenshots/energy_drift.npz"),
    )
    args = parser.parse_args()

    drifts = compute_drifts(
        {"MLP baseline": args.mlp_ckpt, "HNN": args.hnn_ckpt},
        n_states=args.n_states,
        n_steps=args.n_steps,
        dt=args.dt,
        q_max=args.q_max,
        p_max=args.p_max,
        seed=args.seed,
    )

    summary = {
        label: {
            "median_final": float(np.median(curves[:, -1])),
            "p75_final": float(np.quantile(curves[:, -1], 0.75)),
            "mean_final": float(np.mean(curves[:, -1])),
        }
        for label, curves in drifts.items()
    }
    print("relative energy drift at final step:")
    for label, stats in summary.items():
        print(
            f"  {label:22s}  median={stats['median_final']:.3e}  "
            f"p75={stats['p75_final']:.3e}  mean={stats['mean_final']:.3e}"
        )

    plot(drifts, args.dt, args.out)
    print(f"wrote figure -> {args.out}")

    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_data, dt=args.dt, **{k.replace(" ", "_"): v for k, v in drifts.items()})
    print(f"wrote raw data -> {args.out_data}")


if __name__ == "__main__":
    main()
