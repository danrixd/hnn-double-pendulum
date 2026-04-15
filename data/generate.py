r"""Generate supervised training data for the HNN / MLP baselines.

Protocol
--------
For each sampled initial condition we integrate the analytical Hamiltonian
vector field with the symplectic (implicit midpoint) integrator and record the
state at every step along with its analytical time-derivative. Training pairs
are therefore ``(state, d state / dt)`` with state ``= (theta1, theta2, p1, p2)``.

We sample initial conditions from a mild-to-moderately-chaotic regime: angles
uniform in :math:`[-q_{\max}, q_{\max}]` and momenta uniform in
:math:`[-p_{\max}, p_{\max}]`, with ranges chosen so that the total energy
varies across trajectories but the simulator remains numerically well-behaved.
This mirrors the sampling strategy used in Greydanus et al. (2019) for their
double pendulum experiment.

Output
------
A single ``.npz`` archive containing::

    states   : (N, 4) float64     -- canonical states (q, p)
    dstates  : (N, 4) float64     -- analytical d/dt (q, p)
    trajectories : (T, S+1, 4)    -- per-trajectory rollouts (for eval/plots)
    t        : (S+1,) float64     -- shared time grid
    meta     : 0-d object array   -- dict with generation metadata

Training consumers should use ``states`` + ``dstates``; rollouts are kept so
that eval scripts can replay ground truth alongside learned models.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from physics import DEFAULT_PARAMS, dynamics, integrate


def sample_initial_states(
    n: int,
    *,
    q_max: float = 1.5,
    p_max: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n`` initial canonical states from a uniform box."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-q_max, q_max, size=(n, 2))
    p = rng.uniform(-p_max, p_max, size=(n, 2))
    return np.concatenate([q, p], axis=-1)


def rollout(state0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    def ode(s: np.ndarray) -> np.ndarray:
        return dynamics(s, DEFAULT_PARAMS)

    return integrate(ode, state0, dt, n_steps, method="implicit_midpoint")


def build_dataset(
    n_trajectories: int = 200,
    n_steps: int = 200,
    dt: float = 0.01,
    q_max: float = 1.5,
    p_max: float = 1.0,
    seed: int = 0,
) -> dict:
    """Generate the full dataset dictionary described in the module docstring."""
    ics = sample_initial_states(
        n_trajectories, q_max=q_max, p_max=p_max, seed=seed
    )

    trajectories = np.empty((n_trajectories, n_steps + 1, 4), dtype=np.float64)
    for i, ic in enumerate(ics):
        trajectories[i] = rollout(ic, dt, n_steps)

    # Flatten (T, S+1, 4) -> (T*(S+1), 4) and compute analytical derivatives.
    states = trajectories.reshape(-1, 4)
    dstates = dynamics(states, DEFAULT_PARAMS)

    t = np.arange(n_steps + 1, dtype=np.float64) * dt

    meta = {
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "dt": dt,
        "q_max": q_max,
        "p_max": p_max,
        "seed": seed,
        "integrator": "implicit_midpoint",
        "params": asdict(DEFAULT_PARAMS),
    }

    return {
        "states": states,
        "dstates": dstates,
        "trajectories": trajectories,
        "t": t,
        "meta": np.array(json.dumps(meta)),
    }


def save(dataset: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--q-max", type=float, default=1.5)
    parser.add_argument("--p-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/double_pendulum.npz"),
        help="Output path for the .npz archive.",
    )
    args = parser.parse_args()

    dataset = build_dataset(
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        dt=args.dt,
        q_max=args.q_max,
        p_max=args.p_max,
        seed=args.seed,
    )
    save(dataset, args.out)

    print(
        f"wrote {args.out} :: "
        f"{dataset['states'].shape[0]} training pairs, "
        f"{dataset['trajectories'].shape[0]} trajectories of length "
        f"{dataset['trajectories'].shape[1]} at dt={args.dt}"
    )


if __name__ == "__main__":
    main()
