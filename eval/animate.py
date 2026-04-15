"""Side-by-side animation of (truth, MLP, HNN) rollouts as physical pendulums.

The README hero GIF comes from this script. From a single initial condition
we roll out each vector field with its matched integrator (symplectic for
truth, RK4 for both learned models) and animate the three pendulums swinging
next to each other, with a fading trail behind each lower bob so the eye can
follow the divergence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from eval import ground_truth_ode, load_checkpoint, model_to_ode, rollout
from physics import DEFAULT_PARAMS


def _positions(theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, ...]:
    l1, l2 = DEFAULT_PARAMS.l1, DEFAULT_PARAMS.l2
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def _rollout_labels(
    model_ckpts: dict[str, Path], state0: np.ndarray, dt: float, n_steps: int
) -> dict[str, np.ndarray]:
    out = {
        "Ground truth (symplectic)": rollout(
            ground_truth_ode(), state0, dt, n_steps, method="implicit_midpoint"
        )
    }
    for label, ckpt in model_ckpts.items():
        model = load_checkpoint(ckpt)
        out[label] = rollout(model_to_ode(model), state0, dt, n_steps, method="rk4")
    return out


def build_animation(
    trajectories: dict[str, np.ndarray],
    *,
    dt: float,
    trail: int = 80,
    fps: int = 25,
) -> animation.FuncAnimation:
    labels = list(trajectories.keys())
    n_panels = len(labels)
    n_frames = next(iter(trajectories.values())).shape[0]

    palette = {
        "Ground truth (symplectic)": "#444444",
        "MLP baseline": "#d62728",
        "HNN": "#1f77b4",
    }

    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.6 * n_panels, 3.9), dpi=120, sharey=True
    )
    if n_panels == 1:
        axes = [axes]

    # Pre-compute bob positions for each panel.
    panel_positions = {}
    limit = DEFAULT_PARAMS.l1 + DEFAULT_PARAMS.l2 + 0.2
    for label, ax in zip(labels, axes):
        traj = trajectories[label]
        x1, y1, x2, y2 = _positions(traj[:, 0], traj[:, 1])
        panel_positions[label] = (x1, y1, x2, y2)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, 0.4)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11, color=palette.get(label, "black"))
        ax.axhline(0, color="#cccccc", linewidth=0.5)
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Drawable artists.
    rods1, rods2, trails = {}, {}, {}
    bobs = {}
    for label, ax in zip(labels, axes):
        color = palette.get(label, "black")
        (rod1,) = ax.plot([], [], color="#333333", linewidth=1.8)
        (rod2,) = ax.plot([], [], color="#333333", linewidth=1.8)
        (trail_line,) = ax.plot([], [], color=color, linewidth=1.2, alpha=0.65)
        bob_scatter = ax.scatter(
            [0.0, 0.0], [0.0, 0.0], s=[40, 70], color=color, zorder=5
        )
        rods1[label] = rod1
        rods2[label] = rod2
        trails[label] = trail_line
        bobs[label] = bob_scatter

    time_text = fig.suptitle("", fontsize=11)

    def init():
        artists = []
        for label in labels:
            rods1[label].set_data([], [])
            rods2[label].set_data([], [])
            trails[label].set_data([], [])
            bobs[label].set_offsets(np.zeros((2, 2)))
            artists += [rods1[label], rods2[label], trails[label], bobs[label]]
        time_text.set_text("t = 0.00 s")
        return artists + [time_text]

    def update(frame: int):
        artists = []
        for label in labels:
            x1, y1, x2, y2 = panel_positions[label]
            rods1[label].set_data([0.0, x1[frame]], [0.0, y1[frame]])
            rods2[label].set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
            lo = max(0, frame - trail)
            trails[label].set_data(x2[lo : frame + 1], y2[lo : frame + 1])
            bobs[label].set_offsets(
                np.array([[x1[frame], y1[frame]], [x2[frame], y2[frame]]])
            )
            artists += [rods1[label], rods2[label], trails[label], bobs[label]]
        time_text.set_text(f"t = {frame * dt:5.2f} s")
        return artists + [time_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()
    return anim


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mlp-ckpt", type=Path, default=Path("checkpoints/mlp.pt"))
    p.add_argument("--hnn-ckpt", type=Path, default=Path("checkpoints/hnn.pt"))
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--trail", type=int, default=80)
    p.add_argument(
        "--out", type=Path, default=Path("docs/animations/double_pendulum_compare.gif")
    )
    p.add_argument("--subsample", type=int, default=2,
                   help="Keep every k-th frame in the saved GIF to shrink file size.")
    args = p.parse_args()

    state0 = np.array([1.3, -0.9, 0.0, 0.0])
    trajectories = _rollout_labels(
        {"MLP baseline": args.mlp_ckpt, "HNN": args.hnn_ckpt},
        state0, args.dt, args.n_steps,
    )

    # Subsample for GIF file size (drops uniform frames; visual pacing preserved
    # by lowering the effective dt displayed in the title accordingly).
    trajectories_sub = {k: v[:: args.subsample] for k, v in trajectories.items()}
    effective_dt = args.dt * args.subsample

    anim = build_animation(
        trajectories_sub, dt=effective_dt, trail=args.trail, fps=args.fps
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer)
    plt.close("all")
    print(f"wrote animation -> {args.out}")


if __name__ == "__main__":
    main()
