"""Training loop shared by the MLP baseline and the HNN.

Usage
-----
Generate the dataset once::

    python -m data.generate --out data/double_pendulum.npz

Then train either model against it::

    python train.py --model hnn --epochs 2000
    python train.py --model mlp --epochs 2000

Both models learn the same supervised mapping ``(q, p) -> d(q, p)/dt`` against
analytical targets produced by :func:`physics.dynamics`. The only difference
is architectural: the HNN assembles its output from autograd of a scalar
potential, the MLP predicts all four components directly.

The loop is intentionally minimal — full-batch Adam on a fixed train/val
split. This matches the HNN paper's setup and keeps the reproduction script
well under 30 minutes on CPU on the default hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import HNN, MLPBaseline


@dataclass
class TrainConfig:
    model: str = "hnn"
    dataset: Path = Path("data/double_pendulum.npz")
    out_dir: Path = Path("checkpoints")
    hidden_sizes: tuple[int, ...] = (200, 200)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2000
    batch_size: int = 0  # 0 = full-batch
    val_split: float = 0.1
    seed: int = 0
    device: str = "cpu"
    log_every: int = 100


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    return archive["states"], archive["dstates"]


def _split(
    states: np.ndarray, dstates: np.ndarray, val_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(states.shape[0])
    n_val = int(round(val_split * states.shape[0]))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return (
        states[train_idx],
        dstates[train_idx],
        states[val_idx],
        dstates[val_idx],
    )


def _build_model(name: str, hidden_sizes: tuple[int, ...]) -> nn.Module:
    if name == "hnn":
        return HNN(state_dim=4, hidden_sizes=hidden_sizes)
    if name == "mlp":
        return MLPBaseline(state_dim=4, hidden_sizes=hidden_sizes)
    raise ValueError(f"unknown model: {name!r}")


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    states_np, dstates_np = _load_dataset(cfg.dataset)
    x_tr, y_tr, x_val, y_val = _split(states_np, dstates_np, cfg.val_split, cfg.seed)

    device = torch.device(cfg.device)
    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    model = _build_model(cfg.model, cfg.hidden_sizes).to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.batch_size and cfg.batch_size < x_tr_t.shape[0]:
            perm = torch.randperm(x_tr_t.shape[0], device=device)
            total = 0.0
            count = 0
            for start in range(0, x_tr_t.shape[0], cfg.batch_size):
                sel = perm[start : start + cfg.batch_size]
                pred = model(x_tr_t[sel])
                loss = torch.mean((pred - y_tr_t[sel]) ** 2)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                total += loss.item() * sel.shape[0]
                count += sel.shape[0]
            train_loss = total / count
        else:
            pred = model(x_tr_t)
            loss = torch.mean((pred - y_tr_t) ** 2)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss = loss.item()

        # Validation — HNN needs grad for its forward pass, so we don't wrap
        # in torch.no_grad here; we just skip the optimiser step.
        model.eval()
        val_pred = model(x_val_t)
        val_loss = torch.mean((val_pred - y_val_t) ** 2).item()

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            elapsed = time.perf_counter() - t0
            print(
                f"[{cfg.model}] epoch {epoch:5d}/{cfg.epochs}  "
                f"train {train_loss:.3e}  val {val_loss:.3e}  "
                f"({elapsed:.1f}s elapsed)"
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "elapsed_s": elapsed,
                }
            )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.out_dir / f"{cfg.model}.pt"
    torch.save(
        {
            "model": cfg.model,
            "state_dict": model.state_dict(),
            "hidden_sizes": list(cfg.hidden_sizes),
            "history": history,
        },
        ckpt_path,
    )
    print(f"saved checkpoint -> {ckpt_path}")

    return {"model": model, "history": history, "checkpoint": ckpt_path}


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["hnn", "mlp"], default="hnn")
    p.add_argument("--dataset", type=Path, default=Path("data/double_pendulum.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--hidden", type=int, nargs="+", default=[200, 200])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()
    return TrainConfig(
        model=args.model,
        dataset=args.dataset,
        out_dir=args.out_dir,
        hidden_sizes=tuple(args.hidden),
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )


def main() -> None:
    cfg = _parse_args()
    print("config:", json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2))
    train(cfg)


if __name__ == "__main__":
    main()
