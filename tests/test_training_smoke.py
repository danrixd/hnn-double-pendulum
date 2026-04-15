"""End-to-end training smoke test.

Generates a tiny dataset, trains each model for a handful of epochs, and
checks that the validation loss goes down. This doesn't prove the HNN wins
the long-rollout comparison — that's what the eval scripts are for — it only
proves the data → train → checkpoint pipeline is wired correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.generate import build_dataset, save
from train import TrainConfig, train


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data") / "tiny.npz"
    dataset = build_dataset(
        n_trajectories=5, n_steps=20, dt=0.01, q_max=1.0, p_max=0.5, seed=0
    )
    save(dataset, out)
    return out


@pytest.mark.parametrize("model_name", ["mlp", "hnn"])
def test_training_reduces_loss(tiny_dataset: Path, tmp_path: Path, model_name: str):
    cfg = TrainConfig(
        model=model_name,
        dataset=tiny_dataset,
        out_dir=tmp_path / "ckpt",
        hidden_sizes=(32, 32),
        lr=5e-3,
        weight_decay=0.0,
        epochs=300,
        batch_size=0,
        val_split=0.2,
        seed=0,
        device="cpu",
        log_every=20,
    )
    result = train(cfg)
    history = result["history"]
    assert len(history) >= 2
    first = history[0]["val_loss"]
    last = history[-1]["val_loss"]
    assert last < first * 0.5, (
        f"{model_name} val loss did not drop by >=2x: {first:.3e} -> {last:.3e}"
    )
    assert result["checkpoint"].exists()
