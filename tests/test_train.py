import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from deepstego.train import TrainConfig, train


def test_training_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    for split in ("train", "val"):
        folder = tmp_path / split
        folder.mkdir()
        for index in range(2):
            Image.fromarray(rng.integers(0, 256, (136, 136, 3), dtype=np.uint8)).save(
                folder / f"{index}.png"
            )
    config = TrainConfig(
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        output_dir=str(tmp_path / "first"),
        epochs=1,
        batch_size=2,
        width=4,
        num_workers=0,
        crops_per_image=1,
        device="cpu",
    )
    history = train(config)
    assert len(history) == 1
    assert np.isfinite(history[0]["val"]["loss"])
    for filename in ("last.pt", "best.pt", "history.json"):
        assert (tmp_path / "first" / filename).is_file()
    resumed = replace(
        config,
        output_dir=str(tmp_path / "resumed"),
        epochs=2,
        resume=str(tmp_path / "first" / "last.pt"),
    )
    resumed_history = train(resumed)
    full_history = train(replace(config, epochs=2, output_dir=str(tmp_path / "full")))
    assert resumed_history == full_history
    last = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=True)
    assert last["epoch"] == 2
    assert last["optimizer"]["state"]
    for model in ("hide", "reveal"):
        for name in last[model]:
            torch.testing.assert_close(last[model][name], full[model][name])
    assert json.loads((tmp_path / "resumed" / "history.json").read_text()) == resumed_history
    best = torch.load(tmp_path / "resumed" / "best.pt", weights_only=True)
    assert best["best_loss"] == min(row["val"]["loss"] for row in resumed_history)
    with pytest.raises(ValueError, match="ya tiene checkpoints"):
        train(config)
    with pytest.raises(ValueError, match="comparten archivos"):
        train(replace(config, val_dir=config.train_dir))
    with pytest.raises(ValueError, match="beta debe coincidir"):
        train(replace(resumed, beta=2, output_dir=str(tmp_path / "bad")))
