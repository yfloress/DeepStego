import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import deepstego.train as training_module
from deepstego.data import DIV2KDataset
from deepstego.qr import generate_qr_array
from deepstego.train import TrainConfig, _run_epoch, train


@pytest.fixture
def small_config(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    for split in ("train", "val"):
        folder = tmp_path / split
        folder.mkdir()
        for index in range(2):
            Image.fromarray(rng.integers(0, 256, (136, 136, 3), dtype=np.uint8)).save(
                folder / f"{index}.png"
            )
    return TrainConfig(
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


@pytest.mark.parametrize("secret_loss,lr_patience", [("mse", None), ("bce", 0)])
def test_training_checkpoint_resume_matches_uninterrupted_run(
    tmp_path, small_config, secret_loss, lr_patience
):
    config = replace(small_config, secret_loss=secret_loss, lr_patience=lr_patience)
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


def test_finetuning_loads_weights_but_resets_optimizer_and_history(
    tmp_path, small_config, monkeypatch
):
    train(small_config)
    source = tmp_path / "first" / "best.pt"
    original_bytes = source.read_bytes()
    original = torch.load(source, weights_only=True)
    run_epoch = training_module._run_epoch
    inspected = []

    def inspect_initial_state(hide, reveal, loader, device, beta, optimizer=None, **kwargs):
        if optimizer is not None and not inspected:
            for model, key in ((hide, "hide"), (reveal, "reveal")):
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[key][name])
            assert not optimizer.state
            assert optimizer.param_groups[0]["lr"] == 1e-4
            inspected.append(True)
        return run_epoch(hide, reveal, loader, device, beta, optimizer, **kwargs)

    monkeypatch.setattr(training_module, "_run_epoch", inspect_initial_state)
    config = replace(
        small_config,
        output_dir=str(tmp_path / "finetuned"),
        init_from=str(source),
        learning_rate=1e-4,
        beta=0.1,
        secret_loss="bce",
    )
    history = train(config)
    assert inspected
    assert len(history) == 1 and history[0]["epoch"] == 1
    checkpoint = torch.load(tmp_path / "finetuned" / "last.pt", weights_only=True)
    assert checkpoint["data_epoch_offset"] == original["epoch"]
    assert all(state["step"] == 1 for state in checkpoint["optimizer"]["state"].values())
    assert source.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="no ambos"):
        train(replace(config, resume=str(source)))
    with pytest.raises(ValueError, match="ya tiene checkpoints"):
        train(config)


def test_validation_requires_exact_message_and_detects_white_output():
    secret = torch.from_numpy(generate_qr_array("Hola")).repeat(2, 1, 1, 1)
    batch = {
        "cover": secret.expand(-1, 3, -1, -1) * 0.8 + 0.1,
        "secret": secret,
        "text": ["Hola", "Otro"],
    }

    class VisibleQR(torch.nn.Module):
        def forward(self, cover, secret):
            return secret.expand(-1, 3, -1, -1)

    class ReadChannel(torch.nn.Module):
        def forward_logits(self, container):
            return (container[:, :1] - 0.5) * 20

    class WhiteQR(torch.nn.Module):
        def forward_logits(self, container):
            return torch.full_like(container[:, :1], 100)

    metrics = _run_epoch(VisibleQR(), ReadChannel(), [batch], "cpu", 1)
    assert metrics["reading_rate"] == 0.5
    assert metrics["ssim"] > 0.9
    assert metrics["collapsed_fraction"] == 0
    collapsed = _run_epoch(VisibleQR(), WhiteQR(), [batch], "cpu", 1)
    assert collapsed["reading_rate"] == 0
    assert collapsed["collapsed_fraction"] == 1
    assert collapsed["secret_mse"] == pytest.approx((secret == 0).float().mean().item())


def fake_metrics(**changes):
    return {
        "loss": 1.0,
        "cover_mse": 0.01,
        "secret_mse": 0.01,
        "secret_loss": 0.01,
        "psnr": 20.0,
        "ssim": 0.8,
        "reading_rate": 0.0,
        "collapsed_fraction": 0.0,
        **changes,
    }


def test_quality_checkpoint_requires_reading_and_ssim(tmp_path, small_config, monkeypatch):
    validations = iter(
        [
            fake_metrics(loss=0.1, psnr=40, reading_rate=0.89, ssim=0.95),
            fake_metrics(loss=0.2, psnr=29, reading_rate=0.90, ssim=0.90),
            fake_metrics(loss=0.3, psnr=31, reading_rate=0.95, ssim=0.95),
            fake_metrics(loss=0.4, psnr=35, reading_rate=1, ssim=0.89),
        ]
    )

    def run_epoch(hide, reveal, loader, device, beta, optimizer=None, **kwargs):
        return fake_metrics() if optimizer is not None else next(validations)

    monkeypatch.setattr(training_module, "_run_epoch", run_epoch)
    train(replace(small_config, epochs=4))
    best = torch.load(tmp_path / "first" / "best.pt", weights_only=True)
    quality = torch.load(tmp_path / "first" / "best_quality.pt", weights_only=True)
    assert best["epoch"] == 1
    assert quality["epoch"] == 3
    assert quality["best_quality_psnr"] == 31


@pytest.mark.parametrize(
    "collapse,epochs,reason", [(False, 3, "Parada temprana"), (True, 1, "Colapso")]
)
def test_stops_on_stagnation_or_collapse(
    tmp_path, small_config, monkeypatch, collapse, epochs, reason
):
    monkeypatch.setattr(
        training_module,
        "_run_epoch",
        lambda *args, **kwargs: fake_metrics(collapsed_fraction=float(collapse)),
    )
    history = train(replace(small_config, epochs=10, early_stop_patience=2))
    checkpoint = torch.load(tmp_path / "first" / "last.pt", weights_only=True)
    assert len(history) == epochs
    assert checkpoint["stop_reason"].startswith(reason)
    assert not (tmp_path / "first" / "best_quality.pt").exists()


def test_scheduler_and_early_stopping_survive_resume(tmp_path, small_config, monkeypatch):
    monkeypatch.setattr(training_module, "_run_epoch", lambda *args, **kwargs: fake_metrics())
    config = replace(small_config, epochs=2, lr_patience=0, early_stop_patience=3)
    train(config)
    resumed = train(
        replace(
            config,
            epochs=10,
            resume=str(tmp_path / "first" / "last.pt"),
            output_dir=str(tmp_path / "resumed"),
        )
    )
    full = train(replace(config, epochs=10, output_dir=str(tmp_path / "full")))
    assert resumed == full
    assert len(resumed) == 4
    assert [row["learning_rate"] for row in resumed] == [0.001, 0.001, 0.0005, 0.00025]


def test_legacy_checkpoint_can_resume(tmp_path, small_config):
    train(small_config)
    path = tmp_path / "first" / "last.pt"
    checkpoint = torch.load(path, weights_only=True)
    for field in ("init_from", "secret_loss", "lr_patience", "early_stop_patience"):
        checkpoint["config"].pop(field)
    for key in ("scheduler", "best_quality_psnr", "bad_epochs", "data_epoch_offset", "stop_reason"):
        checkpoint.pop(key)
    torch.save(checkpoint, path)
    history = train(replace(small_config, epochs=2, resume=str(path)))
    assert len(history) == 2


def test_notebook_experiment_cells_run_on_cpu(tmp_path, small_config, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    train(small_config)
    project = tmp_path / "project"
    (project / "checkpoints").mkdir(parents=True)
    shutil.copy2(tmp_path / "first" / "best.pt", project / "checkpoints" / "best.pt")
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "train_kaggle.ipynb").read_text()
    )
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    namespace = {
        "Path": Path,
        "torch": torch,
        "plt": plt,
        "PROJECT_DIR": project,
        "WORK_ROOT": tmp_path,
        "TRAIN_DIR": small_config.train_dir,
        "VAL_DIR": small_config.val_dir,
        "SEED": 42,
        "val_data": DIV2KDataset(small_config.val_dir, training=False, seed=43),
    }
    for cell_id in ("configuration", "baseline", "training", "curves", "evaluation", "png-example"):
        source = "".join(cells[cell_id]["source"]).replace('"cuda"', '"cpu"')
        exec(compile(source, f"notebook:{cell_id}", "exec"), namespace)
        if cell_id == "configuration":
            namespace["config"] = replace(
                namespace["config"], epochs=1, num_workers=0, batch_size=2, crops_per_image=1
            )
    output = namespace["OUTPUT_DIR"]
    assert (output / "curves.png").is_file()
    assert (output / "examples.png").is_file()
    assert (namespace["evaluation_root"] / "comparison.csv").is_file()
    assert namespace["baseline"]["seed"] == 43
    assert namespace["summaries"]["best.pt"]["seed"] == 43
    assert len(namespace["history"]) == 1
    plt.close("all")
