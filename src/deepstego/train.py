"""Entrenamiento conjunto y checkpoints por época."""

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deepstego.data import DIV2KDataset
from deepstego.metrics import image_quality
from deepstego.models import HideNet, RevealNet, quantize_8bit
from deepstego.qr import read_qr


@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    output_dir: str = "outputs/m2"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta: float = 1.0
    width: int = 32
    crops_per_image: int = 4
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"
    resume: str | None = None
    init_from: str | None = None
    secret_loss: str = "mse"
    lr_patience: int | None = None
    early_stop_patience: int = 0


def _run_epoch(hide, reveal, loader, device, beta, optimizer=None, *, secret_loss="mse"):
    training = optimizer is not None
    hide.train(training)
    reveal.train(training)
    totals = dict.fromkeys(("loss", "cover_mse", "secret_mse", "secret_loss", "psnr"), 0.0)
    if not training:
        totals.update(ssim=0.0, reading_rate=0.0, collapsed_fraction=0.0)
    samples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            cover = batch["cover"].to(device)
            secret = batch["secret"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            container = quantize_8bit(hide(cover, secret))
            logits = reveal.forward_logits(container)
            recovered = logits.sigmoid()
            cover_mse = F.mse_loss(container, cover)
            secret_mse = F.mse_loss(recovered, secret)
            qr_loss = (
                F.binary_cross_entropy_with_logits(logits, secret)
                if secret_loss == "bce"
                else secret_mse
            )
            loss = cover_mse + beta * qr_loss
            if not torch.isfinite(loss):
                raise RuntimeError("La pérdida no es finita; revisa datos e hiperparámetros.")
            if training:
                loss.backward()
                optimizer.step()
            count = cover.shape[0]
            per_image_mse = (container.detach() - cover).square().mean(dim=(1, 2, 3))
            psnr = -10 * torch.log10(per_image_mse.clamp_min(1e-12))
            for key, value in (
                ("loss", loss),
                ("cover_mse", cover_mse),
                ("secret_mse", secret_mse),
                ("secret_loss", qr_loss),
                ("psnr", psnr.mean()),
            ):
                totals[key] += value.item() * count
            if not training:
                covers = cover.cpu().permute(0, 2, 3, 1).numpy()
                containers = container.cpu().permute(0, 2, 3, 1).numpy()
                recovered_cpu = recovered.cpu()
                for index, text in enumerate(batch["text"]):
                    totals["ssim"] += image_quality(covers[index], containers[index])["ssim"]
                    totals["reading_rate"] += read_qr(recovered_cpu[index]).text == text
                    prediction = recovered_cpu[index]
                    totals["collapsed_fraction"] += int(
                        prediction.min() >= 1 - 1e-6 or prediction.max() <= 1e-6
                    )
            samples += count
    return {key: value / samples for key, value in totals.items()}


def _save_checkpoint(state, path: Path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config: TrainConfig) -> list[dict]:
    for field in ("epochs", "batch_size", "width", "crops_per_image"):
        if getattr(config, field) < 1:
            raise ValueError(f"{field} debe ser positivo.")
    if config.num_workers < 0:
        raise ValueError("num_workers no puede ser negativo.")
    if config.resume and config.init_from:
        raise ValueError("Usa resume o init_from, no ambos.")
    if config.secret_loss not in {"mse", "bce"}:
        raise ValueError("secret_loss debe ser mse o bce.")
    if config.lr_patience is not None and config.lr_patience < 0:
        raise ValueError("lr_patience no puede ser negativo.")
    if config.early_stop_patience < 0:
        raise ValueError("early_stop_patience no puede ser negativo.")
    if not math.isfinite(config.beta) or config.beta <= 0:
        raise ValueError("beta debe ser positivo y finito.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo y finito.")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible. Activa la GPU o selecciona cpu.")
    torch.manual_seed(config.seed)
    train_data = DIV2KDataset(
        config.train_dir,
        training=True,
        seed=config.seed,
        crops_per_image=config.crops_per_image,
    )
    val_data = DIV2KDataset(config.val_dir, training=False, seed=config.seed + 1)
    if {p.resolve() for p in train_data.files} & {p.resolve() for p in val_data.files}:
        raise ValueError("Entrenamiento y validación comparten archivos.")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_names = ("last.pt", "best.pt", "best_quality.pt")
    if not config.resume and any((output / name).exists() for name in checkpoint_names):
        raise ValueError("La salida ya tiene checkpoints. Usa resume o una carpeta nueva.")
    hide, reveal = HideNet(config.width).to(device), RevealNet(config.width).to(device)
    optimizer = torch.optim.Adam(
        list(hide.parameters()) + list(reveal.parameters()), lr=config.learning_rate
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=config.lr_patience
        )
        if config.lr_patience is not None
        else None
    )
    start_epoch, best_loss, history = 0, float("inf"), []
    best_quality_psnr, bad_epochs, data_epoch_offset = -float("inf"), 0, 0
    if config.init_from:
        checkpoint = torch.load(config.init_from, map_location=device, weights_only=True)
        if checkpoint["config"]["width"] != config.width:
            raise ValueError("width debe coincidir con los pesos iniciales.")
        hide.load_state_dict(checkpoint["hide"])
        reveal.load_state_dict(checkpoint["reveal"])
        data_epoch_offset = checkpoint.get("data_epoch_offset", 0) + checkpoint["epoch"]
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device, weights_only=True)
        for field in (
            "width",
            "beta",
            "learning_rate",
            "seed",
            "crops_per_image",
            "batch_size",
            "secret_loss",
            "lr_patience",
            "early_stop_patience",
        ):
            default = TrainConfig.__dataclass_fields__[field].default
            if checkpoint["config"].get(field, default) != getattr(config, field):
                raise ValueError(f"{field} debe coincidir con el checkpoint.")
        hide.load_state_dict(checkpoint["hide"])
        reveal.load_state_dict(checkpoint["reveal"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]
        best_quality_psnr = checkpoint.get("best_quality_psnr", -float("inf"))
        bad_epochs = checkpoint.get("bad_epochs", 0)
        data_epoch_offset = checkpoint.get("data_epoch_offset", 0)
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if config.epochs <= start_epoch:
            raise ValueError("epochs debe superar la época guardada en el checkpoint.")
        required = ["best.pt"]
        if math.isfinite(best_quality_psnr):
            required.append("best_quality.pt")
        for name in required:
            best_path = Path(config.resume).parent / name
            if not best_path.is_file():
                raise ValueError(f"Conserva {name} junto al checkpoint para reanudar.")
            if best_path.resolve() != (output / name).resolve():
                shutil.copy2(best_path, output / name)

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    print(
        f"Dispositivo: {device} | entrenamiento: {len(train_data.files)} imágenes, "
        f"{len(train_data)} recortes/época | validación: {len(val_data)} imágenes",
        flush=True,
    )
    for epoch in range(start_epoch, config.epochs):
        train_data.set_epoch(data_epoch_offset + epoch)
        generator = torch.Generator().manual_seed(config.seed + data_epoch_offset + epoch)
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
        train_metrics = _run_epoch(
            hide,
            reveal,
            train_loader,
            device,
            config.beta,
            optimizer,
            secret_loss=config.secret_loss,
        )
        val_metrics = _run_epoch(
            hide, reveal, val_loader, device, config.beta, secret_loss=config.secret_loss
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        improved = val_metrics["loss"] < best_loss
        best_loss = min(best_loss, val_metrics["loss"])
        bad_epochs = 0 if improved else bad_epochs + 1
        quality_improved = (
            val_metrics["reading_rate"] >= 0.9
            and val_metrics["ssim"] >= 0.9
            and val_metrics["psnr"] > best_quality_psnr
        )
        if quality_improved:
            best_quality_psnr = val_metrics["psnr"]
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])
        stop_reason = None
        if val_metrics["collapsed_fraction"] >= 0.95 and val_metrics["reading_rate"] == 0:
            stop_reason = "Colapso: al menos 95% de los QR son blancos o negros saturados."
        elif config.early_stop_patience and bad_epochs >= config.early_stop_patience:
            stop_reason = "Parada temprana: la pérdida de validación dejó de mejorar."
        state = {
            "epoch": epoch + 1,
            "hide": hide.state_dict(),
            "reveal": reveal.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "history": history,
            "config": asdict(config),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_quality_psnr": best_quality_psnr,
            "bad_epochs": bad_epochs,
            "data_epoch_offset": data_epoch_offset,
            "stop_reason": stop_reason,
        }
        if improved:
            _save_checkpoint(state, output / "best.pt")
        if quality_improved:
            _save_checkpoint(state, output / "best_quality.pt")
        _save_checkpoint(state, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Época {epoch + 1}/{config.epochs} | "
            f"train={train_metrics['loss']:.6f} | val={val_metrics['loss']:.6f} | "
            f"QR MSE={val_metrics['secret_mse']:.6f} | PSNR={val_metrics['psnr']:.2f} dB | "
            f"SSIM={val_metrics['ssim']:.3f} | lectura={val_metrics['reading_rate']:.1%} | "
            f"LR={row['learning_rate']:.2g}",
            flush=True,
        )
        if stop_reason:
            print(stop_reason, flush=True)
            break
    return history


def main():
    parser = argparse.ArgumentParser(description="Entrena M2 con DIV2K.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/m2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--crops-per-image", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    parser.add_argument("--secret-loss", choices=("mse", "bce"), default="mse")
    parser.add_argument("--lr-patience", type=int)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    train(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
