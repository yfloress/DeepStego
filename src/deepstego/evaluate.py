"""Evaluación reproducible de M2 y M3 sobre fotografías de prueba."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deepstego.data import DIV2KDataset
from deepstego.metrics import image_quality
from deepstego.models import HideNet, RevealNet, quantize_8bit
from deepstego.qr import read_qr


def evaluate(checkpoint_path, image_dir, output_dir, *, count=1000, seed=44, device="cpu"):
    if count < 1:
        raise ValueError("count debe ser positivo.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Usa una carpeta de salida vacía para conservar resultados anteriores.")
    dataset = DIV2KDataset(image_dir, training=False, seed=seed)
    if len(dataset) < count:
        raise ValueError(f"Se necesitan {count} fotografías; se encontraron {len(dataset)}.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hide = HideNet(checkpoint["config"]["width"]).to(device)
    reveal = RevealNet(checkpoint["config"]["width"]).to(device)
    hide.load_state_dict(checkpoint["hide"])
    reveal.load_state_dict(checkpoint["reveal"])
    hide.eval()
    reveal.eval()
    output.mkdir(parents=True, exist_ok=True)
    indices = np.random.default_rng(seed).choice(len(dataset), count, replace=False)
    rows = []
    examples = {True: 0, False: 0}
    with torch.inference_mode():
        for number, index in enumerate(indices):
            sample = dataset[int(index)]
            cover = sample["cover"].unsqueeze(0).to(device)
            secret = sample["secret"].unsqueeze(0).to(device)
            container = quantize_8bit(hide(cover, secret))
            pixels = container[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            png_path = output / "container.png"
            Image.fromarray(pixels).save(png_path)
            with Image.open(png_path) as png:
                loaded = np.array(png.convert("RGB"), dtype=np.float32) / 255
            recovered = reveal(torch.from_numpy(loaded).permute(2, 0, 1).unsqueeze(0).to(device))
            result = read_qr(recovered)
            correct = result.text == sample["text"]
            quality = image_quality(sample["cover"].permute(1, 2, 0).numpy(), loaded)
            rows.append(
                {
                    "file": str(dataset.files[int(index)]),
                    "text": sample["text"],
                    "decoded": result.text,
                    "correct": correct,
                    **quality,
                }
            )
            if examples[correct] < 3:
                prefix = output / f"{'success' if correct else 'failure'}_{examples[correct]}"
                Image.fromarray(pixels).save(str(prefix) + "_container.png")
                Image.fromarray(
                    sample["cover"].permute(1, 2, 0).mul(255).round().byte().numpy()
                ).save(str(prefix) + "_cover.png")
                Image.fromarray(np.asarray(result.clean)).save(str(prefix) + "_clean.png")
                Image.fromarray(recovered[0, 0].mul(255).round().byte().cpu().numpy()).save(
                    str(prefix) + "_revealed.png"
                )
                examples[correct] += 1
            if (number + 1) % 100 == 0:
                print(f"Evaluadas {number + 1}/{count}", flush=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": count,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "reading_rate": sum(r["correct"] for r in rows) / count,
    }
    for metric in ("psnr", "ssim"):
        values = np.array([r[metric] for r in rows])
        summary[metric + "_mean"] = float(values.mean())
        summary[metric + "_std"] = float(values.std()) if np.isfinite(values).all() else None
    summary["criteria"] = {
        "psnr": summary["psnr_mean"] >= 30,
        "ssim": summary["ssim_mean"] >= 0.90,
        "reading": summary["reading_rate"] >= 0.90,
    }
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for axis, metric in zip(axes, ("psnr", "ssim")):
        values = [r[metric] for r in rows if np.isfinite(r[metric])]
        axis.hist(values, bins=20)
        axis.set_title(metric.upper())
        axis.set_ylabel("Imágenes")
    fig.tight_layout()
    fig.savefig(output / "metrics.png")
    plt.close(fig)
    # PSNR infinito se representa como texto para mantener JSON válido.
    serialized = {
        k: ("Infinity" if isinstance(v, float) and np.isinf(v) else v) for k, v in summary.items()
    }
    (output / "summary.json").write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        evaluate(
            args.checkpoint,
            args.image_dir,
            args.output_dir,
            count=args.count,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
