"""A6 · Banco de pruebas de M3 sin red entrenada.

Genera QR con textos aleatorios (semilla fija), los degrada con
``simulate_revealed`` para imitar la salida de RevealNet y mide la tasa de
lectura de ``read_qr`` para distintos niveles de daño y tamaños de mediana.

Uso:
    python scripts/bench_qr.py                 # 200 QR por condición
    python scripts/bench_qr.py --n 1000        # como el test fijo de M4
    python scripts/bench_qr.py --save-examples outputs/a6

Los resultados (tabla en consola y CSV opcional) sirven para la bitácora y
para elegir los parámetros por defecto de M3.
"""

from __future__ import annotations

import argparse
import csv
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np

from deepstego.qr import MAX_CHARS, make_qr, read_qr, simulate_revealed

ALPHABET = string.ascii_letters + string.digits + " -_.,!?"

# Condiciones de daño: nombre → parámetros de simulate_revealed.
CONDITIONS = {
    "limpio":            {},
    "grises (c=0.6)":    {"contrast": 0.6},
    "ruido σ=0.10":      {"contrast": 0.6, "noise_std": 0.10},
    "ruido σ=0.20":      {"contrast": 0.6, "noise_std": 0.20},
    "ruido σ=0.30":      {"contrast": 0.6, "noise_std": 0.30},
    "blur σ=0.8":        {"contrast": 0.6, "blur_sigma": 0.8},
    "blur σ=1.2":        {"contrast": 0.6, "blur_sigma": 1.2},
    "sal/pim 2 %":       {"salt_pepper": 0.02},
    "sal/pim 5 %":       {"salt_pepper": 0.05},
    "mixto":             {"contrast": 0.6, "noise_std": 0.15, "blur_sigma": 0.8, "salt_pepper": 0.02},
}


# Configuraciones de M3 a comparar: nombre corto → parámetros de read_qr.
CONFIGS = {
    "propuesta": {"median_ksize": 3, "upscale_first": False, "fallback": False},
    "A6": {"upscale_first": True, "fallback": False},
    "A6+fallback": {"upscale_first": True, "fallback": True},
}


def random_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices(ALPHABET, k=rng.randint(1, MAX_CHARS))) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=200, help="QR por condición")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, help="guardar resultados en un CSV")
    ap.add_argument("--save-examples", type=Path, help="carpeta para guardar ejemplos (S', limpio)")
    args = ap.parse_args()

    texts = random_texts(args.n, args.seed)
    qrs = [make_qr(t) for t in texts]
    rows = []

    header = f"{'condición':<18}" + "".join(f"{name:>14}" for name in CONFIGS)
    print(header)
    print("-" * len(header))
    t0 = time.time()
    for name, kw in CONDITIONS.items():
        revealed = [simulate_revealed(q, seed=args.seed + i, **kw) for i, q in enumerate(qrs)]
        line = f"{name:<18}"
        for cname, ckw in CONFIGS.items():
            ok = sum(read_qr(s, **ckw).text == t for s, t in zip(revealed, texts))
            rate = ok / len(texts)
            rows.append({"condicion": name, "config": cname, "leidos": ok,
                         "total": len(texts), "tasa": rate})
            line += f"{rate:>14.1%}"
        print(line, flush=True)

        if args.save_examples:
            out = args.save_examples
            out.mkdir(parents=True, exist_ok=True)
            res = read_qr(revealed[0])
            slug = "".join(c if c.isascii() and c.isalnum() else "_" for c in name)
            cv2.imwrite(str(out / f"{slug}_revelado.png"), np.round(revealed[0] * 255).astype(np.uint8))
            cv2.imwrite(str(out / f"{slug}_limpio.png"), res.clean)

    print(f"\n{len(CONDITIONS) * len(CONFIGS) * args.n} lecturas en {time.time() - t0:.0f} s")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
