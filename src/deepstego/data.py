"""Pares de fotografías DIV2K y QR."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from deepstego.qr import MAX_MESSAGE_LENGTH, QR_SIZE, RANDOM_ALPHABET, generate_qr_array


def find_div2k_split(root: str | Path, folder: str) -> Path:
    """Localiza una partición por su carpeta de imágenes."""
    candidates = []
    for path in Path(root).rglob("*"):
        if not path.is_dir() or path.name.casefold() != folder.casefold():
            continue
        if any(
            child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for child in path.iterdir()
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No se encontró {folder} con imágenes dentro de {root}.")
    if len(candidates) > 1:
        raise ValueError(f"Hay varias carpetas {folder}; indica una ruta explícita: {candidates}")
    return candidates[0]


class DIV2KDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        training: bool,
        seed: int = 42,
        crops_per_image: int = 1,
    ):
        self.files = sorted(
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.files:
            raise ValueError(f"No se encontraron imágenes en {root}.")
        if crops_per_image < 1:
            raise ValueError("crops_per_image debe ser positivo.")
        if not training and crops_per_image != 1:
            raise ValueError("Validación utiliza un recorte fijo por imagen.")
        self.training = training
        self.seed = seed
        self.crops_per_image = crops_per_image
        self.epoch = 0

    def __len__(self):
        return len(self.files) * self.crops_per_image

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        epoch = self.epoch if self.training else 0
        rng = random.Random(f"{self.seed}:{epoch}:{index}")
        with Image.open(self.files[index // self.crops_per_image]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if min(width, height) < QR_SIZE:
            raise ValueError(
                f"La imagen {self.files[index // self.crops_per_image]} es menor a 128."
            )
        if self.training:
            left = rng.randint(0, width - QR_SIZE)
            top = rng.randint(0, height - QR_SIZE)
        else:
            left = (width - QR_SIZE) // 2
            top = (height - QR_SIZE) // 2
        image = image.crop((left, top, left + QR_SIZE, top + QR_SIZE))
        if self.training and rng.random() < 0.5:
            image = ImageOps.mirror(image)
        text = "".join(rng.choices(RANDOM_ALPHABET, k=rng.randint(1, MAX_MESSAGE_LENGTH)))
        cover = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        return {
            "cover": torch.from_numpy(cover),
            "secret": torch.from_numpy(generate_qr_array(text)),
            "text": text,
        }
