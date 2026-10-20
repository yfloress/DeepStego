"""Generación de QR para M1."""

import argparse
import random
import string
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import qrcode
from PIL import Image

QR_SIZE = 128
MAX_MESSAGE_LENGTH = 20
RANDOM_ALPHABET = string.ascii_letters + string.digits


def generate_qr(text: str) -> Image.Image:
    """Genera un QR binario de 128×128 con corrección H."""
    if not isinstance(text, str):
        raise TypeError("El mensaje debe ser una cadena de texto.")
    if not 1 <= len(text) <= MAX_MESSAGE_LENGTH:
        raise ValueError(f"El mensaje debe tener entre 1 y {MAX_MESSAGE_LENGTH} caracteres.")

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=1,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white").convert("L")
    # El escalado entero mantiene uniformes los módulos.
    scale = QR_SIZE // image.width
    scaled_size = image.width * scale
    image = image.resize((scaled_size, scaled_size), resample=Image.Resampling.NEAREST)
    canvas = Image.new("L", (QR_SIZE, QR_SIZE), color=255)
    offset = (QR_SIZE - scaled_size) // 2
    canvas.paste(image, (offset, offset))
    return canvas


def generate_qr_array(text: str) -> np.ndarray:
    """Devuelve un QR float32 en [0, 1], con forma (1, 128, 128)."""
    image = generate_qr(text)
    return np.asarray(image, dtype=np.float32)[None, :, :] / 255.0


def iter_qr_samples(
    count: int | None = None, *, seed: int | None = None
) -> Iterator[tuple[str, np.ndarray]]:
    """Genera pares (texto, QR normalizado); count=None produce una secuencia infinita."""
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("La cantidad debe ser un entero o None.")
        if count < 0:
            raise ValueError("La cantidad no puede ser negativa.")

    rng = random.Random(seed)
    index = 0
    while count is None or index < count:
        length = rng.randint(1, MAX_MESSAGE_LENGTH)
        text = "".join(rng.choices(RANDOM_ALPHABET, k=length))
        yield text, generate_qr_array(text)
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un QR de 128×128 con corrección H.")
    parser.add_argument("text", help="Mensaje de 1 a 20 caracteres.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("outputs/qr.png"), help="Ruta del PNG."
    )
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("La salida debe tener extensión .png.")
    try:
        image = generate_qr(args.text)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"QR guardado en {args.output}")


if __name__ == "__main__":
    main()
