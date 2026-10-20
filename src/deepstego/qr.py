"""Generación de QR para M1."""

import argparse
import random
import string
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
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


SCALE = 4  # ampliación de 128×128 a 512×512.
MEDIAN_KSIZE = 7  # mediana, medida en la imagen YA ampliada (≈ 2 px originales).
MARGIN = 16  # zona de silencio en píxeles originales (×SCALE al final).
FALLBACK_KSIZES = (0, 11)  # si falla la lectura, se reintenta con estas medianas.

NOT_READABLE = "QR no legible"


@dataclass
class QRReadResult:
    """Resultado de ``read_qr``."""

    text: str | None  # mensaje decodificado, o None si no se pudo leer
    clean: np.ndarray  # QR limpio (uint8, binario, con margen y ampliado)
    threshold: float  # umbral elegido por Otsu (0..255)

    @property
    def ok(self) -> bool:
        return self.text is not None

    @property
    def message(self) -> str:
        """Texto para mostrar en la interfaz: el mensaje o el aviso de no legible."""
        return self.text if self.text is not None else NOT_READABLE


def to_uint8_gray(img) -> np.ndarray:
    """Normaliza la entrada a uint8 (H, W).

    Acepta arreglos NumPy o tensores de PyTorch con forma (H, W), (1, H, W),
    (H, W, 1) o (1, 1, H, W), en float [0, 1] o uint8.
    """
    if hasattr(img, "detach"):  # tensor de PyTorch, sin importar torch aquí
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba una imagen de 1 canal; llegó forma {np.shape(img)}.")

    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() > 1.0 + 1e-6:  # vino en escala 0..255 pero como float
        arr = arr / 255.0
    return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def clean_qr(
    revealed,
    median_ksize: int = MEDIAN_KSIZE,
    margin: int = MARGIN,
    scale: int = SCALE,
    upscale_first: bool = True,
) -> tuple[np.ndarray, float]:
    """Limpia el QR revelado S' y devuelve ``(qr_limpio_uint8, umbral_otsu)``.

    Orden por defecto (``upscale_first=True``, el que mejor lee, ver A6):
        ampliación ×scale (bilineal) → mediana → Otsu → margen blanco

    Orden literal de la propuesta (``upscale_first=False``):
        mediana → Otsu → margen blanco → ampliación ×scale (vecino más cercano)

    En ambos casos ``median_ksize`` se aplica a la resolución en que se filtra
    (ampliada o no), y ``margin`` se expresa en píxeles de la imagen original.
    """
    img = to_uint8_gray(revealed)
    if median_ksize and median_ksize > 1 and median_ksize % 2 == 0:
        raise ValueError("median_ksize debe ser impar.")

    # 1. Ampliación bilineal antes de filtrar: con 37 módulos en 128 px cada
    #    módulo mide ~3,5 px, y una mediana de 3×3 o 5×5 a esa escala ya borra
    #    módulos sueltos. Ampliada, la mediana quita el ruido con más precisión
    #    y Otsu ubica los bordes con resolución subpíxel.
    if upscale_first and scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 2. Filtro de mediana: elimina ruido puntual (sal y pimienta, píxeles
    #    sueltos) sin redondear tanto los bordes como un promedio o un gaussiano.
    if median_ksize and median_ksize > 1:
        img = cv2.medianBlur(img, median_ksize)

    # 3. Umbral global de Otsu: elige automáticamente el corte que mejor separa
    #    los dos grupos del histograma (módulos negros y blancos).
    thr, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Margen blanco (zona de silencio): el detector necesita borde claro
    #    alrededor de los patrones de las esquinas.
    pad = margin * (scale if upscale_first else 1)
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # 5. (Solo orden de la propuesta) ampliación por vecino más cercano.
    if not upscale_first and scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return img, float(thr)


_DETECTOR = None


def decode_qr(img: np.ndarray) -> str | None:
    """Decodifica una imagen de QR (ya limpia) con ``cv2.QRCodeDetector``."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = cv2.QRCodeDetector()
    try:
        text, points, _ = _DETECTOR.detectAndDecode(to_uint8_gray(img))
    except cv2.error:
        return None
    if points is None or not text:
        return None
    return text


def read_qr(
    revealed,
    median_ksize: int = MEDIAN_KSIZE,
    margin: int = MARGIN,
    scale: int = SCALE,
    upscale_first: bool = True,
    fallback: bool = True,
) -> QRReadResult:
    """Pipeline completo de M3: QR revelado S' → texto (o None) + QR limpio.

    Si la lectura falla y ``fallback`` es True, se reintenta con las medianas de
    ``FALLBACK_KSIZES`` (sin mediana sirve cuando S' está borroso pero no ruidoso;
    una mediana más grande, cuando hay mucho ruido). El resultado devuelto es el
    primer intento exitoso o, si todos fallan, el primero.
    """
    ksizes = [median_ksize]
    if fallback:
        ksizes += [k for k in FALLBACK_KSIZES if k != median_ksize]

    first = None
    for k in ksizes:
        clean, thr = clean_qr(
            revealed, median_ksize=k, margin=margin, scale=scale, upscale_first=upscale_first
        )
        res = QRReadResult(text=decode_qr(clean), clean=clean, threshold=thr)
        if res.ok:
            return res
        first = first or res
    return first


def simulate_revealed(
    qr: np.ndarray,
    noise_std: float = 0.0,
    blur_sigma: float = 0.0,
    contrast: float = 1.0,
    salt_pepper: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Degrada un QR limpio para imitar un QR revelado S' (float32 en [0, 1]).

    - ``contrast``: <1 acerca negros y blancos al gris medio (la red no llega a 0 y 1).
    - ``blur_sigma``: desenfoque gaussiano (bordes de módulos poco definidos).
    - ``noise_std``: ruido gaussiano aditivo.
    - ``salt_pepper``: fracción de píxeles invertidos al azar.

    Permite desarrollar y medir M3 antes de tener la red entrenada.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(qr, dtype=np.float32)
    x = 0.5 + (x - 0.5) * contrast
    if blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), blur_sigma)
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, x.shape).astype(np.float32)
    if salt_pepper > 0:
        mask = rng.random(x.shape) < salt_pepper
        x = np.where(mask, 1.0 - x, x)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    main()
