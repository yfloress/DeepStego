"""M1+M3. Generar QR (nivel H) y leerlo (×4 → mediana → Otsu → margen → cv2).

Convención de imágenes en todo el módulo:
    - Escala de grises, 1 canal, forma (H, W).
    - float en [0, 1] (como salen de las redes) o uint8 en [0, 255].
    - Blanco = 1 (o 255), negro = 0. Los módulos oscuros del QR valen 0.

Uso típico desde el receptor (M5) o la evaluación (M4):

    >>> from deepstego.qr import make_qr, read_qr
    >>> qr = make_qr("Hola TEL-328")         # (128, 128) float32
    >>> res = read_qr(qr_revelado)            # S' de RevealNet
    >>> res.text                              # "Hola TEL-328" o None
    >>> res.clean                             # QR limpio (uint8) para mostrar
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H

IMG_SIZE = 128
MAX_CHARS = 20

# Parámetros por defecto de la lectura (M3). Ajustables en A11 según resultados.
# Ver docs/A6_lectura_qr.md para el porqué de cada valor.
SCALE = 4             # ampliación de 128×128 a 512×512.
MEDIAN_KSIZE = 7      # mediana, medida en la imagen YA ampliada (≈ 2 px originales).
MARGIN = 16           # zona de silencio en píxeles originales (×SCALE al final).
FALLBACK_KSIZES = (0, 11)  # si falla la lectura, se reintenta con estas medianas.

NOT_READABLE = "QR no legible"


# --------------------------------------------------------------------------- #
# M1 · Generación
# --------------------------------------------------------------------------- #
def make_qr(text: str, size: int = IMG_SIZE, border: int = 4) -> np.ndarray:
    """Convierte un texto en un QR nivel H de ``size``×``size`` (float32 en {0, 1}).

    El QR se genera a 1 píxel por módulo y luego se escala con vecino más
    cercano para mantener bordes nítidos (sin grises intermedios).
    """
    if not text:
        raise ValueError("El mensaje no puede estar vacío.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"El mensaje tiene {len(text)} caracteres; el máximo es {MAX_CHARS}.")

    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, box_size=1, border=border)
    qr.add_data(text)
    qr.make(fit=True)
    modules = np.array(qr.get_matrix(), dtype=bool)  # True = módulo oscuro

    img = np.where(modules, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return (img / 255.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# M3 · Lectura
# --------------------------------------------------------------------------- #
@dataclass
class QRReadResult:
    """Resultado de ``read_qr``."""

    text: str | None      # mensaje decodificado, o None si no se pudo leer
    clean: np.ndarray     # QR limpio (uint8, binario, con margen y ampliado)
    threshold: float      # umbral elegido por Otsu (0..255)

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
        clean, thr = clean_qr(revealed, median_ksize=k, margin=margin,
                              scale=scale, upscale_first=upscale_first)
        res = QRReadResult(text=decode_qr(clean), clean=clean, threshold=thr)
        if res.ok:
            return res
        first = first or res
    return first


# --------------------------------------------------------------------------- #
# Solo para pruebas: simular la salida imperfecta de RevealNet
# --------------------------------------------------------------------------- #
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
