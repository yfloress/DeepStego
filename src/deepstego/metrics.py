"""Métricas de calidad entre imágenes RGB normalizadas."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_quality(cover, container) -> dict[str, float]:
    cover, container = np.asarray(cover), np.asarray(container)
    if cover.shape != container.shape or cover.ndim != 3 or cover.shape[-1] != 3:
        raise ValueError("Se esperan imágenes RGB de igual forma (H, W, 3).")
    if not all(np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1 for x in (cover, container)):
        raise ValueError("Las imágenes deben tener valores finitos en [0, 1].")
    return {
        "psnr": float(peak_signal_noise_ratio(cover, container, data_range=1)),
        "ssim": float(structural_similarity(cover, container, data_range=1, channel_axis=-1)),
    }
