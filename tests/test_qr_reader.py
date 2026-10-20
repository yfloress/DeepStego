"""Generar un QR y leerlo de vuelta sin modelo de por medio."""

import numpy as np
import pytest

from deepstego.qr import (
    NOT_READABLE,
    clean_qr,
    generate_qr_array,
    read_qr,
    simulate_revealed,
    to_uint8_gray,
)
from deepstego.qr import (
    QR_SIZE as IMG_SIZE,
)


def make_qr(text):
    return generate_qr_array(text)[0]


TEXTS = ["Hola TEL-328", "a", "12345678901234567890", "DeepStego 2026!", "Ñandú y café"]


@pytest.mark.parametrize("text", TEXTS)
def test_ida_y_vuelta_sin_ruido(text):
    res = read_qr(make_qr(text))
    assert res.ok
    assert res.text == text


@pytest.mark.parametrize("text", TEXTS)
def test_lee_qr_revelado_simulado(text):
    """Grises, desenfoque leve y ruido: lo que se espera de RevealNet."""
    noisy = simulate_revealed(
        make_qr(text), noise_std=0.15, blur_sigma=0.6, contrast=0.6, salt_pepper=0.01, seed=0
    )
    assert read_qr(noisy).text == text


def test_tasa_de_lectura_con_ruido_supera_90():
    """Meta de RF4 (≥ 90 %) sobre QR con ruido gaussiano fuerte simulado."""
    texts = [f"msg-{i:03d}-TEL328" for i in range(40)]
    ok = sum(
        read_qr(simulate_revealed(make_qr(t), noise_std=0.20, contrast=0.6, seed=i)).text == t
        for i, t in enumerate(texts)
    )
    assert ok / len(texts) >= 0.90


def test_orden_de_la_propuesta_tambien_lee():
    qr = simulate_revealed(make_qr("Hola"), noise_std=0.1, contrast=0.6, seed=0)
    assert read_qr(qr, median_ksize=3, upscale_first=False).text == "Hola"


def test_ruido_puro_no_es_legible():
    rng = np.random.default_rng(0)
    res = read_qr(rng.random((IMG_SIZE, IMG_SIZE)).astype(np.float32))
    assert not res.ok
    assert res.text is None
    assert res.message == NOT_READABLE


def test_qr_limpio_es_binario_con_margen_y_ampliado():
    clean, thr = clean_qr(make_qr("Hola"), margin=16, scale=4)
    assert clean.dtype == np.uint8
    assert clean.shape == ((IMG_SIZE + 32) * 4, (IMG_SIZE + 32) * 4)
    assert set(np.unique(clean)) <= {0, 255}
    assert (clean[:64] == 255).all()  # margen superior blanco
    assert 0 <= thr <= 255


@pytest.mark.parametrize("shape", [(128, 128), (1, 128, 128), (128, 128, 1), (1, 1, 128, 128)])
def test_acepta_formas_de_tensor(shape):
    qr = make_qr("Hola").reshape(shape)
    assert read_qr(qr).text == "Hola"


def test_acepta_uint8_y_float_0_255():
    qr = make_qr("Hola")
    assert read_qr((qr * 255).astype(np.uint8)).text == "Hola"
    assert read_qr(qr * 255.0).text == "Hola"


def test_rechaza_imagen_de_3_canales():
    with pytest.raises(ValueError):
        to_uint8_gray(np.zeros((128, 128, 3), dtype=np.float32))


def test_acepta_tensor_torch():
    torch = pytest.importorskip("torch")
    t = torch.from_numpy(make_qr("Hola")).unsqueeze(0)  # (1, 128, 128)
    assert read_qr(t).text == "Hola"
