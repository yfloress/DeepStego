import random
import subprocess
import sys
from itertools import islice

import cv2
import numpy as np
import pytest
from PIL import Image

from deepstego.qr import generate_qr, generate_qr_array, iter_qr_samples


def decode_qr(pixels):
    # ArUco detecta patrones que el detector clásico puede omitir.
    for detector in (cv2.QRCodeDetector(), cv2.QRCodeDetectorAruco()):
        decoded, _, _ = detector.detectAndDecode(pixels)
        if decoded:
            return decoded
    return ""


@pytest.mark.parametrize(
    "text",
    [
        "A",
        "Hola DeepStego",
        "a" * 20,
        "12345678901234567890",
        "¡Hola, niño!",
        "  mensaje  ",
        "á" * 20,
        "😀" * 20,
        "heEJQjY6fpIzKMWx",
    ],
)
def test_png_round_trip(text, tmp_path):
    image = generate_qr(text)
    assert image.size == (128, 128)
    assert image.mode == "L"
    pixels = np.asarray(image)
    assert set(np.unique(pixels)) == {0, 255}
    assert np.all(pixels[0, :] == 255)
    assert np.all(pixels[-1, :] == 255)
    assert np.all(pixels[:, 0] == 255)
    assert np.all(pixels[:, -1] == 255)

    path = tmp_path / "qr.png"
    image.save(path)
    reloaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert decode_qr(reloaded) == text


def test_training_array():
    array = generate_qr_array("Entrenamiento")
    assert array.shape == (1, 128, 128)
    assert array.dtype == np.float32
    assert set(np.unique(array)) == {0.0, 1.0}
    np.testing.assert_array_equal(array[0] * 255, np.asarray(generate_qr("Entrenamiento")))


@pytest.mark.parametrize("text", ["", "a" * 21])
def test_invalid_message_length(text):
    with pytest.raises(ValueError, match="entre 1 y 20"):
        generate_qr(text)


@pytest.mark.parametrize("text", [None, 123, b"hola"])
def test_invalid_message_type(text):
    with pytest.raises(TypeError, match="cadena"):
        generate_qr(text)


def test_seed_reproduces_samples_without_changing_global_random():
    state = random.getstate()
    first = list(iter_qr_samples(10, seed=42))
    second = list(iter_qr_samples(10, seed=42))
    assert random.getstate() == state
    assert len(first) == 10
    for (text, array), (other_text, other_array) in zip(first, second):
        assert text == other_text
        assert 1 <= len(text) <= 20
        np.testing.assert_array_equal(array, other_array)
    assert [text for text, _ in first] != [text for text, _ in iter_qr_samples(10, seed=43)]


def test_random_samples_are_readable():
    for text, array in iter_qr_samples(100, seed=42):
        pixels = (array[0] * 255).astype(np.uint8)
        assert decode_qr(pixels) == text


def test_empty_and_infinite_sample_streams():
    assert list(iter_qr_samples(0, seed=42)) == []
    assert len(list(islice(iter_qr_samples(seed=42), 3))) == 3


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_sample_count(count):
    with pytest.raises((TypeError, ValueError)):
        list(iter_qr_samples(count))


def test_cli_creates_png(tmp_path):
    path = tmp_path / "nested" / "qr.png"
    subprocess.run(
        [sys.executable, "-m", "deepstego.qr", "Hola", "--output", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (128, 128)


def test_cli_rejects_lossy_output(tmp_path):
    path = tmp_path / "qr.jpg"
    result = subprocess.run(
        [sys.executable, "-m", "deepstego.qr", "Hola", "--output", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not path.exists()
