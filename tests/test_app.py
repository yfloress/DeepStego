from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from skimage import data

from deepstego.app import ModelDemo, build_app


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    demo = ModelDemo()
    yield demo
    demo.files.cleanup()


def test_real_checkpoint_png_round_trip(model):
    photo = Image.fromarray(data.astronaut())
    cover, secret, preview, download, metrics = model.hide_message(photo, "Hola TEL-328")
    assert cover.size == secret.size == (128, 128)
    assert preview == download
    with Image.open(download) as png:
        assert png.format == "PNG"
        assert png.size == (128, 128)
    raw, clean, text = model.reveal_image(download)
    assert raw.shape == (128, 128)
    assert clean.shape == (640, 640)
    assert text == "Hola TEL-328"
    assert "PSNR" in metrics and "SSIM" in metrics
    second = model.hide_message(photo, "Otro mensaje")[3]
    assert second != download
    assert Path(download).is_file()


def test_invalid_inputs(model, tmp_path):
    with pytest.raises(ValueError, match="Carga una fotografía"):
        model.hide_message(None, "Hola")
    photo = Image.new("RGB", (128, 128))
    with pytest.raises(ValueError, match="entre 1 y 20"):
        model.hide_message(photo, "x" * 21)
    with pytest.raises(ValueError, match="Carga la imagen"):
        model.reveal_image(None)
    for filename, size in (("wrong.jpg", (128, 128)), ("resized.png", (256, 256))):
        path = tmp_path / filename
        Image.new("RGB", size).save(path)
        with pytest.raises(ValueError, match="PNG de 128"):
            model.reveal_image(path)
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")
    with pytest.raises(ValueError, match="No se pudo abrir"):
        model.reveal_image(invalid)


def test_blank_png_is_not_readable(model, tmp_path):
    path = tmp_path / "blank.png"
    Image.fromarray(np.full((128, 128, 3), 255, dtype=np.uint8)).save(path)
    assert model.reveal_image(path)[2] is None


def test_gradio_builds_with_both_actions(model):
    pytest.importorskip("gradio")
    app = build_app(model)
    assert {fn.api_name for fn in app.fns.values()} == {"hide", "reveal"}
    app.close()
