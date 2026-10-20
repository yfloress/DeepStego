import numpy as np
import pytest

from deepstego.metrics import image_quality


def test_known_quality():
    cover = np.zeros((128, 128, 3), dtype=np.float32)
    assert image_quality(cover, cover) == {"psnr": float("inf"), "ssim": 1.0}
    result = image_quality(cover, cover + 0.1)
    assert result["psnr"] == pytest.approx(20)
    assert 0 <= result["ssim"] < 1


def test_invalid_shape_and_range():
    cover = np.zeros((128, 128, 3))
    with pytest.raises(ValueError):
        image_quality(cover, cover[0])
    with pytest.raises(ValueError):
        image_quality(cover, cover + 2)
