import json

import numpy as np
import pytest
import torch
from PIL import Image

from deepstego.evaluate import evaluate
from deepstego.models import HideNet, RevealNet


def test_checkpoint_png_evaluation(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "config": {"width": 4},
            "epoch": 2,
            "hide": HideNet(4).state_dict(),
            "reveal": RevealNet(4).state_dict(),
        },
        checkpoint,
    )
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.full((140, 140, 3), 100, dtype=np.uint8)).save(images / "one.png")
    output = tmp_path / "evaluation"
    summary = evaluate(checkpoint, images, output, count=1)
    assert summary["count"] == 1
    assert summary["reading_rate"] == 0
    assert summary["epoch"] == 2
    assert set(summary["criteria"]) == {"psnr", "ssim", "reading"}
    assert json.loads((output / "summary.json").read_text())["count"] == 1
    assert (output / "results.csv").is_file()
    assert (output / "metrics.png").is_file()
    with Image.open(output / "container.png") as image:
        assert image.format == "PNG"
        assert image.size == (128, 128)
    with pytest.raises(ValueError, match="salida vacía"):
        evaluate(checkpoint, images, output, count=1)
    with pytest.raises(ValueError, match="fotografías"):
        evaluate(checkpoint, images, tmp_path / "other", count=2)
