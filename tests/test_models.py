import numpy as np
import torch
from PIL import Image

from deepstego.models import HideNet, RevealNet, quantize_8bit


def test_shapes_and_gradient_through_quantization():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    hide, reveal = HideNet(4), RevealNet(4)
    cover = torch.rand(2, 3, 128, 128)
    secret = torch.randint(0, 2, (2, 1, 128, 128)).float()
    container = quantize_8bit(hide(cover, secret))
    recovered = reveal(container)
    assert container.shape == cover.shape
    assert recovered.shape == secret.shape
    assert container.min() >= 0 and container.max() <= 1
    assert recovered.min() >= 0 and recovered.max() <= 1
    (recovered - secret).square().mean().backward()
    for model in (hide, reveal):
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert sum(parameter.grad.abs().sum() for parameter in model.parameters()) > 0


def test_quantization_matches_png_round_trip(tmp_path):
    image = torch.rand(3, 128, 128, requires_grad=True)
    quantized = quantize_8bit(image)
    path = tmp_path / "container.png"
    pixels = image.detach().permute(1, 2, 0).mul(255).round().byte().numpy()
    Image.fromarray(pixels).save(path)
    with Image.open(path) as png:
        reloaded = torch.from_numpy(np.array(png)).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(quantized, reloaded)
    quantized.sum().backward()
    torch.testing.assert_close(image.grad, torch.ones_like(image))


def test_logits_keep_inference_compatible_and_bce_gradient_when_saturated():
    reveal = RevealNet(4)
    container = torch.rand(1, 3, 16, 16)
    torch.testing.assert_close(reveal(container), reveal.forward_logits(container).sigmoid())
    with torch.no_grad():
        for parameter in reveal.parameters():
            parameter.zero_()
        reveal.network[-2].bias.fill_(100)
    logits = reveal.forward_logits(container)
    assert torch.all(reveal(container) == 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert reveal.network[-2].bias.grad.item() == 1
