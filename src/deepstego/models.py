"""CNN iniciales de ocultamiento y revelado."""

import torch
from torch import nn


def _network(in_channels: int, out_channels: int, width: int):
    if width < 1:
        raise ValueError("width debe ser positivo.")
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, out_channels, 3, padding=1),
        nn.Sigmoid(),
    )


class HideNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.network = _network(4, 3, width)

    def forward(self, cover, secret):
        return self.network(torch.cat((cover, secret), dim=1))


class RevealNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.network = _network(3, 1, width)

    def forward(self, container):
        return self.network(container)


def quantize_8bit(image: torch.Tensor) -> torch.Tensor:
    """Redondea a 8 bits conservando el gradiente mediante STE."""
    image = image.clamp(0, 1)
    rounded = image.mul(255).round().div(255)
    return image + (rounded - image).detach()
