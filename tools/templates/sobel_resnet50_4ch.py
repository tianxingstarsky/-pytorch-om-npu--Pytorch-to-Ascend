"""
Template example for pth2om_gui.py

Drop this file into the `templates/` directory next to pth2om_gui.py to add a selectable template.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

TEMPLATE_NAME = "SobelResNet50(4ch)"


class _SobelResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                backbone.conv1.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build(num_classes: int, pretrained: bool = False) -> nn.Module:
    return _SobelResNet50(num_classes=num_classes, pretrained=pretrained)

