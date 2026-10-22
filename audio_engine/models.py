from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18 as build_resnet18


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ClassicalModelSpec:
    name: str
    estimator: Any
    feature_type: str = "1d"


class PCGCNN(nn.Module):
    """Compact CNN used on log-Mel spectrograms for binary PCG classification."""

    def __init__(self, in_channels: int = 1, n_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class PCGResNet18(nn.Module):
    """ResNet18 adapted for single-channel mel inputs."""

    def __init__(self, n_classes: int = 2, in_channels: int = 1, *, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()
        # Network access is opt-in: pretrained=True is an explicit training-time request.
        model = build_resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_channels == 1:
            with torch.no_grad():
                model.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TransformerCNNHybrid(nn.Module):
    """Small CNN stem + transformer encoder for 2D acoustic features."""

    def __init__(self, input_dim: int = 64, seq_len: int = 32, n_classes: int = 2, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((input_dim, 1)),
        )
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.squeeze(-1).transpose(1, 2)
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_classical_1d_models() -> dict[str, Any]:
    """Create the mandatory 1D classical models used in the benchmark."""
    return {
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=5, class_weight="balanced"),
        "rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=2.0, gamma="scale", kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ]),
    }


def build_deep_2d_models(*, pretrained_resnet: bool = False, freeze_backbone: bool = False) -> dict[str, nn.Module]:
    """Return the deep CNN-based architectures used for log-mel training."""
    return {
        "cnn": PCGCNN(in_channels=1, n_classes=2),
        "resnet18": PCGResNet18(n_classes=2, in_channels=1, pretrained=pretrained_resnet, freeze_backbone=freeze_backbone),
        "transformer_cnn": TransformerCNNHybrid(input_dim=64, seq_len=32, n_classes=2),
    }


__all__ = [
    "set_seed",
    "ClassicalModelSpec",
    "PCGCNN",
    "PCGResNet18",
    "TransformerCNNHybrid",
    "build_classical_1d_models",
    "build_deep_2d_models",
]
