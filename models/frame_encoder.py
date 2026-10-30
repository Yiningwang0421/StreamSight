"""Per-frame CNN encoder.

Each frame is encoded independently to a 128-d token. Adaptive average pooling
stops at a 4x4 grid so the token still carries coarse position; a global
average would erase the spatial cue the classifier needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(32 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 1, H, W] -> [B, T, out_dim]."""
        if frames.dim() != 5:
            raise ValueError(f"expected [B, T, 1, H, W], got {tuple(frames.shape)}")
        if frames.shape[2] != 1 or frames.shape[-2:] != (32, 32):
            raise ValueError(f"expected grayscale 32x32 frames, got {tuple(frames.shape)}")
        batch, time, channels, height, width = frames.shape
        x = frames.reshape(batch * time, channels, height, width)
        x = self.features(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x.view(batch, time, self.out_dim)
