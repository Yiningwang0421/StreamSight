"""Temporal-convolution baseline.

Two left-padded Conv1d layers (kernel 3) so the hidden state at time t depends
only on inputs t-4 ... t. Classification reads the last timestep. That receptive
field is five frames, which is the main architectural difference from the
transformer's full-prefix attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from data.synthetic_video import NUM_CLASSES
from models.frame_encoder import FrameEncoder


class TemporalCNN(nn.Module):
    def __init__(self, dim: int = 128, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.dim = dim
        self.encoder = FrameEncoder(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=0)
        self.head = nn.Linear(dim, num_classes)

    def temporal_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, T, C] -> [B, T, C] with a causal receptive field of 5."""
        x = tokens.transpose(1, 2)
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv2(x))
        return x.transpose(1, 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(frames)
        hidden = self.temporal_hidden(tokens)
        return self.head(hidden[:, -1])
