"""Put the project root on sys.path and keep FP32 math in full precision."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.set_float32_matmul_precision("highest")
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = False
