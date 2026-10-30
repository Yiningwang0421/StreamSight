"""Synthetic 32x32 grayscale videos of an object moving in one direction.

The start position is drawn from a region in which every class is legal, and
it is sampled without looking at the label. Direction is therefore not a
function of the initial coordinates. Some clips hide the object for a few
interior frames; the latent trajectory keeps moving, so the label depends on
temporal history rather than a single frame.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ("left", "right", "up", "down")
NUM_CLASSES = 4
IMAGE_SIZE = 32
NUM_FRAMES = 16
MIN_OBJECT = 4
MAX_OBJECT = 5
MIN_TRAVEL = 6
MAX_TRAVEL = 8

# (dx, dy). Image x is the column, y is the row. Up decreases y.
DIRECTIONS = {
    0: (-1.0, 0.0),
    1: (1.0, 0.0),
    2: (0.0, -1.0),
    3: (0.0, 1.0),
}

_SPLIT_SALT = {"train": 1_000_003, "val": 2_000_003, "test": 3_000_003}


def _circle_mask(size: int) -> np.ndarray:
    radius = (size - 1) / 2.0
    coords = np.arange(size, dtype=np.float32) - radius
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (yy * yy + xx * xx) <= (radius + 0.15) ** 2


_CIRCLE_MASKS = {size: _circle_mask(size) for size in range(MIN_OBJECT, MAX_OBJECT + 1)}


def _occlusion_window(rng: np.random.Generator, frames: int) -> Optional[tuple[int, int]]:
    """Return [start, end) of an interior occlusion, or None if the clip is too short."""
    if frames < 6:
        return None
    length = int(rng.integers(2, 4))
    length = min(length, frames - 3)
    start_hi = frames - length - 1
    if start_hi <= 1:
        return None
    start = int(rng.integers(1, start_hi))
    return start, start + length


def generate_split(
    split: str,
    n: int,
    seed: int = 0,
    frames: int = NUM_FRAMES,
    image_size: int = IMAGE_SIZE,
    noise_std: float = 0.05,
    occlusion_prob: float = 0.45,
) -> Dict[str, torch.Tensor]:
    """Generate a deterministic split.

    Returns tensors:
      videos:    float32 [N, T, 1, H, W] in [0, 1]
      labels:    int64   [N]
      positions: float32 [N, T, 2] latent top-left (x, y) before rasterization
      visible:   bool    [N, T]
      travel:    float32 [N]
      obj_size:  int64   [N]
    """
    if split not in _SPLIT_SALT:
        raise ValueError(f"split must be one of {sorted(_SPLIT_SALT)}, got {split!r}")
    if n < 0:
        raise ValueError("n must be non-negative")
    if image_size != IMAGE_SIZE:
        raise ValueError(
            f"this generator is calibrated for {IMAGE_SIZE}x{IMAGE_SIZE} frames, got {image_size}"
        )
    start_lo = MAX_TRAVEL
    start_hi = image_size - MAX_OBJECT - MAX_TRAVEL
    if start_hi <= start_lo:
        raise RuntimeError("image size is too small for the configured travel and object size")

    rng = np.random.default_rng(int(seed) + _SPLIT_SALT[split])
    videos = np.zeros((n, frames, 1, image_size, image_size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    positions = np.zeros((n, frames, 2), dtype=np.float32)
    visible = np.ones((n, frames), dtype=bool)
    travel = np.zeros(n, dtype=np.float32)
    obj_size = np.zeros(n, dtype=np.int64)

    # Balanced labels, shuffled independently of the geometry stream below.
    label_ids = np.arange(n, dtype=np.int64) % NUM_CLASSES
    rng.shuffle(label_ids)

    time = np.arange(frames, dtype=np.float32)
    denom = float(max(frames - 1, 1))

    for i in range(n):
        # Geometry does not read the label. The shared start window is valid
        # for every direction at the maximum object size and travel.
        obj = int(rng.integers(MIN_OBJECT, MAX_OBJECT + 1))
        dist = int(rng.integers(MIN_TRAVEL, MAX_TRAVEL + 1))
        x0 = int(rng.integers(start_lo, start_hi))
        y0 = int(rng.integers(start_lo, start_hi))
        brightness = float(rng.uniform(0.75, 1.0))
        shape = "circle" if rng.random() < 0.5 else "square"
        label = int(label_ids[i])
        dx, dy = DIRECTIONS[label]

        xs = x0 + dx * dist * time / denom
        ys = y0 + dy * dist * time / denom
        positions[i, :, 0] = xs
        positions[i, :, 1] = ys
        labels[i] = label
        travel[i] = dist
        obj_size[i] = obj

        frame_visible = np.ones(frames, dtype=bool)
        if occlusion_prob > 0.0 and float(rng.random()) < occlusion_prob:
            window = _occlusion_window(rng, frames)
            if window is not None:
                frame_visible[window[0] : window[1]] = False
        visible[i] = frame_visible

        patch_mask = _CIRCLE_MASKS[obj] if shape == "circle" else None
        for t in range(frames):
            frame = videos[i, t, 0]
            if frame_visible[t]:
                xi = int(np.rint(xs[t]))
                yi = int(np.rint(ys[t]))
                if xi < 0 or yi < 0 or xi + obj > image_size or yi + obj > image_size:
                    raise RuntimeError(
                        f"trajectory left the frame at sample {i} frame {t}: "
                        f"origin=({xi}, {yi}) obj={obj}"
                    )
                if patch_mask is None:
                    frame[yi : yi + obj, xi : xi + obj] = brightness
                else:
                    region = frame[yi : yi + obj, xi : xi + obj]
                    region[patch_mask] = brightness
            if noise_std > 0.0:
                noise = rng.normal(0.0, noise_std, size=frame.shape).astype(np.float32)
                np.clip(noise, 0.0, 0.2, out=noise)
                frame += noise
            np.clip(frame, 0.0, 1.0, out=frame)

    return {
        "videos": torch.from_numpy(videos),
        "labels": torch.from_numpy(labels),
        "positions": torch.from_numpy(positions),
        "visible": torch.from_numpy(visible),
        "travel": torch.from_numpy(travel),
        "obj_size": torch.from_numpy(obj_size),
    }


class SyntheticMotionDataset(Dataset):
    """In-memory split. ``videos`` has shape [N, T, 1, H, W]."""

    def __init__(
        self,
        split: str,
        n: int,
        seed: int = 0,
        frames: int = NUM_FRAMES,
        noise_std: float = 0.05,
        occlusion_prob: float = 0.45,
    ) -> None:
        self.split = split
        self.seed = seed
        packed = generate_split(
            split,
            n,
            seed=seed,
            frames=frames,
            noise_std=noise_std,
            occlusion_prob=occlusion_prob,
        )
        self.videos = packed["videos"]
        self.labels = packed["labels"]
        self.positions = packed["positions"]
        self.visible = packed["visible"]
        self.travel = packed["travel"]
        self.obj_size = packed["obj_size"]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.videos[index], self.labels[index]


def make_dataloaders(
    train_n: int,
    val_n: int,
    test_n: int,
    batch_size: int,
    seed: int = 0,
    frames: int = NUM_FRAMES,
    noise_std: float = 0.05,
    occlusion_prob: float = 0.45,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split, count in ("train", train_n), ("val", val_n), ("test", test_n):
        dataset = SyntheticMotionDataset(
            split,
            count,
            seed=seed,
            frames=frames,
            noise_std=noise_std,
            occlusion_prob=occlusion_prob,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders
