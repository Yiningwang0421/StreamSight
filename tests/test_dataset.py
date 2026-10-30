"""Dataset shape, labels, trajectories, occlusion, and shortcut checks."""

from __future__ import annotations

import numpy as np
import torch

from data.synthetic_video import (
    CLASS_NAMES,
    DIRECTIONS,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_FRAMES,
    generate_split,
)


def test_shapes_labels_and_balance():
    data = generate_split("train", n=32, seed=0)
    assert data["videos"].shape == (32, NUM_FRAMES, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert data["videos"].dtype == torch.float32
    assert data["labels"].dtype == torch.int64
    assert int(data["labels"].min()) >= 0
    assert int(data["labels"].max()) < NUM_CLASSES
    assert data["positions"].shape == (32, NUM_FRAMES, 2)
    assert data["visible"].shape == (32, NUM_FRAMES)
    assert data["videos"].min() >= 0 and data["videos"].max() <= 1
    counts = torch.bincount(data["labels"], minlength=NUM_CLASSES)
    assert int(counts.max() - counts.min()) <= 1


def test_reproducible_and_splits_differ():
    first = generate_split("train", n=16, seed=7)
    second = generate_split("train", n=16, seed=7)
    other = generate_split("test", n=16, seed=7)
    assert torch.equal(first["videos"], second["videos"])
    assert torch.equal(first["labels"], second["labels"])
    assert torch.equal(first["visible"], second["visible"])
    assert not torch.equal(first["videos"], other["videos"])


def test_trajectories_match_labels_and_stay_inside():
    data = generate_split("val", n=64, seed=1, noise_std=0.0, occlusion_prob=0.0)
    delta = data["positions"][:, -1] - data["positions"][:, 0]
    for class_id, (dx, dy) in DIRECTIONS.items():
        mask = data["labels"] == class_id
        assert int(mask.sum()) > 0
        expected_x = dx * data["travel"][mask]
        expected_y = dy * data["travel"][mask]
        assert torch.allclose(delta[mask, 0], expected_x, atol=1e-3)
        assert torch.allclose(delta[mask, 1], expected_y, atol=1e-3)
    obj = data["obj_size"].to(dtype=data["positions"].dtype)
    assert torch.all(data["positions"][..., 0] >= -1e-3)
    assert torch.all(data["positions"][..., 1] >= -1e-3)
    assert torch.all(data["positions"][..., 0] + obj[:, None] <= IMAGE_SIZE + 1e-3)
    assert torch.all(data["positions"][..., 1] + obj[:, None] <= IMAGE_SIZE + 1e-3)
    assert bool(data["visible"].all())


def test_occlusion_hides_interior_frames_and_motion_continues():
    data = generate_split("test", n=24, seed=3, noise_std=0.0, occlusion_prob=1.0)
    assert bool((~data["visible"]).any())
    for i in range(data["labels"].shape[0]):
        vis = data["visible"][i]
        assert bool(vis[0]) and bool(vis[-1])
        assert bool((~vis).any())
        hidden = (~vis).nonzero(as_tuple=False).flatten()
        for t in hidden.tolist():
            assert torch.count_nonzero(data["videos"][i, t]) == 0
        shown = vis.nonzero(as_tuple=False).flatten()
        assert torch.any(data["videos"][i, int(shown[0])] > 0.5)
        # Latent motion does not stop while the object is hidden.
        step = data["positions"][i, 1:] - data["positions"][i, :-1]
        travel_per_frame = float(data["travel"][i]) / (NUM_FRAMES - 1)
        assert torch.allclose(step.abs().sum(dim=-1), torch.full((NUM_FRAMES - 1,), travel_per_frame), atol=1e-3)


def test_start_position_does_not_encode_the_label():
    data = generate_split("train", n=400, seed=11, noise_std=0.0, occlusion_prob=0.0)
    starts = data["positions"][:, 0].numpy()
    labels = data["labels"].numpy()
    means = [starts[labels == c].mean(axis=0) for c in range(NUM_CLASSES)]
    for axis in (0, 1):
        spread = max(mean[axis] for mean in means) - min(mean[axis] for mean in means)
        assert spread < 3.0, f"start axis {axis} differs across classes by {spread}"

    signed_x = np.zeros(len(labels), dtype=np.float64)
    signed_y = np.zeros(len(labels), dtype=np.float64)
    signed_x[labels == 0] = -1.0
    signed_x[labels == 1] = 1.0
    signed_y[labels == 2] = -1.0
    signed_y[labels == 3] = 1.0
    corr_x = np.corrcoef(starts[:, 0], signed_x)[0, 1]
    corr_y = np.corrcoef(starts[:, 1], signed_y)[0, 1]
    assert abs(corr_x) < 0.2
    assert abs(corr_y) < 0.2

    design = np.concatenate([starts, np.ones((len(labels), 1))], axis=1)
    target = np.eye(NUM_CLASSES, dtype=np.float64)[labels]
    coeff, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = (design @ coeff).argmax(axis=1)
    accuracy = float((predicted == labels).mean())
    assert accuracy < 0.45, f"first-frame position probe accuracy {accuracy:.3f} is too high"
    assert set(CLASS_NAMES) == {"left", "right", "up", "down"}
