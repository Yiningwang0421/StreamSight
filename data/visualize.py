"""Save a grid of synthetic motion clips.

Run from the repository root:

    python data/visualize.py --out results/sample_sequences.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic_video import CLASS_NAMES, generate_split  # noqa: E402


def save_samples(path: Path, n_per_class: int = 1, seed: int = 0) -> None:
    import matplotlib.pyplot as plt

    n = n_per_class * len(CLASS_NAMES)
    data = generate_split("train", n=max(n * 4, 32), seed=seed, occlusion_prob=0.8)
    chosen = []
    for class_id in range(len(CLASS_NAMES)):
        hits = (data["labels"] == class_id).nonzero(as_tuple=False).flatten().tolist()
        if len(hits) < n_per_class:
            raise RuntimeError(f"not enough samples for class {CLASS_NAMES[class_id]}")
        chosen.extend(hits[:n_per_class])

    frames = data["videos"].shape[1]
    fig, axes = plt.subplots(
        len(chosen),
        frames,
        figsize=(frames * 0.55, len(chosen) * 0.7),
        squeeze=False,
    )
    for row, index in enumerate(chosen):
        label = int(data["labels"][index])
        for t in range(frames):
            ax = axes[row, t]
            ax.imshow(data["videos"][index, t, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if not bool(data["visible"][index, t]):
                for spine in ax.spines.values():
                    spine.set_color("crimson")
                    spine.set_linewidth(1.5)
            if t == 0:
                ax.set_ylabel(CLASS_NAMES[label], fontsize=8)
            if row == 0:
                ax.set_title(str(t), fontsize=7)
    fig.suptitle("Synthetic motion clips (crimson border = occluded frame)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "sample_sequences.png")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    save_samples(args.out, n_per_class=args.per_class, seed=args.seed)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
