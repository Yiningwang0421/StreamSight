"""Plot benchmark CSVs. Rows with a non-ok status are omitted, not drawn as zeros."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ok(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok" and row.get("median_ms", row.get("median_total_ms", ""))]


def plot_conv(csv_path: Path, png_path: Path) -> None:
    if not csv_path.exists():
        print(f"no conv csv at {csv_path}")
        return
    rows = _ok(_read(csv_path))
    if not rows:
        print("conv plot skipped: no successful measurements")
        return
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        key = f"N{row['batch']} C{row['cin']}->{row['cout']} {row['height']}x{row['width']}"
        grouped[row["backend"]].append((key, float(row["median_ms"])))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    backends = list(grouped)
    shapes = [item[0] for item in grouped[backends[0]]]
    width = 0.8 / max(len(backends), 1)
    import numpy as np

    xs = np.arange(len(shapes))
    for index, backend in enumerate(backends):
        values = [value for _, value in grouped[backend]]
        ax.bar(xs + index * width, values, width=width, label=backend)
    ax.set_xticks(xs + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(shapes, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("median latency (ms)")
    ax.set_title(f"Convolution ({rows[0]['device']})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")


def plot_attention(csv_path: Path, png_path: Path) -> None:
    if not csv_path.exists():
        print(f"no attention csv at {csv_path}")
        return
    rows = [row for row in _read(csv_path) if row.get("status") == "ok" and row.get("workload") == "full_sequence"]
    if not rows:
        print("attention plot skipped: no successful full-sequence measurements")
        return
    import matplotlib.pyplot as plt
    import numpy as np

    # One panel per batch size present in the file. Batch 1 is the usual view.
    batches = sorted({int(row["batch"]) for row in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4), squeeze=False)
    for ax, batch in zip(axes[0], batches):
        subset = [row for row in rows if int(row["batch"]) == batch]
        backends = []
        for row in subset:
            if row["backend"] not in backends:
                backends.append(row["backend"])
        seqs = sorted({int(row["seq_kv"]) for row in subset})
        for backend in backends:
            ys = []
            for seq in seqs:
                match = [row for row in subset if row["backend"] == backend and int(row["seq_kv"]) == seq]
                ys.append(float(match[0]["median_ms"]) if match else np.nan)
            ax.plot(seqs, ys, marker="o", label=backend)
        ax.set_xlabel("sequence length")
        ax.set_ylabel("median latency (ms)")
        ax.set_title(f"Full sequence, batch {batch}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(rows[0]["device"], fontsize=10)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")


def plot_streaming(csv_path: Path, png_path: Path) -> None:
    if not csv_path.exists():
        print(f"no streaming csv at {csv_path}")
        return
    rows = [row for row in _read(csv_path) if row.get("status") == "ok" and row.get("median_total_ms")]
    if not rows:
        print("streaming plot skipped: no successful measurements")
        return
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["mode"] for row in rows]
    values = [float(row["median_total_ms"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median sequence latency (ms)")
    ax.set_title(f"Streaming vs prefix recompute ({rows[0]['device']})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    plot_conv(results / "benchmark_conv.csv", results / "benchmark_conv.png")
    plot_attention(results / "benchmark_attention.csv", results / "benchmark_attention.png")
    plot_streaming(results / "benchmark_streaming.csv", results / "benchmark_streaming.png")


if __name__ == "__main__":
    main()
