"""CSV writing and device descriptions shared by the benchmark scripts."""

from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    processor = platform.processor() or platform.machine()
    return f"cpu ({processor})"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write for {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def blank_if_nan(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return value


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
