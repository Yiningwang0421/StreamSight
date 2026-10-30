"""Latency helpers.

CUDA timings use CUDA events around the callable and exclude warmup iterations.
They still include Python launch overhead; Nsight Compute is what isolates
kernel duration. CPU timings use ``perf_counter`` and are labeled as CPU time,
not GPU kernel time.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_ms(fn, device: torch.device, warmup: int = 5, iters: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(float(start.elapsed_time(end)))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            synchronize(device)
            samples.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "samples_ms": [float(v) for v in arr.tolist()],
    }


def cuda_allocation_mb(fn, device: torch.device) -> tuple[float, float]:
    """Return (peak allocated MiB, additional MiB above the pre-call baseline).

    On CPU both values are empty strings, written by the caller as blanks.
    Peak is ``max_memory_allocated`` after ``reset_peak_memory_stats``, so it
    includes tensors that were already live.
    """
    if device.type != "cuda":
        return float("nan"), float("nan")
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return peak / (1024 ** 2), max(peak - baseline, 0) / (1024 ** 2)
