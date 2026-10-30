"""Compare PyTorch/cuDNN conv, explicit im2col GEMM, and the custom implicit GEMM.

On CPU the PyTorch row is the CPU conv implementation, not cuDNN. The custom
CUDA kernel is recorded as unavailable rather than replaced with the Python
reference. TF32 is disabled so the FP32 kernels are comparable.

    python benchmarks/benchmark_conv.py
    python benchmarks/benchmark_conv.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import blank_if_nan, describe_device, resolve_device, write_csv  # noqa: E402
from benchmarks.plot_results import plot_conv  # noqa: E402
from inference.cuda_backend import implicit_gemm_conv2d, kernels_available, unavailable_reason  # noqa: E402
from inference.timing import cuda_allocation_mb, measure_ms  # noqa: E402
from models.conv_mapping import explicit_im2col_conv2d, gemm_extents  # noqa: E402

SHAPES = (
    (1, 1, 16, 16, 16),
    (4, 1, 32, 32, 16),
    (2, 16, 16, 16, 32),
    (2, 16, 32, 32, 32),
)


def _row(**kwargs):
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_conv.csv")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    warmup = args.warmup if args.warmup is not None else (5 if device.type == "cuda" else 2)
    iters = args.iters if args.iters is not None else (20 if device.type == "cuda" else 5)
    conv_name = "cudnn_conv2d" if device.type == "cuda" else "pytorch_conv2d_cpu"
    print(f"device: {describe_device(device)}")
    print("TF32 disabled. Custom CUDA implicit GEMM is FP32 and bias-free.")
    if device.type != "cuda":
        print("This process has no CUDA device, so the cuDNN and custom-kernel rows are not GPU measurements.")
    rows = []
    for n, cin, height, width, cout in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(n, cin, height, width, device=device)
        weight = torch.randn(cout, cin, 3, 3, device=device)
        reference = F.conv2d(x, weight, padding=1)
        extents = gemm_extents(n, cin, height, width, cout)
        im2col_mb = extents["full_im2col_elements"] * 4 / (1024 ** 2)
        jobs = [
            (conv_name, lambda x=x, weight=weight: F.conv2d(x, weight, padding=1)),
            (
                "explicit_im2col_gemm",
                lambda x=x, weight=weight: explicit_im2col_conv2d(x, weight)[0],
            ),
        ]
        if kernels_available() and device.type == "cuda":
            jobs.append(("cuda_implicit_gemm", lambda x=x, weight=weight: implicit_gemm_conv2d(x, weight)))
        else:
            rows.append(
                _row(
                    device=describe_device(device),
                    backend="cuda_implicit_gemm",
                    batch=n,
                    cin=cin,
                    cout=cout,
                    height=height,
                    width=width,
                    median_ms="",
                    mean_ms="",
                    extra_alloc_mb="",
                    theoretical_im2col_mb=f"{im2col_mb:.4f}",
                    max_abs_err="",
                    status="unavailable",
                    note=unavailable_reason() or "CUDA extension was not executed",
                )
            )
        for name, fn in jobs:
            try:
                got = fn()
                err = float((got - reference).abs().max().item())
                stats = measure_ms(fn, device, warmup=warmup, iters=iters)
                _, extra = cuda_allocation_mb(fn, device)
                rows.append(
                    _row(
                        device=describe_device(device),
                        backend=name,
                        batch=n,
                        cin=cin,
                        cout=cout,
                        height=height,
                        width=width,
                        median_ms=f"{stats['median_ms']:.4f}",
                        mean_ms=f"{stats['mean_ms']:.4f}",
                        extra_alloc_mb=blank_if_nan(extra),
                        theoretical_im2col_mb=f"{im2col_mb:.4f}",
                        max_abs_err=f"{err:.3e}",
                        status="ok",
                        note="theoretical_im2col_mb is the explicit unfold buffer, not a measured allocation",
                    )
                )
                print(f"{name} N={n} Cin={cin} {height}x{width} Cout={cout}  {stats['median_ms']:.3f} ms  err {err:.2e}")
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "error"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rows.append(
                    _row(
                        device=describe_device(device),
                        backend=name,
                        batch=n,
                        cin=cin,
                        cout=cout,
                        height=height,
                        width=width,
                        median_ms="",
                        mean_ms="",
                        extra_alloc_mb="",
                        theoretical_im2col_mb=f"{im2col_mb:.4f}",
                        max_abs_err="",
                        status=status,
                        note=str(exc).splitlines()[0][:200],
                    )
                )
                if status != "oom":
                    raise
    # One unavailable row per shape makes the CSV repeat the same note. That is intentional.
    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    plot_conv(args.out, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
