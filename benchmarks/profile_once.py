"""Launch each custom kernel once so Nsight Compute has a stable process to attach to.

This does not time the kernels and does not fall back to PyTorch. If the
extension is missing it exits with status 1 and an explanation.

    ncu --set full -o results/ncu_decode python benchmarks/profile_once.py --kernel decode
    ncu --set full -o results/ncu_conv python benchmarks/profile_once.py --kernel conv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.cuda_backend import (  # noqa: E402
    decode_attention,
    implicit_gemm_conv2d,
    kernels_available,
    unavailable_reason,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("decode", "conv", "both"), default="both")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    args = parser.parse_args()
    if not kernels_available():
        print(unavailable_reason(), file=sys.stderr)
        return 1
    if args.kernel in ("decode", "both"):
        heads, dim = 4, 32
        q = torch.randn(args.batch, heads, dim, device="cuda")
        k = torch.randn(args.batch, heads, args.context, dim, device="cuda")
        v = torch.randn(args.batch, heads, args.context, dim, device="cuda")
        out = decode_attention(q, k, v, args.context)
        torch.cuda.synchronize()
        print(f"decode output {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
    if args.kernel in ("conv", "both"):
        x = torch.randn(args.batch, 16, 32, 32, device="cuda")
        weight = torch.randn(32, 16, 3, 3, device="cuda")
        y = implicit_gemm_conv2d(x, weight)
        torch.cuda.synchronize()
        print(f"implicit gemm output {tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
