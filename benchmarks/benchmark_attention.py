"""Attention latency and memory.

Full-sequence rows compare materialized QK^T, PyTorch SDPA, and official
FlashAttention-2. The custom CUDA kernel is only timed on the one-query
workload it actually implements. CPU defaults are shorter than the GPU sweep
so a laptop can finish; pass the full lists explicitly on an NVIDIA GPU.

    python benchmarks/benchmark_attention.py
    python benchmarks/benchmark_attention.py --device cuda --seq-lens 128 256 512 1024 2048 --batches 1 4 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import blank_if_nan, describe_device, resolve_device, write_csv  # noqa: E402
from benchmarks.plot_results import plot_attention  # noqa: E402
from inference.attention_backends import (  # noqa: E402
    attention_explicit,
    attention_flash2,
    attention_sdpa,
    flash2_status,
)
from inference.cuda_backend import decode_attention, kernels_available, unavailable_reason  # noqa: E402
from inference.timing import cuda_allocation_mb, measure_ms  # noqa: E402


def _score_mb(batch, heads, seq_q, seq_kv, dtype) -> float:
    element = torch.tensor([], dtype=dtype).element_size()
    return batch * heads * seq_q * seq_kv * element / (1024 ** 2)


def _record(rows, **kwargs):
    rows.append(kwargs)


def _time_backend(rows, name, fn, reference, device, meta, warmup, iters):
    try:
        got = fn()
        err = float((got.float() - reference.float()).abs().max().item())
        stats = measure_ms(fn, device, warmup=warmup, iters=iters)
        _, extra = cuda_allocation_mb(fn, device)
        _record(
            rows,
            status="ok",
            median_ms=f"{stats['median_ms']:.4f}",
            mean_ms=f"{stats['mean_ms']:.4f}",
            extra_alloc_mb=blank_if_nan(extra),
            max_abs_err=f"{err:.3e}",
            note="",
            backend=name,
            **meta,
        )
        print(
            f"{name} B={meta['batch']} Sq={meta['seq_q']} Skv={meta['seq_kv']} "
            f"{stats['median_ms']:.3f} ms  err {err:.2e}"
        )
    except RuntimeError as exc:
        message = str(exc).splitlines()[0][:240]
        status = "oom" if "out of memory" in message.lower() else "error"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _record(
            rows,
            status=status,
            median_ms="",
            mean_ms="",
            extra_alloc_mb="",
            max_abs_err="",
            note=message,
            backend=name,
            **meta,
        )
        if status != "oom":
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seq-lens", type=int, nargs="*", default=None)
    parser.add_argument("--batches", type=int, nargs="*", default=None)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_attention.csv")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        seqs = args.seq_lens or [128, 256, 512, 1024, 2048]
        batches = args.batches or [1, 4, 16]
        dtype = torch.float16
        warmup = args.warmup if args.warmup is not None else 5
        iters = args.iters if args.iters is not None else 20
    else:
        seqs = args.seq_lens or [128, 256]
        batches = args.batches or [1, 4]
        dtype = torch.float32
        warmup = args.warmup if args.warmup is not None else 1
        iters = args.iters if args.iters is not None else 3
        print("CPU defaults are seq 128/256 and batch 1/4. The 128..2048 GPU sweep is not being invented here.")
    print(f"device: {describe_device(device)}  dtype: {dtype}")
    flash_ok, flash_reason = flash2_status()
    cuda_ok = kernels_available() and device.type == "cuda"
    rows = []
    for batch in batches:
        for seq in seqs:
            torch.manual_seed(0)
            q = torch.randn(batch, args.heads, seq, args.head_dim, device=device, dtype=dtype)
            k = torch.randn(batch, args.heads, seq, args.head_dim, device=device, dtype=dtype)
            v = torch.randn(batch, args.heads, seq, args.head_dim, device=device, dtype=dtype)
            reference = attention_explicit(q, k, v)
            meta = {
                "device": describe_device(device),
                "precision": str(dtype).replace("torch.", ""),
                "workload": "full_sequence",
                "batch": batch,
                "heads": args.heads,
                "seq_q": seq,
                "seq_kv": seq,
                "head_dim": args.head_dim,
                "theoretical_score_mb": f"{_score_mb(batch, args.heads, seq, seq, dtype):.4f}",
            }
            _time_backend(rows, "explicit_qk_softmax_v", lambda: attention_explicit(q, k, v), reference, device, meta, warmup, iters)
            _time_backend(rows, "pytorch_sdpa", lambda: attention_sdpa(q, k, v), reference, device, meta, warmup, iters)
            if flash_ok and dtype in (torch.float16, torch.bfloat16):
                _time_backend(
                    rows,
                    "flashattention2",
                    lambda: attention_flash2(q, k, v),
                    reference,
                    device,
                    meta,
                    warmup,
                    iters,
                )
            else:
                _record(
                    rows,
                    status="unavailable",
                    median_ms="",
                    mean_ms="",
                    extra_alloc_mb="",
                    max_abs_err="",
                    note=flash_reason or "FlashAttention-2 was not executed for this dtype",
                    backend="flashattention2",
                    **meta,
                )

            # One-query decode, the only workload the custom kernel implements.
            q1 = q[:, :, -1, :]
            reference_1 = attention_explicit(q1.unsqueeze(2), k, v).squeeze(2)
            meta_1 = dict(meta)
            meta_1.update(
                {
                    "workload": "one_query",
                    "seq_q": 1,
                    "precision": "float32" if not cuda_ok else meta["precision"],
                    "theoretical_score_mb": f"{_score_mb(batch, args.heads, 1, seq, dtype):.4f}",
                }
            )
            # Compare one-query backends in FP32 when the custom kernel is the
            # subject, and also keep an FP32 explicit/SDPA measurement on CPU.
            q1_fp = q1.float()
            k_fp = k.float()
            v_fp = v.float()
            reference_fp = attention_explicit(q1_fp.unsqueeze(2), k_fp, v_fp).squeeze(2)
            meta_fp = dict(meta_1)
            meta_fp["precision"] = "float32"
            meta_fp["theoretical_score_mb"] = f"{_score_mb(batch, args.heads, 1, seq, torch.float32):.4f}"
            _time_backend(
                rows,
                "explicit_qk_softmax_v",
                lambda: attention_explicit(q1_fp.unsqueeze(2), k_fp, v_fp).squeeze(2),
                reference_fp,
                device,
                meta_fp,
                warmup,
                iters,
            )
            _time_backend(
                rows,
                "pytorch_sdpa",
                lambda: attention_sdpa(q1_fp.unsqueeze(2), k_fp, v_fp).squeeze(2),
                reference_fp,
                device,
                meta_fp,
                warmup,
                iters,
            )
            if cuda_ok and args.head_dim == 32:
                _time_backend(
                    rows,
                    "cuda_decode",
                    lambda: decode_attention(q1_fp, k_fp, v_fp, seq),
                    reference_fp,
                    device,
                    meta_fp,
                    warmup,
                    iters,
                )
            else:
                _record(
                    rows,
                    status="unavailable",
                    median_ms="",
                    mean_ms="",
                    extra_alloc_mb="",
                    max_abs_err="",
                    note=unavailable_reason()
                    or "custom decode kernel was not executed (it is FP32, head dim 32, one query)",
                    backend="cuda_decode",
                    **meta_fp,
                )
            # Silence unused FP16 one-query reference when dtype is fp32; kept
            # so the full-sequence reference above stays the comparison point.
            del reference_1
    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    plot_attention(args.out, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
