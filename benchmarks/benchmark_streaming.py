"""Prefix recomputation versus KV-cache streaming.

End-to-end timing includes the CNN, projections, attention, and the head.
Attention-only timing starts from projected Q/K/V, so it isolates score-matrix
traffic from projection reuse. Random weights are valid for latency; pass a
checkpoint to time the trained model. Outputs are compared either way.

    python benchmarks/benchmark_streaming.py
    python benchmarks/benchmark_streaming.py --checkpoint checkpoints/transformer.pt --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import describe_device, resolve_device, write_csv  # noqa: E402
from benchmarks.plot_results import plot_streaming  # noqa: E402
from inference.attention_backends import attention_explicit, attention_sdpa  # noqa: E402
from inference.cuda_backend import decode_attention, kernels_available, unavailable_reason  # noqa: E402
from inference.streaming import recompute_logits, stream_logits  # noqa: E402
from models.temporal_transformer import TemporalTransformer  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_passes(fn, device, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    totals = []
    per_frame = []
    for _ in range(iters):
        _sync(device)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            frame_ms = fn()
            end.record()
            _sync(device)
            totals.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            frame_ms = fn()
            _sync(device)
            totals.append((time.perf_counter() - t0) * 1e3)
        per_frame.extend(frame_ms)
    return {
        "median_total_ms": float(np.median(totals)),
        "median_per_frame_ms": float(np.median(per_frame)),
    }


def _load_model(path: Path | None, device: torch.device):
    model = TemporalTransformer().to(device).eval()
    note = "random_init"
    if path is not None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        note = f"checkpoint:{path}"
    return model, note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_streaming.csv")
    args = parser.parse_args()
    device = resolve_device(args.device)
    warmup = args.warmup if args.warmup is not None else (3 if device.type == "cuda" else 1)
    iters = args.iters if args.iters is not None else (10 if device.type == "cuda" else 3)
    model, weight_note = _load_model(args.checkpoint, device)
    torch.manual_seed(0)
    frames = torch.rand(args.batch, args.frames, 1, 32, 32, device=device)
    print(f"device: {describe_device(device)}")
    print(f"weights: {weight_note}")
    rows = []

    def add(mode, stats, err, status="ok", note=""):
        frames_per_sec = ""
        if stats is not None and stats["median_total_ms"] > 0:
            frames_per_sec = f"{(args.frames * args.batch) / (stats['median_total_ms'] / 1e3):.2f}"
        rows.append(
            {
                "device": describe_device(device),
                "mode": mode,
                "batch": args.batch,
                "frames": args.frames,
                "median_total_ms": "" if stats is None else f"{stats['median_total_ms']:.4f}",
                "median_per_frame_ms": "" if stats is None else f"{stats['median_per_frame_ms']:.4f}",
                "frames_per_sec": frames_per_sec,
                "max_abs_err_vs_recompute": "" if err is None else f"{err:.3e}",
                "status": status,
                "note": note or weight_note,
            }
        )

    def e2e_recompute():
        times = []
        for t in range(args.frames):
            _sync(device)
            t0 = time.perf_counter()
            model(frames[:, : t + 1], backend="torch")
            _sync(device)
            times.append((time.perf_counter() - t0) * 1e3)
        return times

    def e2e_stream():
        cache = model.init_cache(args.batch, device=device, dtype=frames.dtype)
        times = []
        for t in range(args.frames):
            _sync(device)
            t0 = time.perf_counter()
            model.step(frames[:, t], cache, t=t, backend="torch")
            _sync(device)
            times.append((time.perf_counter() - t0) * 1e3)
        return times

    with torch.no_grad():
        full, _ = recompute_logits(model, frames, backend="torch")
        streamed, _ = stream_logits(model, frames, backend="torch")
        err = float((full - streamed).abs().max().item())
        print(f"torch streaming vs prefix max abs {err:.3e}")
        add("e2e_recompute_prefix", _time_passes(e2e_recompute, device, warmup, iters), 0.0, note=weight_note + "; includes repeated CNN and QKV")
        add("e2e_torch_kv_cache", _time_passes(e2e_stream, device, warmup, iters), err, note=weight_note + "; new frame only, PyTorch attention")

        if kernels_available() and device.type == "cuda":
            streamed_cuda, _ = stream_logits(model, frames, backend="cuda")
            cuda_err = float((full - streamed_cuda).abs().max().item())

            def e2e_cuda():
                cache = model.init_cache(args.batch, device=device, dtype=frames.dtype)
                times = []
                for t in range(args.frames):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model.step(frames[:, t], cache, t=t, backend="cuda")
                    end.record()
                    _sync(device)
                    times.append(float(start.elapsed_time(end)))
                return times

            add("e2e_cuda_kv_decode", _time_passes(e2e_cuda, device, warmup, iters), cuda_err, note=weight_note + "; custom decode kernel")
        else:
            add(
                "e2e_cuda_kv_decode",
                None,
                None,
                status="unavailable",
                note=unavailable_reason() or "custom CUDA decode was not executed",
            )

        tokens = model.embed(frames)
        q, k, v = model.project_qkv(tokens)

        def attn_recompute():
            times = []
            for t in range(args.frames):
                _sync(device)
                t0 = time.perf_counter()
                attention_explicit(q[:, :, t : t + 1], k[:, :, : t + 1], v[:, :, : t + 1])
                _sync(device)
                times.append((time.perf_counter() - t0) * 1e3)
            return times

        def attn_cache():
            cache = model.init_cache(args.batch, device=device, dtype=q.dtype)
            times = []
            for t in range(args.frames):
                cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])
                kk, vv = cache.get_cached_kv()
                _sync(device)
                t0 = time.perf_counter()
                attention_sdpa(q[:, :, t : t + 1], kk.contiguous(), vv.contiguous())
                _sync(device)
                times.append((time.perf_counter() - t0) * 1e3)
            return times

        add(
            "attention_only_recompute_explicit",
            _time_passes(attn_recompute, device, warmup, iters),
            None,
            note="projections already done; explicit 1xT scores are materialized",
        )
        add(
            "attention_only_torch_kv_cache",
            _time_passes(attn_cache, device, warmup, iters),
            None,
            note="projections already done; PyTorch SDPA on the cached prefix",
        )
        if kernels_available() and device.type == "cuda":

            def attn_cuda():
                cache = model.init_cache(args.batch, device=device, dtype=q.dtype)
                times = []
                for t in range(args.frames):
                    cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    decode_attention(q[:, :, t].contiguous(), cache.k, cache.v, cache.get_valid_length())
                    end.record()
                    _sync(device)
                    times.append(float(start.elapsed_time(end)))
                return times

            add(
                "attention_only_cuda_kv_decode",
                _time_passes(attn_cuda, device, warmup, iters),
                None,
                note="projections already done; custom kernel does not write scores",
            )
        else:
            add(
                "attention_only_cuda_kv_decode",
                None,
                None,
                status="unavailable",
                note=unavailable_reason() or "custom CUDA decode was not executed",
            )

    # Theoretical intermediates for the video length, independent of the clock.
    heads, dim = model.heads, model.head_dim
    score_bytes = args.batch * heads * args.frames * args.frames * 4
    kv_bytes = 2 * args.batch * heads * args.frames * dim * 4
    print(
        f"theoretical full score matrix {score_bytes / 1024:.1f} KiB; "
        f"KV cache {kv_bytes / 1024:.1f} KiB. "
        "These are sizes, not measured allocations."
    )
    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    for row in rows:
        print(f"  {row['mode']}: {row['status']} median_total_ms={row['median_total_ms']} per_frame_ms={row['median_per_frame_ms']}")
    plot_streaming(args.out, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
