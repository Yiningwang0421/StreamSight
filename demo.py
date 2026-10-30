"""Run one synthetic clip through a trained classifier.

    python demo.py --model cnn
    python demo.py --model transformer --backend torch
    python demo.py --model transformer --backend torch --streaming
    python demo.py --model transformer --backend flash2
    python demo.py --model transformer --backend cuda --streaming

``flash2`` is official Dao-AILab FlashAttention-2 on the full prefix (causal=False).
``cuda --streaming`` is this project's single-query decode kernel over a KV cache.
If the requested backend cannot run, the demo exits. It does not substitute PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic_video import CLASS_NAMES, generate_split  # noqa: E402
from inference.cuda_backend import CudaBackendUnavailable  # noqa: E402
from inference.streaming import stream_logits  # noqa: E402
from inference.timing import measure_ms  # noqa: E402
from train import build_model, run_training, parse_args as train_parse_args  # noqa: E402


BACKEND_HELP = {
    "torch": "PyTorch scaled dot-product attention, full prefix, non-causal",
    "flash2": "official FlashAttention-2, full prefix, causal=False (not the custom kernel)",
    "cuda": "custom CUDA streaming decode kernel (not FlashAttention-2)",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "transformer"), required=True)
    parser.add_argument("--backend", choices=("torch", "flash2", "cuda"), default="torch")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-frames", type=Path, default=None)
    parser.add_argument("--train-if-missing", action="store_true", default=True)
    parser.add_argument("--no-train-if-missing", action="store_false", dest="train_if_missing")
    return parser.parse_args(argv)


def _load(args):
    path = args.checkpoint or (ROOT / "checkpoints" / f"{args.model}.pt")
    if not path.exists():
        if not args.train_if_missing:
            raise SystemExit(f"checkpoint not found: {path}. Train with train.py first.")
        print(f"no checkpoint at {path}; training a smoke {args.model} model with the PyTorch backend")
        smoke = train_parse_args(["--model", args.model, "--smoke", "--seed", str(args.seed), "--checkpoint", str(path)])
        if args.device is not None:
            smoke.device = args.device
        run_training(smoke)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("model") != args.model:
        raise SystemExit(f"checkpoint model is {payload.get('model')!r}, demo requested {args.model!r}")
    model = build_model(args.model)
    model.load_state_dict(payload["state_dict"])
    return model, payload, path


def _predict(model, frames, args):
    if args.model == "cnn":
        if args.streaming or args.backend != "torch":
            raise SystemExit(
                "The CNN baseline is a left-padded temporal convolution, not an attention backend. "
                "Run it as: python demo.py --model cnn"
            )
        return model(frames)

    if args.streaming:
        if args.backend == "flash2":
            raise SystemExit(
                "backend=flash2 in this demo is full-prefix FlashAttention-2. "
                "Streaming decode is --backend cuda --streaming, or --backend torch --streaming "
                "for the PyTorch KV-cache reference. Refusing to relabel one as the other."
            )
        logits, _ = stream_logits(model, frames, backend=args.backend)
        return logits

    if args.backend == "cuda":
        raise SystemExit(
            "The custom CUDA kernel does not implement full-prefix attention. "
            "Use --backend cuda --streaming, or --backend flash2 for official FlashAttention-2."
        )
    return model(frames, backend=args.backend)


def _save_frames(path: Path, video: torch.Tensor, label: str, visible) -> None:
    import matplotlib.pyplot as plt

    frames = video.shape[0]
    fig, axes = plt.subplots(1, frames, figsize=(frames * 0.55, 1.1), squeeze=False)
    for t in range(frames):
        ax = axes[0, t]
        ax.imshow(video[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if not bool(visible[t]):
            for spine in ax.spines.values():
                spine.set_color("crimson")
                spine.set_linewidth(1.5)
    fig.suptitle(f"ground truth: {label}", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    model, payload, path = _load(args)
    model.to(device).eval()
    data = generate_split("test", n=max(args.index + 1, 8), seed=args.seed)
    video = data["videos"][args.index]
    label_id = int(data["labels"][args.index])
    label = CLASS_NAMES[label_id]
    frames = video.unsqueeze(0).to(device)

    def run():
        with torch.no_grad():
            return _predict(model, frames, args)

    try:
        logits = run()
    except (CudaBackendUnavailable, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pred_id = int(logits.argmax(dim=-1).item())
    pred = CLASS_NAMES[pred_id]
    timing = measure_ms(run, device, warmup=2, iters=5)
    mode = "streaming" if args.streaming else "full-prefix"
    backend_line = BACKEND_HELP[args.backend] if args.model == "transformer" else "temporal convolution baseline"
    print(f"checkpoint: {path}")
    if "metrics" in payload:
        metrics = payload["metrics"]
        if metrics.get("test_acc") is not None:
            print(f"checkpoint_test_acc: {metrics['test_acc']:.3f}")
    print(f"model: {args.model}")
    print(f"execution: {mode}")
    print(f"backend: {backend_line}")
    print(f"ground_truth: {label}")
    print(f"prediction: {pred}")
    print(f"match: {pred_id == label_id}")
    print(f"median_latency_ms: {timing['median_ms']:.3f}")
    if args.save_frames is not None:
        _save_frames(args.save_frames, video.cpu(), label, data["visible"][args.index])
        print(f"frames: {args.save_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
