"""Train the temporal CNN or the single-layer transformer.

Training always uses PyTorch reference attention. FlashAttention-2 and the
custom CUDA decode kernel have no backward path in this project; evaluate a
saved checkpoint with those backends instead of training separate weights.

    python train.py --model cnn --smoke
    python train.py --model transformer --epochs 8 --train-size 1024
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic_video import CLASS_NAMES, make_dataloaders  # noqa: E402
from inference.streaming import stream_logits  # noqa: E402
from models.temporal_cnn import TemporalCNN  # noqa: E402
from models.temporal_transformer import TemporalTransformer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> torch.nn.Module:
    if name == "cnn":
        return TemporalCNN()
    if name == "transformer":
        return TemporalTransformer()
    raise ValueError(f"unknown model {name!r}")


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            pred = model(frames).argmax(dim=-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "transformer"), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="one short epoch on a tiny split")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> dict:
    if args.backend != "torch":
        raise SystemExit(
            "Training uses the PyTorch reference attention backend only. "
            "This project has no FlashAttention-2 or custom-CUDA backward kernels, "
            "and it will not silently train with PyTorch when another backend is requested. "
            "Train with --backend torch, then evaluate the checkpoint with demo.py."
        )
    if args.smoke:
        args.epochs = 1
        args.train_size = min(args.train_size, 128)
        args.val_size = min(args.val_size, 32)
        args.test_size = min(args.test_size, 32)
        args.batch_size = min(args.batch_size, 16)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)
    loaders = make_dataloaders(
        args.train_size,
        args.val_size,
        args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for frames, labels in loaders["train"]:
            frames = frames.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.shape[0]
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.shape[0])
        val_acc = accuracy(model, loaders["val"], device)
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
            f"train_acc {train_acc:.3f}  val_acc {val_acc:.3f}"
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    test_acc = accuracy(model, loaders["test"], device)
    print(f"test_acc {test_acc:.3f}  best_val_acc {best_val:.3f}")

    streaming_error = None
    if args.model == "transformer":
        model.eval()
        frames, _ = next(iter(loaders["test"]))
        frames = frames[:2].to(device)
        with torch.no_grad():
            full = model(frames)
            streamed, _ = stream_logits(model, frames, backend="torch")
        streaming_error = float((full - streamed).abs().max().item())
        print(f"streaming_vs_prefix_max_abs {streaming_error:.3e}")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = ROOT / "checkpoints" / f"{args.model}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "state_dict": best_state,
        "class_names": list(CLASS_NAMES),
        "seed": args.seed,
        "metrics": {
            "best_val_acc": best_val,
            "test_acc": test_acc,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "streaming_vs_prefix_max_abs": streaming_error,
        },
    }
    torch.save(payload, ckpt_path)
    print(f"saved {ckpt_path}")
    return payload


def main(argv=None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
