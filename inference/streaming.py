"""Prefix recomputation versus KV-cache streaming.

Both paths call the same module, so they share weights, positions, and
attention semantics. ``recompute_logits`` runs the full prefix after every new
frame (CNN, QKV projections, and attention over every token in the prefix).
``stream_logits`` encodes only the new frame, appends its K/V, and attends with
the newest query.

Those are different savings. Skipping historical projections avoids repeated
CNN and GEMM work. The attention kernel, separately, avoids writing an S-by-S
or 1-by-S score tensor. A long prefix can be projection-bound, attention-bound,
or both; the streaming benchmark times them apart.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


@torch.no_grad()
def recompute_logits(
    model, frames: torch.Tensor, backend: str = "torch"
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Re-run the transformer on frames[:, :t+1] at every step."""
    if backend == "cuda":
        raise RuntimeError(
            "Prefix recomputation is a full-sequence attention workload. "
            "backend='cuda' is only the single-query decode kernel. "
            "Use backend='torch' or backend='flash2' for recomputation."
        )
    steps: List[torch.Tensor] = []
    length = frames.shape[1]
    for t in range(length):
        steps.append(model(frames[:, : t + 1], backend=backend))
    return steps[-1], steps


@torch.no_grad()
def stream_logits(
    model, frames: torch.Tensor, backend: str = "torch"
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Incremental inference. ``backend='cuda'`` uses the custom decode kernel."""
    batch = frames.shape[0]
    cache = model.init_cache(batch, device=frames.device, dtype=frames.dtype)
    steps: List[torch.Tensor] = []
    for t in range(frames.shape[1]):
        steps.append(model.step(frames[:, t], cache, t=t, backend=backend))
    return steps[-1], steps
