"""Attention implementations used by the temporal transformer.

Backends
--------
torch / sdpa
    ``F.scaled_dot_product_attention``, non-causal. This is the training and
    correctness reference.
explicit
    Materialized QK^T, softmax, then multiply by V. This is the memory-traffic
    baseline: the score matrix is a real tensor.
flash2
    Official Dao-AILab FlashAttention-2 (``flash_attn.flash_attn_func``),
    ``causal=False``. FP16 or BF16 CUDA tensors only. Missing installations and
    unsupported GPUs raise; this function does not call PyTorch instead.
cuda
    Custom streaming decode kernel. One query token, KV already cached.
    Implemented in ``cuda/decode_attention.cu``. Not FlashAttention-2.

The online-softmax helper is the FP32 recurrence implemented by the custom
kernel (FlashAttention-2's unnormalized accumulator: rescale by the running
max, divide by the normalizer once at the end). It exists so CPU tests can
check that recurrence against dense attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def attention_explicit(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Non-causal attention. q, k, v are [B, H, S, D]. Scores are materialized."""
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"incompatible attention shapes q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    scale = _scale(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float())
    return out.to(dtype=q.dtype)


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch scaled dot-product attention, non-causal, dropout 0."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


def attention_online_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-query online softmax.

    q: [B, H, D]
    k, v: [B, H, S, D]
    returns [B, H, D]

    This is the recurrence in ``decode_attention.cu``. It does not allocate an
    [S] score buffer as the values that participate in the output; the Python
    loop keeps only the running max, normalizer, and output accumulator.
    """
    if q.dim() != 3 or k.dim() != 4 or v.shape != k.shape:
        raise ValueError("expected q [B,H,D] and k,v [B,H,S,D]")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[2] != k.shape[3]:
        raise ValueError(f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)}")
    length = k.shape[2]
    if length <= 0:
        raise ValueError("valid length must be positive")
    scale = _scale(q.shape[-1])
    qf = q.float()
    kf = k.float()
    vf = v.float()
    running_max = torch.full(
        (q.shape[0], q.shape[1], 1),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    normalizer = torch.zeros(q.shape[0], q.shape[1], 1, device=q.device, dtype=torch.float32)
    acc = torch.zeros(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32)
    for t in range(length):
        score = (qf * kf[:, :, t, :]).sum(dim=-1, keepdim=True) * scale
        new_max = torch.maximum(running_max, score)
        alpha = torch.exp(running_max - new_max)
        alpha = torch.where(torch.isneginf(running_max), torch.zeros_like(alpha), alpha)
        weight = torch.exp(score - new_max)
        normalizer = normalizer * alpha + weight
        acc = acc * alpha + weight * vf[:, :, t, :]
        running_max = new_max
    return (acc / normalizer).to(dtype=q.dtype)


def flash2_status() -> tuple[bool, str]:
    """Whether official FlashAttention-2 can run on this process."""
    if not torch.cuda.is_available():
        return (
            False,
            "CUDA is not available. Official FlashAttention-2 was not executed.",
        )
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return (
            False,
            "flash_attn is not installed. Official FlashAttention-2 was not executed. "
            "On a supported NVIDIA GPU, install Dao-AILab flash-attention. "
            "This project does not vendor that source and does not replace it with PyTorch.",
        )
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        return (
            False,
            f"GPU compute capability {major}.{minor} is below 8.0. "
            "The standard FlashAttention-2 CUDA build expects Ampere (SM80) or newer, "
            "so the FlashAttention-2 backend was skipped.",
        )
    return True, ""


def attention_flash2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Official FlashAttention-2. q, k, v are [B, H, S, D], FP16 or BF16, CUDA.

    Raises if the library or the GPU cannot run it. There is no PyTorch fallback.
    """
    ok, reason = flash2_status()
    if not ok:
        raise RuntimeError(reason)
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise RuntimeError(
            "FlashAttention-2 was requested with a non-CUDA tensor. "
            "Refusing to fall back to PyTorch."
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"FlashAttention-2 supports FP16/BF16, got {q.dtype}. "
            "Refusing to run an FP32 PyTorch kernel under the flash2 backend."
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError("FlashAttention-2 requires Q, K, and V to share a dtype")
    from flash_attn import flash_attn_func

    # flash_attn_func expects [B, S, H, D].
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=_scale(q.shape[-1]),
        causal=False,
    )
    return out.transpose(1, 2).contiguous()


def dispatch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str = "torch") -> torch.Tensor:
    """Run non-causal attention. ``cuda`` requires a single query token."""
    if backend in ("torch", "sdpa"):
        return attention_sdpa(q, k, v)
    if backend == "explicit":
        return attention_explicit(q, k, v)
    if backend == "flash2":
        return attention_flash2(q, k, v)
    if backend == "cuda":
        if q.shape[-2] != 1:
            raise RuntimeError(
                "Custom CUDA decode attention expects one query token "
                f"(q length {q.shape[-2]}). It is not a full-sequence FlashAttention-2 kernel."
            )
        from inference.cuda_backend import decode_attention

        out = decode_attention(q.squeeze(-2), k, v, k.shape[-2])
        return out.unsqueeze(-2)
    raise ValueError(
        f"unknown attention backend {backend!r}. "
        "Choose torch, explicit, flash2, or cuda."
    )
