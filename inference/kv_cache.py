"""Preallocated KV cache.

Layout follows the streaming decode kernel:

    K, V: [batch, heads, max_context, head_dim]

``append`` writes into the next free slots. It does not call ``torch.cat`` and
does not reallocate. ``get_cached_kv`` returns a view of the valid prefix; that
view is not contiguous when ``length < max_context`` because the physical
stride still spans the full capacity. The CUDA kernel therefore takes the base
tensors plus ``valid_length`` instead of the view.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        batch_size: int,
        heads: int,
        max_context: int,
        head_dim: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(batch_size, heads, max_context, head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.max_context = int(max_context)
        self.k = torch.zeros(batch_size, heads, max_context, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, heads, max_context, head_dim, device=device, dtype=dtype)
        self.length = 0

    def reset(self) -> None:
        self.length = 0
        self.k.zero_()
        self.v.zero_()

    def get_valid_length(self) -> int:
        return self.length

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append tokens. ``k`` and ``v`` are [B, H, S, D]."""
        if k.shape != v.shape:
            raise ValueError(f"K and V shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.dim() != 4:
            raise ValueError(f"expected K/V rank 4 [B, H, S, D], got {tuple(k.shape)}")
        if k.shape[0] != self.k.shape[0] or k.shape[1] != self.k.shape[1] or k.shape[3] != self.k.shape[3]:
            raise ValueError(
                f"K/V {tuple(k.shape)} does not match cache "
                f"[B={self.k.shape[0]}, H={self.k.shape[1]}, Cap={self.max_context}, D={self.k.shape[3]}]"
            )
        if k.dtype != self.k.dtype or v.dtype != self.v.dtype:
            raise TypeError(
                f"cache dtype is {self.k.dtype}; refusing to cast appended K/V "
                f"({k.dtype}, {v.dtype})"
            )
        incoming = k.shape[2]
        if self.length + incoming > self.max_context:
            raise RuntimeError(
                f"KV cache capacity overflow: length {self.length} + {incoming} "
                f"> capacity {self.max_context}"
            )
        end = self.length + incoming
        self.k[:, :, self.length : end].copy_(k)
        self.v[:, :, self.length : end].copy_(v)
        self.length = end

    def get_cached_kv(self):
        length = self.length
        return self.k[:, :, :length], self.v[:, :, :length]
