"""One-layer temporal transformer with interchangeable attention backends.

Non-causal self-attention runs over the frames that have actually been
provided. Nothing in this module builds a causal mask. Streaming inference
feeds only the arrived prefix, and the classifier reads the newest token.

K and V are linear projections of the per-token pre-attention representation
(absolute position added, then LayerNorm). With a single attention layer those
projections do not depend on other frames, so a KV cache filled one frame at
a time matches the newest-token output of full-prefix attention.

A second unmasked layer would make the next K/V depend on attention outputs
that have already mixed in later frames. That cache would not match streaming,
so this MVP stays at one layer.
"""

from __future__ import annotations

import torch
from torch import nn

from data.synthetic_video import NUM_CLASSES
from inference.attention_backends import dispatch_attention
from inference.kv_cache import KVCache
from models.frame_encoder import FrameEncoder


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        head_dim: int = 32,
        ffn_dim: int = 256,
        max_context: int = 256,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if dim != heads * head_dim:
            raise ValueError(f"dim ({dim}) must equal heads * head_dim ({heads * head_dim})")
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.max_context = max_context
        self.encoder = FrameEncoder(dim)
        self.pos = nn.Parameter(torch.zeros(1, max_context, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.ln1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)
        self.head = nn.Linear(dim, num_classes)

    def embed(self, frames: torch.Tensor, time_offset: int = 0) -> torch.Tensor:
        tokens = self.encoder(frames)
        length = tokens.shape[1]
        if time_offset < 0 or time_offset + length > self.max_context:
            raise ValueError(
                f"positions {time_offset}:{time_offset + length} exceed max_context {self.max_context}"
            )
        return tokens + self.pos[:, time_offset : time_offset + length]

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x = x.view(batch, length, self.heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch, length, heads * dim)

    def project_qkv(self, tokens: torch.Tensor):
        hidden = self.ln1(tokens)
        q = self.split_heads(self.q_proj(hidden))
        k = self.split_heads(self.k_proj(hidden))
        v = self.split_heads(self.v_proj(hidden))
        return q, k, v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
        if backend == "flash2" and q.dtype == torch.float32:
            # Official FlashAttention-2 does not accept FP32. Cast only the
            # attention inputs; the residual stream stays in the model dtype.
            attended = dispatch_attention(q.half(), k.half(), v.half(), backend="flash2")
            return attended.to(dtype=q.dtype)
        return dispatch_attention(q, k, v, backend=backend)

    def apply_block(self, tokens: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        mixed = self.out_proj(self.merge_heads(attended))
        hidden = tokens + mixed
        ffn = self.ffn_down(torch.nn.functional.gelu(self.ffn_up(self.ln2(hidden))))
        return hidden + ffn

    def forward(self, frames: torch.Tensor, backend: str = "torch") -> torch.Tensor:
        if backend == "cuda":
            raise RuntimeError(
                "The custom CUDA kernel is a single-query decode kernel. "
                "Call step() / streaming inference with backend='cuda'. "
                "Full-prefix attention is backend='torch' or backend='flash2'. "
                "Refusing to run the CUDA request as PyTorch attention."
            )
        tokens = self.embed(frames, time_offset=0)
        q, k, v = self.project_qkv(tokens)
        attended = self._attend(q, k, v, backend=backend)
        hidden = self.apply_block(tokens, attended)
        return self.head(hidden[:, -1])

    def init_cache(self, batch_size: int, device=None, dtype=None) -> KVCache:
        device = device if device is not None else self.pos.device
        dtype = dtype if dtype is not None else self.pos.dtype
        return KVCache(
            batch_size=batch_size,
            heads=self.heads,
            max_context=self.max_context,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def step(self, frame: torch.Tensor, cache: KVCache, t: int, backend: str = "torch") -> torch.Tensor:
        """Classify the prefix ending at absolute time ``t``.

        ``frame`` is [B, 1, H, W] or [B, 1, 1, H, W]. Only this frame is encoded.
        Historical K/V stay in ``cache``.
        """
        if frame.dim() == 4:
            frame = frame.unsqueeze(1)
        if frame.shape[1] != 1:
            raise ValueError("step() expects a single new frame")
        tokens = self.embed(frame, time_offset=t)
        q, k, v = self.project_qkv(tokens)
        cache.append(k, v)
        if backend == "cuda":
            # Read the preallocated buffer in place. Do not slice-and-copy it.
            from inference.cuda_backend import decode_attention

            attended = decode_attention(
                q.squeeze(2), cache.k, cache.v, cache.get_valid_length()
            ).unsqueeze(2)
        else:
            k_prefix, v_prefix = cache.get_cached_kv()
            attended = self._attend(
                q, k_prefix.contiguous(), v_prefix.contiguous(), backend=backend
            )
        hidden = self.apply_block(tokens, attended)
        return self.head(hidden[:, -1])
