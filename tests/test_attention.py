"""Reference attention, online softmax, non-causal semantics, and FlashAttention-2."""

from __future__ import annotations

import pytest
import torch

from inference.attention_backends import (
    attention_explicit,
    attention_flash2,
    attention_online_decode,
    attention_sdpa,
    flash2_status,
)
from models.temporal_transformer import TemporalTransformer


def _qkv(batch=2, heads=4, seq=16, dim=32, scale=1.0, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq, dim) * scale
    k = torch.randn(batch, heads, seq, dim) * scale
    v = torch.randn(batch, heads, seq, dim)
    return q, k, v


def test_explicit_matches_sdpa():
    q, k, v = _qkv()
    explicit = attention_explicit(q, k, v)
    sdpa = attention_sdpa(q, k, v)
    assert torch.allclose(explicit, sdpa, atol=1e-5, rtol=1e-5)


def test_online_decode_matches_explicit_across_lengths():
    for length in (1, 16, 128, 256):
        q, k, v = _qkv(batch=2, heads=4, seq=length, seed=length)
        query = q[:, :, -1, :]
        online = attention_online_decode(query, k, v)
        explicit = attention_explicit(query.unsqueeze(2), k, v).squeeze(2)
        assert torch.allclose(online, explicit, atol=1e-4, rtol=1e-4), length
    q, k, v = _qkv(batch=1, heads=1, seq=1024, seed=1024)
    online = attention_online_decode(q[:, :, -1, :], k, v)
    explicit = attention_explicit(q[:, :, -1:, :], k, v).squeeze(2)
    assert torch.isfinite(online).all()
    assert torch.allclose(online, explicit, atol=1e-3, rtol=1e-3)


def test_online_softmax_is_stable_for_large_scores():
    q, k, v = _qkv(batch=2, heads=2, seq=32, scale=40.0, seed=9)
    query = q[:, :, -1, :]
    online = attention_online_decode(query, k, v)
    explicit = attention_explicit(query.unsqueeze(2), k, v).squeeze(2)
    assert torch.isfinite(online).all()
    assert torch.allclose(online, explicit, atol=1e-3, rtol=1e-3)


def test_attention_is_not_causal():
    q, k, v = _qkv(seq=8, seed=3)
    changed_k = k.clone()
    changed_k[:, :, -1, :] += 4
    original = attention_explicit(q, k, v)
    updated = attention_explicit(q, changed_k, v)
    # Position 0 can see the last key. A causal mask would have hidden it.
    assert not torch.allclose(original[:, :, 0], updated[:, :, 0], atol=1e-5)


def test_single_key_returns_that_value():
    q, k, v = _qkv(seq=4, seed=2)
    out = attention_online_decode(q[:, :, 0, :], k[:, :, :1, :], v[:, :, :1, :])
    assert torch.allclose(out, v[:, :, 0, :], atol=1e-5, rtol=1e-5)


def test_flash2_matches_reference_when_available():
    ok, reason = flash2_status()
    if not ok:
        pytest.skip(reason)
    torch.manual_seed(0)
    q, k, v = _qkv(batch=2, heads=4, seq=128)
    q, k, v = q.cuda().half(), k.cuda().half(), v.cuda().half()
    reference = attention_explicit(q, k, v)
    try:
        flashed = attention_flash2(q, k, v)
    except RuntimeError as exc:
        pytest.skip(f"FlashAttention-2 build rejected this GPU: {exc}")
    assert flashed.shape == reference.shape
    assert torch.isfinite(flashed).all()
    assert torch.allclose(flashed, reference, atol=1e-2, rtol=1e-2)


def test_transformer_flash2_logits_when_available():
    ok, reason = flash2_status()
    if not ok:
        pytest.skip(reason)
    torch.manual_seed(1)
    model = TemporalTransformer().cuda().eval()
    frames = torch.rand(2, 8, 1, 32, 32, device="cuda")
    with torch.no_grad():
        reference = model(frames, backend="torch")
        try:
            flashed = model(frames, backend="flash2")
        except RuntimeError as exc:
            pytest.skip(f"FlashAttention-2 build rejected this GPU: {exc}")
    assert torch.allclose(reference, flashed, atol=5e-2, rtol=5e-2)


def test_cuda_backend_refuses_to_silently_run_pytorch():
    from inference.cuda_backend import CudaBackendUnavailable, kernels_available

    if kernels_available():
        pytest.skip("CUDA extension is available; the refusal path is the other environment")
    model = TemporalTransformer().eval()
    frames = torch.rand(1, 2, 1, 32, 32)
    with pytest.raises(RuntimeError, match="single-query"):
        model(frames, backend="cuda")
    with pytest.raises((CudaBackendUnavailable, RuntimeError)):
        from inference.streaming import stream_logits

        stream_logits(model, frames, backend="cuda")
