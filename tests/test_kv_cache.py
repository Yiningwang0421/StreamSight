"""KV cache bookkeeping and incremental equivalence with full-prefix attention."""

from __future__ import annotations

import pytest
import torch

from inference.kv_cache import KVCache
from inference.streaming import recompute_logits, stream_logits
from models.temporal_transformer import TemporalTransformer


def _token(value: float, length: int = 1) -> torch.Tensor:
    tensor = torch.full((1, 1, length, 32), value, dtype=torch.float32)
    tensor[..., :] = torch.arange(32, dtype=torch.float32) + value
    return tensor


def test_append_reset_and_contents():
    cache = KVCache(1, 1, 4, 32)
    first_k = _token(0.0)
    first_v = _token(10.0)
    cache.append(first_k, first_v)
    assert cache.get_valid_length() == 1
    k, v = cache.get_cached_kv()
    assert torch.equal(k, first_k)
    assert torch.equal(v, first_v)

    second_k = _token(1.0, length=2)
    second_v = _token(20.0, length=2)
    cache.append(second_k, second_v)
    assert cache.get_valid_length() == 3
    k, v = cache.get_cached_kv()
    assert torch.equal(k[:, :, 1:], second_k)
    assert torch.equal(v[:, :, :1], first_v)
    # The physical buffer is not reallocated as the prefix grows.
    assert cache.k.shape[2] == 4
    wide = KVCache(2, 2, 4, 32)
    wide.append(torch.ones(2, 2, 1, 32), torch.ones(2, 2, 1, 32))
    view, _ = wide.get_cached_kv()
    assert not view.is_contiguous()

    cache.reset()
    assert cache.get_valid_length() == 0
    assert torch.count_nonzero(cache.k) == 0
    cache.append(second_k, second_v)
    k, v = cache.get_cached_kv()
    assert torch.equal(k, second_k)
    assert cache.get_valid_length() == 2


def test_capacity_overflow_does_not_advance_length():
    cache = KVCache(2, 4, 3, 32)
    cache.append(torch.ones(2, 4, 2, 32), torch.ones(2, 4, 2, 32))
    with pytest.raises(RuntimeError, match="capacity overflow"):
        cache.append(torch.ones(2, 4, 2, 32), torch.ones(2, 4, 2, 32))
    assert cache.get_valid_length() == 2
    k, _ = cache.get_cached_kv()
    assert torch.equal(k, torch.ones(2, 4, 2, 32))


def test_streaming_matches_full_prefix_and_cached_projections():
    torch.manual_seed(4)
    model = TemporalTransformer().eval()
    frames = torch.rand(2, 6, 1, 32, 32)
    with torch.no_grad():
        full, recomputed = recompute_logits(model, frames, backend="torch")
        streamed, steps = stream_logits(model, frames, backend="torch")
        tokens = model.embed(frames, time_offset=0)
        _, k_full, v_full = model.project_qkv(tokens)
        cache = model.init_cache(2)
        for t in range(frames.shape[1]):
            model.step(frames[:, t], cache, t=t, backend="torch")
        k_cache, v_cache = cache.get_cached_kv()
    assert torch.allclose(full, streamed, atol=1e-4, rtol=1e-4)
    for t, step in enumerate(steps):
        assert torch.allclose(step, recomputed[t], atol=1e-4, rtol=1e-4)
    assert torch.allclose(k_cache, k_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_cache, v_full, atol=1e-5, rtol=1e-5)


def test_single_frame_streaming():
    torch.manual_seed(5)
    model = TemporalTransformer().eval()
    frames = torch.rand(1, 1, 1, 32, 32)
    with torch.no_grad():
        full = model(frames)
        streamed, _ = stream_logits(model, frames, backend="torch")
    assert torch.allclose(full, streamed, atol=1e-5, rtol=1e-5)
