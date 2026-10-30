"""Implicit-GEMM reference tests, plus CUDA kernel tests when the extension is built.

CPU runs execute the address map, the tiled reference, and explicit im2col.
They do not execute the CUDA kernels. Skipped CUDA tests are not passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from inference.attention_backends import attention_explicit
from inference.cuda_backend import kernels_available, unavailable_reason
from models.conv_mapping import (
    explicit_im2col_conv2d,
    implicit_gemm_conv2d_reference,
    scalar_implicit_gemm,
)

ROOT = Path(__file__).resolve().parents[1]


def _conv_pair(n, cin, cout, height, width, seed):
    torch.manual_seed(seed)
    x = torch.randn(n, cin, height, width)
    weight = torch.randn(cout, cin, 3, 3)
    return x, weight


def test_hand_checked_padding_and_center():
    x = torch.ones(1, 1, 3, 3)
    weight = torch.ones(1, 1, 3, 3)
    y, _ = implicit_gemm_conv2d_reference(x, weight)
    ref = F.conv2d(x, weight, padding=1)
    assert torch.equal(y, ref)
    assert y[0, 0, 0, 0].item() == 4
    assert y[0, 0, 0, 1].item() == 6
    assert y[0, 0, 1, 1].item() == 9

    x = torch.zeros(1, 2, 1, 1)
    x[0, :, 0, 0] = torch.tensor([3.0, 5.0])
    weight = torch.randn(4, 2, 3, 3)
    manual = (weight[:, :, 1, 1] * x[0, :, 0, 0].view(1, 2)).sum(dim=-1)
    y, _ = implicit_gemm_conv2d_reference(x, weight)
    assert torch.allclose(y.view(-1), manual, atol=1e-5)


@pytest.mark.parametrize(
    ("n", "cin", "cout", "height", "width"),
    [
        (1, 1, 1, 4, 4),
        (2, 1, 16, 8, 8),
        (1, 3, 5, 7, 5),
        (2, 1, 16, 32, 32),
        (1, 16, 32, 16, 16),
        (2, 8, 7, 9, 6),
    ],
)
def test_implicit_reference_matches_conv_and_explicit_im2col(n, cin, cout, height, width):
    x, weight = _conv_pair(n, cin, cout, height, width, seed=height * 100 + cin)
    ref = F.conv2d(x, weight, padding=1)
    implicit, stats = implicit_gemm_conv2d_reference(x, weight, tile_k=8)
    explicit, explicit_stats = explicit_im2col_conv2d(x, weight)
    assert torch.allclose(implicit, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(explicit, ref, atol=1e-4, rtol=1e-4)
    assert stats["max_tile_elements"] <= stats["gemm_m"] * 8
    assert stats["max_tile_elements"] < stats["full_im2col_elements"]
    assert explicit_stats["im2col_elements"] == explicit_stats["full_im2col_elements"]


def test_scalar_address_map_matches_conv2d():
    x, weight = _conv_pair(1, 2, 3, 4, 4, seed=11)
    y = scalar_implicit_gemm(x, weight)
    ref = F.conv2d(x, weight, padding=1)
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


def test_implicit_gemm_source_does_not_allocate_im2col():
    source = (ROOT / "cuda" / "implicit_gemm_conv.cu").read_text()
    assert "load_virtual_a" in source
    assert "__shared__" in source
    assert "cudaMalloc" not in source
    assert "at::im2col" not in source
    assert "BM" in source and "BN" in source and "BK" in source


def test_decode_kernel_source_contract():
    source = (ROOT / "cuda" / "decode_attention.cu").read_text()
    assert "__shfl_xor_sync" in source
    assert "running_max" in source
    assert "normalizer" in source
    assert "cudaMalloc" not in source
    lowered = source.lower()
    assert "this kernel is not flashattention-2" in lowered
    assert "no score vector is written to global memory" in lowered


def _require_cuda_kernels():
    if not kernels_available():
        pytest.skip(unavailable_reason())


@pytest.mark.parametrize("length", [1, 16, 128, 256, 1024])
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_cuda_decode_matches_pytorch(length, batch):
    _require_cuda_kernels()
    from inference.cuda_backend import decode_attention

    torch.manual_seed(length + batch)
    heads, dim, capacity = 4, 32, length
    q = torch.randn(batch, heads, dim, device="cuda")
    k = torch.randn(batch, heads, capacity, dim, device="cuda")
    v = torch.randn(batch, heads, capacity, dim, device="cuda")
    got = decode_attention(q, k, v, length)
    ref = attention_explicit(q.unsqueeze(2), k, v).squeeze(2)
    atol = 1e-3 if length >= 1024 else 1e-4
    assert torch.allclose(got, ref, atol=atol, rtol=atol)


@pytest.mark.parametrize(
    ("n", "cin", "cout", "height", "width"),
    [
        (1, 1, 1, 3, 3),
        (2, 1, 16, 32, 32),
        (1, 16, 32, 16, 16),
        (2, 3, 7, 7, 5),
    ],
)
def test_cuda_implicit_gemm_matches_conv2d(n, cin, cout, height, width):
    _require_cuda_kernels()
    from inference.cuda_backend import implicit_gemm_conv2d

    x, weight = _conv_pair(n, cin, cout, height, width, seed=17)
    x = x.cuda()
    weight = weight.cuda()
    got = implicit_gemm_conv2d(x, weight)
    ref = F.conv2d(x, weight, padding=1)
    explicit, stats = explicit_im2col_conv2d(x.cpu(), weight.cpu())
    assert stats["im2col_elements"] == n * cin * 9 * height * width
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(got.cpu(), explicit, atol=1e-4, rtol=1e-4)


def test_cuda_streaming_matches_full_prefix():
    _require_cuda_kernels()
    from inference.streaming import stream_logits
    from models.temporal_transformer import TemporalTransformer

    torch.manual_seed(0)
    model = TemporalTransformer().cuda().eval()
    frames = torch.rand(2, 8, 1, 32, 32, device="cuda")
    with torch.no_grad():
        full = model(frames, backend="torch")
        streamed, _ = stream_logits(model, frames, backend="cuda")
    assert torch.allclose(full, streamed, atol=1e-4, rtol=1e-4)
