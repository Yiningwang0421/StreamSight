"""Convolution as GEMM, with and without a materialized im2col matrix.

For 3x3 stride-1 padding-1 convolution the GEMM problem is

    M = N * H * W
    N = Cout
    K = Cin * 3 * 3
    C[m, n] = sum_k A_virtual[m, k] * W[k, n]

``m`` linearizes (batch, output_y, output_x). ``k`` linearizes
(cin, kh, kw) in that order, which matches ``F.unfold`` and
``weight.view(Cout, -1)``. The spatial address is PyTorch cross-correlation

    ih = oh - pad + kh
    iw = ow - pad + kw

with pad = 1. CUTLASS's analytic reference flips the filter index because its
convolution primitive is defined with that flip. This file uses the unflipped
map so the result matches ``F.conv2d`` (bias-free). The tiling structure is
the implicit-GEMM idea: load a K-tile of virtual A on demand, load the matching
filter tile, and accumulate. It is not a CUTLASS template, not a Tensor Core
MMA pipeline, and it does not claim that cuDNN selects implicit GEMM.

``implicit_gemm_conv2d_reference`` keeps only an ``[M, tile_k]`` tile.
``explicit_im2col_conv2d`` materializes the full ``[N, K, H*W]`` unfold buffer.
``scalar_implicit_gemm`` is the address formula transcribed into the CUDA kernel.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _check_conv_args(x: torch.Tensor, weight: torch.Tensor) -> Tuple[int, int, int, int, int]:
    if x.dim() != 4 or weight.dim() != 4:
        raise ValueError("expected NCHW activations and OIHW weights")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("convolution mapping references are FP32")
    n, cin, height, width = x.shape
    cout, cin_w, kh, kw = weight.shape
    if (kh, kw) != (3, 3):
        raise ValueError("only 3x3 kernels are supported")
    if cin != cin_w:
        raise ValueError(f"channel mismatch: x has {cin}, weight has {cin_w}")
    if min(n, cin, cout, height, width) <= 0:
        raise ValueError("empty convolution is not supported")
    return n, cin, height, width, cout


def gemm_extents(n: int, cin: int, height: int, width: int, cout: int) -> Dict[str, int]:
    return {
        "gemm_m": n * height * width,
        "gemm_n": cout,
        "gemm_k": cin * 9,
        "full_im2col_elements": n * height * width * cin * 9,
    }


def virtual_a_element(x_nhwc: torch.Tensor, m: int, k: int) -> float:
    """One entry of the virtual im2col matrix. Out-of-bounds reads are 0.

    This is the address map implemented by ``load_virtual_a`` in
    ``cuda/implicit_gemm_conv.cu``. Keep the two copies in sync.
    """
    batch_n, height, width, cin = x_nhwc.shape
    spatial = height * width
    kdim = cin * 9
    if m < 0 or k < 0 or m >= batch_n * spatial or k >= kdim:
        return 0.0
    batch = m // spatial
    rem = m % spatial
    oh = rem // width
    ow = rem % width
    ic = k // 9
    ks = k % 9
    kh = ks // 3
    kw = ks % 3
    ih = oh - 1 + kh
    iw = ow - 1 + kw
    if ih < 0 or iw < 0 or ih >= height or iw >= width:
        return 0.0
    return float(x_nhwc[batch, ih, iw, ic])


def weight_to_kn(weight: torch.Tensor) -> torch.Tensor:
    """OIHW 3x3 weight -> row-major [K, Cout] with K = Cin*9."""
    cout, cin, kh, kw = weight.shape
    return weight.permute(1, 2, 3, 0).contiguous().view(cin * kh * kw, cout)


def gather_virtual_a_tile(x_nhwc: torch.Tensor, k_start: int, tile_k: int) -> torch.Tensor:
    """On-demand [M, tile_k] tile. Does not build the full [M, K] matrix."""
    batch_n, height, width, cin = x_nhwc.shape
    spatial = height * width
    m_count = batch_n * spatial
    kdim = cin * 9
    device = x_nhwc.device
    m_index = torch.arange(m_count, device=device)
    batch = torch.div(m_index, spatial, rounding_mode="floor")
    rem = m_index % spatial
    oh = torch.div(rem, width, rounding_mode="floor")
    ow = rem % width
    ks = torch.arange(k_start, k_start + tile_k, device=device)
    ic = torch.div(ks, 9, rounding_mode="floor")
    spat = ks % 9
    kh = torch.div(spat, 3, rounding_mode="floor")
    kw = spat % 3
    ih = oh[:, None] - 1 + kh[None, :]
    iw = ow[:, None] - 1 + kw[None, :]
    ic_grid = ic[None, :].expand(m_count, tile_k)
    batch_grid = batch[:, None].expand(m_count, tile_k)
    valid = (
        (ih >= 0)
        & (iw >= 0)
        & (ih < height)
        & (iw < width)
        & (ic_grid >= 0)
        & (ic_grid < cin)
        & (ks[None, :] < kdim)
    )
    gathered = x_nhwc[
        batch_grid,
        ih.clamp(0, height - 1),
        iw.clamp(0, width - 1),
        ic_grid.clamp(0, cin - 1),
    ]
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def implicit_gemm_conv2d_reference(
    x: torch.Tensor, weight: torch.Tensor, tile_k: int = 8
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Tiled implicit GEMM. The largest virtual-A buffer is [M, tile_k]."""
    n, cin, height, width, cout = _check_conv_args(x, weight)
    if tile_k <= 0:
        raise ValueError("tile_k must be positive")
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    w_kn = weight_to_kn(weight)
    extents = gemm_extents(n, cin, height, width, cout)
    acc = torch.zeros(extents["gemm_m"], cout, dtype=torch.float32, device=x.device)
    max_tile_elements = 0
    for k0 in range(0, extents["gemm_k"], tile_k):
        bk = min(tile_k, extents["gemm_k"] - k0)
        a_tile = gather_virtual_a_tile(x_nhwc, k0, bk)
        max_tile_elements = max(max_tile_elements, int(a_tile.numel()))
        acc += a_tile.matmul(w_kn[k0 : k0 + bk])
    y = acc.view(n, height, width, cout).permute(0, 3, 1, 2).contiguous()
    stats = dict(extents)
    stats["max_tile_elements"] = max_tile_elements
    stats["tile_k"] = tile_k
    return y, stats


def explicit_im2col_conv2d(x: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Materialize im2col with ``F.unfold``, then GEMM. The unfold buffer is the extra traffic."""
    n, cin, height, width, cout = _check_conv_args(x, weight)
    columns = F.unfold(x, kernel_size=3, padding=1, stride=1)
    flattened = weight.view(cout, -1)
    produced = flattened.matmul(columns)
    y = produced.view(n, cout, height, width)
    stats = gemm_extents(n, cin, height, width, cout)
    stats["im2col_elements"] = int(columns.numel())
    return y, stats


def scalar_implicit_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Direct accumulation through ``virtual_a_element``. Tiny tensors only."""
    n, cin, height, width, cout = _check_conv_args(x, weight)
    extents = gemm_extents(n, cin, height, width, cout)
    if extents["gemm_m"] * extents["gemm_n"] * extents["gemm_k"] > 250_000:
        raise RuntimeError("scalar_implicit_gemm is only for tiny correctness checks")
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    weight = weight.contiguous()
    y = torch.empty(n, cout, height, width, dtype=torch.float32, device=x.device)
    for batch in range(n):
        for oh in range(height):
            for ow in range(width):
                m = (batch * height + oh) * width + ow
                for oc in range(cout):
                    acc = 0.0
                    for ic in range(cin):
                        for kh in range(3):
                            for kw in range(3):
                                k = ic * 9 + kh * 3 + kw
                                acc += virtual_a_element(x_nhwc, m, k) * float(weight[oc, ic, kh, kw])
                    y[batch, oc, oh, ow] = acc
    return y
