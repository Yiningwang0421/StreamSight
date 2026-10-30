"""Loaders for the custom CUDA extension.

Importing this module does not require CUDA. Calling a kernel on a machine
without the extension raises ``CudaBackendUnavailable`` and does not run a
PyTorch implementation under the CUDA name.
"""

from __future__ import annotations

import torch


class CudaBackendUnavailable(RuntimeError):
    """The custom CUDA extension is not built, or the tensor is not on CUDA."""


_EXTENSION = None
_EXTENSION_TRIED = False
_EXTENSION_ERROR: Exception | None = None


def _import_extension():
    global _EXTENSION, _EXTENSION_TRIED, _EXTENSION_ERROR
    if _EXTENSION_TRIED:
        return _EXTENSION
    _EXTENSION_TRIED = True
    try:
        import streamsight_cuda_kernels as extension
    except ImportError as exc:
        _EXTENSION_ERROR = exc
        _EXTENSION = None
    else:
        _EXTENSION = extension
    return _EXTENSION


def kernels_available() -> bool:
    return _import_extension() is not None and torch.cuda.is_available()


def unavailable_reason() -> str:
    if _import_extension() is None:
        detail = f" Import error: {_EXTENSION_ERROR}." if _EXTENSION_ERROR else ""
        return (
            "Custom CUDA kernels are not built (module streamsight_cuda_kernels)."
            + detail
            + " Build on Linux with an NVIDIA GPU and the CUDA toolkit: "
            "python setup.py build_ext --inplace. "
            "CPU mode does not need this extension. "
            "The CUDA backend does not fall back to PyTorch."
        )
    if not torch.cuda.is_available():
        return (
            "streamsight_cuda_kernels imported, but torch.cuda.is_available() is False. "
            "Refusing to run the CUDA backend on another device."
        )
    return ""


def load_kernels():
    extension = _import_extension()
    if extension is None or not torch.cuda.is_available():
        raise CudaBackendUnavailable(unavailable_reason())
    return extension


def _check_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, valid_length: int) -> None:
    if q.device.type != "cuda" or k_cache.device.type != "cuda" or v_cache.device.type != "cuda":
        raise CudaBackendUnavailable(
            "Custom CUDA decode attention was given a non-CUDA tensor. "
            "Refusing to fall back to PyTorch."
        )
    if q.dtype != torch.float32 or k_cache.dtype != torch.float32 or v_cache.dtype != torch.float32:
        raise RuntimeError(
            "Custom CUDA decode attention is an FP32 kernel. "
            f"Got q={q.dtype}, k={k_cache.dtype}, v={v_cache.dtype}. "
            "Refusing to cast or to call a PyTorch kernel."
        )
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError(
            f"expected q [B,H,D] and k/v [B,H,Cap,D], got {tuple(q.shape)} {tuple(k_cache.shape)}"
        )
    if q.shape[-1] != 32 or k_cache.shape[-1] != 32:
        raise RuntimeError(
            "Custom CUDA decode attention is specialized to head dimension 32 "
            f"(one element per warp lane). Got D={q.shape[-1]}."
        )
    if valid_length < 1 or valid_length > k_cache.shape[2]:
        raise ValueError(
            f"valid_length {valid_length} is outside 1..capacity {k_cache.shape[2]}"
        )


def _launch_decode(q, k_cache, v_cache, valid_length: int) -> torch.Tensor:
    extension = load_kernels()
    _check_decode(q, k_cache, v_cache, valid_length)
    return extension.decode_attention(
        q.contiguous(), k_cache.contiguous(), v_cache.contiguous(), int(valid_length)
    )


class _DecodeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cache, v_cache, valid_length):
        return _launch_decode(q, k_cache, v_cache, int(valid_length))

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "Custom CUDA decode attention has no backward kernel. "
            "Train with the PyTorch attention backend."
        )


def decode_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, valid_length: int
) -> torch.Tensor:
    """Single-query attention over the valid KV prefix. Output is [B, H, D]."""
    needs_grad = any(tensor.requires_grad for tensor in (q, k_cache, v_cache))
    if needs_grad:
        return _DecodeAttention.apply(q, k_cache, v_cache, int(valid_length))
    return _launch_decode(q, k_cache, v_cache, int(valid_length))


def _check_conv(x_nhwc: torch.Tensor, weight_kn: torch.Tensor) -> None:
    if x_nhwc.device.type != "cuda" or weight_kn.device.type != "cuda":
        raise CudaBackendUnavailable(
            "Custom implicit GEMM convolution was given a non-CUDA tensor. "
            "Refusing to fall back to PyTorch or cuDNN."
        )
    if x_nhwc.dtype != torch.float32 or weight_kn.dtype != torch.float32:
        raise RuntimeError(
            "Custom implicit GEMM convolution is an FP32 kernel. "
            "Refusing to cast or to call cuDNN under this backend."
        )


def _launch_conv(x_nhwc: torch.Tensor, weight_kn: torch.Tensor) -> torch.Tensor:
    extension = load_kernels()
    _check_conv(x_nhwc, weight_kn)
    return extension.implicit_gemm_conv(x_nhwc.contiguous(), weight_kn.contiguous())


class _ImplicitGemmConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_nhwc, weight_kn):
        return _launch_conv(x_nhwc, weight_kn)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "Custom implicit GEMM convolution has no backward kernel. "
            "Train with PyTorch Conv2d."
        )


def implicit_gemm_conv_nhwc(x_nhwc: torch.Tensor, weight_kn: torch.Tensor) -> torch.Tensor:
    """NHWC activations times a [K, N] filter matrix. K = Cin * 9, N = Cout."""
    if x_nhwc.requires_grad or weight_kn.requires_grad:
        return _ImplicitGemmConv.apply(x_nhwc, weight_kn)
    return _launch_conv(x_nhwc, weight_kn)


def implicit_gemm_conv2d(x_nchw: torch.Tensor, weight_oihw: torch.Tensor) -> torch.Tensor:
    """3x3, stride 1, padding 1, bias-free convolution. x is NCHW, weight is OIHW.

    The extension consumes NHWC activations and a [Cin*9, Cout] filter matrix.
    No im2col buffer is allocated on either side of this wrapper.
    """
    if x_nchw.dim() != 4 or weight_oihw.dim() != 4:
        raise ValueError("expected NCHW activations and OIHW weights")
    if weight_oihw.shape[2:] != (3, 3):
        raise ValueError("educational implicit GEMM supports only 3x3 kernels")
    if x_nchw.shape[1] != weight_oihw.shape[1]:
        raise ValueError(
            f"input channels {x_nchw.shape[1]} != weight channels {weight_oihw.shape[1]}"
        )
    cout, cin, _, _ = weight_oihw.shape
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    weight_kn = weight_oihw.permute(1, 2, 3, 0).contiguous().view(cin * 9, cout)
    y_nhwc = implicit_gemm_conv_nhwc(x_nhwc, weight_kn)
    return y_nhwc.permute(0, 3, 1, 2).contiguous()
