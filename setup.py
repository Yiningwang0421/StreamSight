"""Build the custom CUDA extension when an NVIDIA GPU is visible to PyTorch.

CPU development does not need this file. On macOS, ``python setup.py build_ext``
exits with an explanation instead of pretending the extension compiled.

    python setup.py build_ext --inplace
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed; cannot build CUDA extensions.", file=sys.stderr)
        return False
    return bool(torch.cuda.is_available())


def _extension():
    if not _cuda_available():
        if "build_ext" in sys.argv:
            raise SystemExit(
                "CUDA extension build requested, but torch.cuda.is_available() is False.\n"
                "CPU mode does not need this extension: run pytest and train.py directly.\n"
                "Build on Linux with an NVIDIA GPU, a matching CUDA toolkit, and a CUDA build of PyTorch:\n"
                "    python setup.py build_ext --inplace\n"
                "This project will not substitute PyTorch kernels for the CUDA backend."
            )
        return [], {}
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    extension = CUDAExtension(
        name="streamsight_cuda_kernels",
        sources=[
            "cuda/bindings.cpp",
            "cuda/decode_attention.cu",
            "cuda/implicit_gemm_conv.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # No --use_fast_math: the decode kernel is checked against FP32
            # reference attention, and fast-math exp would widen that gap.
            "nvcc": ["-O3", "-std=c++17"],
        },
    )
    return [extension], {"build_ext": BuildExtension}


if __name__ == "__main__":
    extensions, cmdclass = _extension()
    setup(
        name="streamsight-cuda",
        version="0.1.0",
        description="Streaming video inference experiments: FlashAttention-2, KV cache, implicit GEMM.",
        python_requires=">=3.10",
        packages=["data", "models", "inference", "benchmarks"],
        ext_modules=extensions,
        cmdclass=cmdclass,
        install_requires=[
            "numpy>=1.24",
            "torch>=2.2",
            "matplotlib>=3.7",
        ],
    )
