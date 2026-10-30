#include <torch/extension.h>

// Defined in the .cu translation units. The Python names are the project
// kernels; neither one dispatches to cuDNN or to a PyTorch fallback.
torch::Tensor decode_attention_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t valid_length);

torch::Tensor implicit_gemm_conv_cuda(torch::Tensor input_nhwc, torch::Tensor weight_kn);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "StreamSight custom CUDA kernels (decode attention and implicit GEMM convolution).";
    module.def(
        "decode_attention",
        &decode_attention_cuda,
        "FP32 single-query online-softmax attention over a KV cache. Not FlashAttention-2.");
    module.def(
        "implicit_gemm_conv",
        &implicit_gemm_conv_cuda,
        "FP32 3x3 implicit-GEMM convolution. Does not allocate an im2col matrix.");
}
