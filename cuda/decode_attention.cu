// Streaming decode attention.
//
// This is a FlashAttention-inspired single-query kernel.
// This kernel is not FlashAttention-2 and it is not a prefill kernel.
//
// FlashAttention-2 assigns different blocks of query rows to different thread
// blocks, and splits those rows across warps so each warp owns a slice of Q
// while sharing the K/V tile (Dao, 2023, Sections 3.2 and 3.3). One-query
// decoding has a single query row per head, so that query-parallel partition
// does not exist. Available parallelism is batch x heads, plus any split of
// the KV sequence that later merges online-softmax state. This MVP uses one
// warp per (batch, head) and scans the KV cache in order.
//
// Numerics follow the FlashAttention-2 forward recurrence: an unnormalized
// output accumulator, a running max, and a running sum of exponentials. The
// accumulator is divided by the normalizer once, after the scan.
// No score vector is written to global memory.
//
// Constraints: FP32, head dimension 32 (one element per lane), one valid
// length for the whole batch, no dropout, no causal mask, no backward.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

namespace {

__global__ void decode_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int batch,
    int heads,
    int capacity,
    int head_dim,
    int valid_length,
    float scale) {
    const int bh = static_cast<int>(blockIdx.x);
    if (bh >= batch * heads) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);

    const int b = bh / heads;
    const int h = bh - b * heads;
    const long long head_offset = (static_cast<long long>(b) * heads + h);
    const float* q_ptr = q + head_offset * head_dim;
    const float* k_ptr = k_cache + head_offset * capacity * head_dim;
    const float* v_ptr = v_cache + head_offset * capacity * head_dim;
    float* o_ptr = out + head_offset * head_dim;

    const float q_lane = q_ptr[lane];
    float acc = 0.f;
    float running_max = -INFINITY;
    float normalizer = 0.f;

    for (int t = 0; t < valid_length; ++t) {
        const long long token = static_cast<long long>(t) * head_dim;
        // Lanes read consecutive channels: one 128-byte transaction when D = 32.
        float partial = q_lane * k_ptr[token + lane];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        const float score = partial * scale;

        const float new_max = fmaxf(running_max, score);
        // The first step has running_max = -inf. exp(-inf) is 0, which drops
        // the empty accumulator. The explicit test avoids a -inf - -inf NaN
        // if a score is ever non-finite.
        const float alpha = (running_max == -INFINITY) ? 0.f : expf(running_max - new_max);
        const float weight = expf(score - new_max);
        normalizer = normalizer * alpha + weight;
        acc = acc * alpha + weight * v_ptr[token + lane];
        running_max = new_max;
    }

    o_ptr[lane] = acc / normalizer;
}

}  // namespace

torch::Tensor decode_attention_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t valid_length) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "decode attention is FP32 only");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat32, "k_cache must be FP32");
    TORCH_CHECK(v_cache.scalar_type() == torch::kFloat32, "v_cache must be FP32");
    TORCH_CHECK(q.dim() == 3, "q must be [B, H, D]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [B, H, capacity, D]");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "k_cache and v_cache shapes must match");
    TORCH_CHECK(q.size(0) == k_cache.size(0) && q.size(1) == k_cache.size(1), "batch/head mismatch");
    TORCH_CHECK(q.size(2) == 32 && k_cache.size(3) == 32, "head dimension must be 32");
    TORCH_CHECK(valid_length >= 1 && valid_length <= k_cache.size(2), "valid_length out of range");

    c10::cuda::CUDAGuard device_guard(q.device());
    const int batch = static_cast<int>(q.size(0));
    const int heads = static_cast<int>(q.size(1));
    const int capacity = static_cast<int>(k_cache.size(2));
    const int head_dim = 32;
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

    auto out = torch::empty({batch, heads, head_dim}, q.options());
    decode_attention_kernel<<<batch * heads, head_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        heads,
        capacity,
        head_dim,
        static_cast<int>(valid_length),
        scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
