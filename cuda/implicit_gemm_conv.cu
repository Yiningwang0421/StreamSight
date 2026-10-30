// Educational implicit-GEMM convolution.
//
// CUTLASS maps forward convolution to GEMM without writing the im2col matrix:
// a tile iterator forms A in shared memory by computing activation coordinates
// as the tile is loaded (see the CUTLASS implicit GEMM convolution guide).
// This kernel demonstrates that mapping for one fixed workload:
//
//   3x3, stride 1, padding 1, FP32, NHWC activations, no groups, no bias.
//
//   M = N * H * W
//   N = Cout
//   K = Cin * 9
//   C[m, n] = A_virtual[m, k] * W[k, n]
//
// A_virtual is never allocated in global memory. Each thread block stages an
// [BM, BK] activation tile and a [BK, BN] filter tile in shared memory, then
// accumulates the outer product in registers.
//
// The address map matches PyTorch conv2d (cross-correlation):
//
//   ih = oh - 1 + kh
//   iw = ow - 1 + kw
//   k  = cin * 9 + kh * 3 + kw
//
// CUTLASS's published analytic f(p, r) flips the filter index because that
// reference defines convolution with the flip. Using the flip here would not
// match F.conv2d, so this kernel does not flip. It also does not use Tensor
// Cores, CUTLASS templates, or async copy pipelines. Activation loads are
// gathers; filter loads along N are contiguous. cuDNN may or may not pick
// implicit GEMM for a given shape, so the PyTorch conv baseline is "whatever
// algorithm the library selects," not a claim that cuDNN is this kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 8;

__device__ __forceinline__ float load_virtual_a(
    const float* __restrict__ input,
    int batch_n,
    int height,
    int width,
    int cin,
    int m,
    int k) {
    // Keep this coordinate arithmetic aligned with virtual_a_element()
    // in models/conv_mapping.py.
    const int spatial = height * width;
    const int kdim = cin * 9;
    if (m < 0 || k < 0 || m >= batch_n * spatial || k >= kdim) {
        return 0.f;
    }
    const int batch = m / spatial;
    const int rem = m - batch * spatial;
    const int oh = rem / width;
    const int ow = rem - oh * width;
    const int ic = k / 9;
    const int ks = k - ic * 9;
    const int kh = ks / 3;
    const int kw = ks - kh * 3;
    const int ih = oh - 1 + kh;
    const int iw = ow - 1 + kw;
    if (ih < 0 || iw < 0 || ih >= height || iw >= width) {
        return 0.f;
    }
    const long long index =
        ((static_cast<long long>(batch) * height + ih) * width + iw) * cin + ic;
    return input[index];
}

__global__ void implicit_gemm_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_n,
    int height,
    int width,
    int cin,
    int cout) {
    const int kdim = cin * 9;
    const int gemm_m = batch_n * height * width;
    const int m0 = static_cast<int>(blockIdx.y) * BM;
    const int n0 = static_cast<int>(blockIdx.x) * BN;
    const int tx = static_cast<int>(threadIdx.x);  // column inside the BM x BN tile
    const int ty = static_cast<int>(threadIdx.y);  // row inside the tile
    const int tid = ty * BN + tx;

    // +1 padding reduces shared-memory bank conflicts on the reduction axis.
    __shared__ float a_tile[BM][BK + 1];
    __shared__ float b_tile[BK][BN + 1];

    float acc = 0.f;
    for (int k0 = 0; k0 < kdim; k0 += BK) {
        if (tid < BM * BK) {
            const int lm = tid / BK;
            const int lk = tid - lm * BK;
            const int gm = m0 + lm;
            const int gk = k0 + lk;
            a_tile[lm][lk] = load_virtual_a(input, batch_n, height, width, cin, gm, gk);
        }
        if (tid < BK * BN) {
            const int lk = tid / BN;
            const int ln = tid - lk * BN;
            const int gk = k0 + lk;
            const int gn = n0 + ln;
            float value = 0.f;
            if (gk < kdim && gn < cout) {
                value = weight[static_cast<long long>(gk) * cout + gn];
            }
            b_tile[lk][ln] = value;
        }
        __syncthreads();

        const int gm = m0 + ty;
        const int gn = n0 + tx;
        if (gm < gemm_m && gn < cout) {
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                acc += a_tile[ty][kk] * b_tile[kk][tx];
            }
        }
        __syncthreads();
    }

    const int gm = m0 + ty;
    const int gn = n0 + tx;
    if (gm < gemm_m && gn < cout) {
        output[static_cast<long long>(gm) * cout + gn] = acc;
    }
}

}  // namespace

torch::Tensor implicit_gemm_conv_cuda(torch::Tensor input_nhwc, torch::Tensor weight_kn) {
    TORCH_CHECK(input_nhwc.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight_kn.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input_nhwc.is_contiguous(), "input must be contiguous NHWC");
    TORCH_CHECK(weight_kn.is_contiguous(), "weight must be contiguous [K, Cout]");
    TORCH_CHECK(input_nhwc.scalar_type() == torch::kFloat32, "implicit GEMM is FP32 only");
    TORCH_CHECK(weight_kn.scalar_type() == torch::kFloat32, "weight must be FP32");
    TORCH_CHECK(input_nhwc.dim() == 4, "input must be NHWC");
    TORCH_CHECK(weight_kn.dim() == 2, "weight must be [Cin*9, Cout]");

    const int batch_n = static_cast<int>(input_nhwc.size(0));
    const int height = static_cast<int>(input_nhwc.size(1));
    const int width = static_cast<int>(input_nhwc.size(2));
    const int cin = static_cast<int>(input_nhwc.size(3));
    const int kdim = static_cast<int>(weight_kn.size(0));
    const int cout = static_cast<int>(weight_kn.size(1));
    TORCH_CHECK(cin > 0 && height > 0 && width > 0 && cout > 0, "empty convolution");
    TORCH_CHECK(kdim == cin * 9, "weight K dimension must be Cin * 9 for a 3x3 kernel");

    const int gemm_m = batch_n * height * width;
    const int grid_m = (gemm_m + BM - 1) / BM;
    const int grid_n = (cout + BN - 1) / BN;
    TORCH_CHECK(grid_m <= 65535 && grid_n <= 65535, "implicit GEMM tile grid exceeds launch limits");

    c10::cuda::CUDAGuard device_guard(input_nhwc.device());
    auto output = torch::empty({batch_n, height, width, cout}, input_nhwc.options());
    dim3 block(BN, BM);
    dim3 grid(grid_n, grid_m);
    implicit_gemm_conv_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_nhwc.data_ptr<float>(),
        weight_kn.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_n,
        height,
        width,
        cin,
        cout);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
