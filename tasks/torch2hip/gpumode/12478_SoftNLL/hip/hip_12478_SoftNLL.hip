// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdint>
#include <cmath>

// Wavefront (warp) reduce using HIP shuffle: assumes warpSize==64 on AMD
__device__ inline float warp_reduce_sum(float val) {
    // Use full mask style intrinsic compatible with HIP
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Fused accumulation kernel: computes total_sum = sum(log(input[i]) * target[i]) over all elements.
// Uses vectorized float4 path when aligned, then handles tail scalars.
// In-block reduction: wavefront-level shuffles + shared memory for warp leaders + final shuffle.
__global__ void fused_softnll_accumulate_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ global_sum,
    int64_t numel,
    int vec4_enabled)
{
    extern __shared__ float warp_partials[]; // one float per warp
    const int tid = threadIdx.x;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int num_warps = blockDim.x / warpSize;

    float local_sum = 0.0f;

    if (vec4_enabled) {
        const int64_t numel4 = numel >> 2; // floor(numel/4)
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        const float4* __restrict__ tg4 = reinterpret_cast<const float4*>(target);

        // Vectorized grid-stride loop over float4 chunks
        for (int64_t idx4 = (int64_t)blockIdx.x * blockDim.x + tid; idx4 < numel4; idx4 += (int64_t)blockDim.x * gridDim.x) {
            float4 xv = in4[idx4];
            float4 tv = tg4[idx4];
            local_sum += logf(xv.x) * tv.x;
            local_sum += logf(xv.y) * tv.y;
            local_sum += logf(xv.z) * tv.z;
            local_sum += logf(xv.w) * tv.w;
        }
        // Scalar tail for remaining elements
        int64_t tail_start = numel4 << 2;
        for (int64_t i = tail_start + (int64_t)blockIdx.x * blockDim.x + tid; i < numel; i += (int64_t)blockDim.x * gridDim.x) {
            float x = input[i];
            float t = target[i];
            local_sum += logf(x) * t;
        }
    } else {
        // Scalar-only grid-stride loop
        for (int64_t i = (int64_t)blockIdx.x * blockDim.x + tid; i < numel; i += (int64_t)blockDim.x * gridDim.x) {
            float x = input[i];
            float t = target[i];
            local_sum += logf(x) * t;
        }
    }

    // Intra-warp reduction
    float warp_sum = warp_reduce_sum(local_sum);

    // Write warp leaders to shared memory
    if (lane == 0) {
        warp_partials[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final reduction among warp leaders using first warp
    float block_sum = 0.0f;
    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_partials[lane] : 0.0f;
        block_sum = warp_reduce_sum(val);
        if (lane == 0) {
            atomicAdd(global_sum, block_sum);
        }
    }
}

// Finalize kernel: writes loss = -global_sum / M into out[0]
__global__ void finalize_softnll_kernel(const float* __restrict__ global_sum, float* __restrict__ out, float denom_M) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total = *global_sum;
        out[0] = - total / denom_M;
    }
}

// Forward function: computes -mean(sum(log(input) * target, dim=1)) for tensors with rank >= 2
at::Tensor forward(at::Tensor input, at::Tensor target) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on HIP/CUDA device (ROCm)");
    TORCH_CHECK(target.device() == input.device(), "Target must be on the same device as input");
    TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have the same shape");
    TORCH_CHECK(input.dim() >= 2, "Expected tensor with rank >= 2 (..., num_classes)");

    // Make contiguous and cast to float32 for predictable layout and math
    at::Tensor input_f = input.contiguous().to(torch::kFloat);
    at::Tensor target_f = target.contiguous().to(torch::kFloat);

    // Compute M = product of all dims except dim=1
    const int64_t ndim = input_f.dim();
    int64_t M = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != 1) M *= input_f.size(d);
    }

    const int64_t numel = input_f.numel();

    auto opts = torch::TensorOptions().device(input_f.device()).dtype(torch::kFloat);
    at::Tensor global_sum_tensor = torch::zeros({1}, opts);
    at::Tensor out = torch::empty({1}, opts);

    // Choose block size (multiples of 64 for AMD wavefronts)
    int blockSize;
    if (numel < 8192) blockSize = 128;
    else if (numel < (1LL << 22)) blockSize = 256;
    else blockSize = 512;

    // Shared memory: one float per warp
    int numWarps = blockSize / warpSize;
    size_t smem = numWarps * sizeof(float);

    // Occupancy-aware grid sizing
    int device = input_f.get_device();
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    int activeBlocksPerCU = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerCU, fused_softnll_accumulate_kernel, blockSize, smem);
    int numCUs = prop.multiProcessorCount;
    int maxGrid = 65535;
    int targetBlocks = activeBlocksPerCU * numCUs * 2; // modest oversubscription
    if (targetBlocks < 1) targetBlocks = numCUs * 2;
    int gridSize = targetBlocks;
    if (gridSize > maxGrid) gridSize = maxGrid;
    if (gridSize < 1) gridSize = 1;

    // Vectorization check: 16-byte aligned pointers enable float4 path
    const void* in_ptr_v = input_f.data_ptr<float>();
    const void* tg_ptr_v = target_f.data_ptr<float>();
    bool aligned16 = (((uintptr_t)in_ptr_v | (uintptr_t)tg_ptr_v) & 0xF) == 0;
    int vec4_enabled = aligned16 ? 1 : 0;

    hipLaunchKernelGGL(
        fused_softnll_accumulate_kernel,
        dim3(gridSize), dim3(blockSize), smem, 0,
        input_f.data_ptr<float>(),
        target_f.data_ptr<float>(),
        global_sum_tensor.data_ptr<float>(),
        numel,
        vec4_enabled);
    hipError_t err1 = hipGetLastError();
    TORCH_CHECK(err1 == hipSuccess, "HIP kernel launch failed (accumulate): ", hipGetErrorString(err1));

    float denom = static_cast<float>(M);
    hipLaunchKernelGGL(
        finalize_softnll_kernel,
        dim3(1), dim3(1), 0, 0,
        global_sum_tensor.data_ptr<float>(),
        out.data_ptr<float>(),
        denom);
    hipError_t err2 = hipGetLastError();
    TORCH_CHECK(err2 == hipSuccess, "HIP kernel launch failed (finalize): ", hipGetErrorString(err2));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SoftNLL forward (fused reduction with wavefront shuffles, ROCm/HIP)");
}

