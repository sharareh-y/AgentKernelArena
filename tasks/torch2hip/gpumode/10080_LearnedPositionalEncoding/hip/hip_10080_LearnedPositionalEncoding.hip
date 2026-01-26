// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <cstdint>
#include <type_traits>

// Scalar fallback kernel: processes elementwise add with broadcasting over [B, H]
template <typename scalar_t>
__global__ void fused_add_positional_encoding_scalar_kernel(
    const scalar_t* __restrict__ x,          // [B, H, S, E]
    const scalar_t* __restrict__ pos_embed,   // [1, S, E]
    scalar_t* __restrict__ out,               // [B, H, S, E]
    int64_t total_elems,                      // B*H*S*E
    int64_t E,                                // embed_dim
    int64_t S                                 // seq_len
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    // Layout: [B, H, S, E] contiguous
    int64_t tmp = idx;
    int64_t d3 = tmp % E;    // embed_idx
    tmp /= E;
    int64_t d2 = tmp % S;    // seq_idx
    // d1,d0 (H,B) are broadcasted

    int64_t pos_idx = d2 * E + d3; // [S, E]
    out[idx] = x[idx] + pos_embed[pos_idx];
}

// Vectorized kernel for float32 using float4 along the last dimension (E)
__global__ void fused_add_positional_encoding_vec4_kernel(
    const float* __restrict__ x,              // [B, H, S, E]
    const float* __restrict__ pos_embed,      // [1, S, E]
    float* __restrict__ out,                  // [B, H, S, E]
    int64_t total_vec4,                       // B*H*S*(E/4)
    int64_t S,                                // seq_len
    int64_t pack_E                            // E/4
) {
    int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx4 >= total_vec4) return;

    // Reinterpret as float4 streams aligned on the last dim
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    const float4* __restrict__ pe4 = reinterpret_cast<const float4*>(pos_embed);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out);

    // Index decomposition over [B, H, S, E/4]
    int64_t tmp = idx4;
    int64_t d3v = tmp % pack_E;  // vector index along E/4
    tmp /= pack_E;
    int64_t d2 = tmp % S;        // seq_idx
    // d1,d0 are broadcasted

    // Position embeddings index in float4 units: [S, E/4]
    int64_t pe_idx4 = d2 * pack_E + d3v;

    // Load, add, store
    float4 xv = x4[idx4];
    float4 pv = pe4[pe_idx4];
    float4 ov;
    ov.x = xv.x + pv.x;
    ov.y = xv.y + pv.y;
    ov.z = xv.z + pv.z;
    ov.w = xv.w + pv.w;
    out4[idx4] = ov;
}

// Kernel launcher: forward(x, position_embeddings)
torch::Tensor forward(
    torch::Tensor x,                // [B, H, S, E]
    torch::Tensor position_embeddings // [1, S, E]
) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA/HIP device (ROCm)");
    TORCH_CHECK(position_embeddings.is_cuda(), "position_embeddings must be on CUDA/HIP device (ROCm)");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(position_embeddings.is_contiguous(), "position_embeddings must be contiguous");
    TORCH_CHECK(x.scalar_type() == position_embeddings.scalar_type(), "Dtype mismatch between x and position_embeddings");
    TORCH_CHECK(x.dim() == 4, "x must be 4D: [B, H, S, E]");
    TORCH_CHECK(position_embeddings.dim() == 3, "position_embeddings must be 3D: [1, S, E]");
    TORCH_CHECK(position_embeddings.size(0) == 1, "position_embeddings first dim must be 1 for broadcasting");

    const int64_t B = x.size(0);
    const int64_t H = x.size(1);
    const int64_t S = x.size(2);
    const int64_t E = x.size(3);

    TORCH_CHECK(position_embeddings.size(1) == S && position_embeddings.size(2) == E,
                "position_embeddings shape must be [1, S, E] matching x's last two dims");

    auto out = torch::empty_like(x);

    const int threads = 256;
    hipStream_t stream = at::hip::getCurrentHIPStream();

    // Vectorized path only for float32 when E % 4 == 0 and pointers are 16-byte aligned
    bool use_vec4 = false;
    if (x.scalar_type() == at::kFloat && (E % 4 == 0)) {
        uintptr_t xp = reinterpret_cast<uintptr_t>(x.data_ptr());
        uintptr_t pp = reinterpret_cast<uintptr_t>(position_embeddings.data_ptr());
        uintptr_t op = reinterpret_cast<uintptr_t>(out.data_ptr());
        if ((xp % 16u == 0u) && (pp % 16u == 0u) && (op % 16u == 0u)) {
            use_vec4 = true;
        }
    }

    if (use_vec4) {
        const int64_t pack_E = E / 4; // number of float4 along last dim
        const int64_t total_vec4 = B * H * S * pack_E;
        const int blocks = static_cast<int>((total_vec4 + threads - 1) / threads);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fused_add_positional_encoding_vec4_kernel),
            dim3(blocks), dim3(threads), 0, stream,
            x.data_ptr<float>(),
            position_embeddings.data_ptr<float>(),
            out.data_ptr<float>(),
            total_vec4,
            S,
            pack_E
        );
    } else {
        const int64_t total_elems = x.numel();
        const int blocks = static_cast<int>((total_elems + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_add_positional_encoding_scalar_kernel", [&] {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(fused_add_positional_encoding_scalar_kernel<scalar_t>),
                dim3(blocks), dim3(threads), 0, stream,
                x.data_ptr<scalar_t>(),
                position_embeddings.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                total_elems,
                E,
                S
            );
        });
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LearnedPositionalEncoding fused add forward (HIP)");
}

