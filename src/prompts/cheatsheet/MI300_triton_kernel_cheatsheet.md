# Triton Kernel Quick Guide for MI300X (gfx942)

- **Architecture**: gfx942 (CDNA3) | **Wavefront**: 64 | **CUs**: 304 | **SIMDs/CU**: 4
- **Caches/LDS**: L1 32KB/CU (128B line), L2 4MB, L3 256MB, LDS 64KB/CU (GROUP)
- **Limits**: ≤1024 threads/block, ≤32 waves/CU, XNACK off, SRAMECC on
- **HBM**: ~192GB; align large buffers to 2MB when possible

## Launch / Tiling
- Choose `num_warps` so `num_warps*32` is a multiple of 64; start with `num_warps=4`, `num_stages=2`.
- Keep blocks balanced: aim for 256–512 threads; watch register/LDS pressure vs occupancy.
- Tile to fit L1/LDS; avoid oversized `tl.shared_memory` that drops active waves.

## Memory
- Use contiguous, aligned loads/stores; add `tl.multiple_of` and `tl.assume` to help vectorization.
- Guard tails: `tl.load(..., mask=mask, other=0)`; avoid branch-heavy per-element conditionals.
- Prefer vector shapes (e.g., `BLOCK_K` multiples of 16/32) for wide memory ops.

## Math & Numerics
- Use `tl.dot`/`tl.matmul` with `BLOCK_M/N/K` autotuning; benchmark a small config grid.
- Accumulate in fp32 for fp16/bf16 inputs; cast once on write-back.
- Fuse simple epilogues in-kernel (bias/activation) to save bandwidth.

## Profiling
- Build with target `gfx942`; run with `TRITON_DEBUG=0`.
- Check occupancy/resource use with `TRITON_PRINT_AUTOTUNING=1`; adjust `num_stages/warps` if stalled.