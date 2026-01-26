# HIP Kernel Quick Guide – AMD MI300X (gfx942)

## Hardware Snapshot
- gfx942 (CDNA3), wavefront 64, 304 CUs (4 SIMDs/CU), 32 SEs
- Caches/LDS: L1 32KB (128B line), L2 4MB, L3 256MB, LDS 64KB/CU
- Limits: ≤1024 threads/block, ≤32 waves/CU; XNACK off, SRAMECC on; HBM ~192GB

## Minimal Structure
- Block size multiple of 64 (start 256; adjust 128–512 for regs/LDS).
- Wrapper keeps device pointers only; no `hipMalloc`/`hipFree` inside.
- PyTorch extension entry: `PYBIND11_MODULE(...){ m.def("run", &run_kernel); }`
- Bound-check early; sync + error check: `hipDeviceSynchronize()` and `hipGetLastError()`.

## Fast Patterns
- Coalesced/vectorized access (float4/double2); align to 128B; avoid large strides.
- LDS with padding to dodge bank conflicts: `__shared__ float tile[T][T+1];`
- Control divergence: prefer predication over branchy per-thread logic.
- Occupancy balance: watch VGPR/LDS; `__launch_bounds__` only if launch matches exactly.
- Reductions: block-level sum with LDS + `__syncthreads`; use wave shuffles when possible.

## Profiling & Build
- Build: `hipcc -O3 -march=gfx942` (or `--offload-arch=gfx9-4-generic` for fallback).
- Profile: `rocprof --stats ./kernel`; inspect occupancy vs VGPR/LDS usage.
