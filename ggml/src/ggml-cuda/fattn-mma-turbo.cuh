// Turbo MMA flash-attention launcher (stub).
//
// The reference implementation dispatches turbo KV decode through the VEC kernel
// (fattn-vec.cuh) and handles prefill by converting K/V to f16 (need_f16_K/V in
// launch_fattn). A fused MMA path with in-kernel turbo dequant would require
// flash_attn_ext_f16 (fattn-mma-f16.cuh) to be templated on type_K/type_V, which
// this upstream vintage does not support. The launcher is kept as a stub so the
// pre-declared template instances compile; fattn.cu never selects this path.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("turbo MMA flash-attention path is not implemented on this branch; use the VEC kernel");
}

#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                           \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

#define DECL_FATTN_MMA_TURBO_ALL(DKQ, DV, tK, tV)        \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 1, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 2, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 2, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 4, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 8, 1, tK, tV); \

DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
