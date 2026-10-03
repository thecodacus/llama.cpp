#include "common.cuh"
#include "fattn-listed.cuh"

// Flash attention over a list of cells per query (ggml_flash_attn_ext_set_indices), as sparse-attention indexers give
// it. One block per (query, K/V head, sequence) holds every query head of the K/V head (up to 16, the rows of the MMA
// tile), so each listed K/V row is read once for all of them, in its cache type (q8_0 rows are 34 bytes per 32 values)
// and dequantized to f16 as it is staged. Per chunk of FATTN_LISTED_CH cells: gather K and V, scores with the f16 MMA
// (a warp per 8 cells), an online softmax in base 2, and P V (a warp per 64 dims).

#define FATTN_LISTED_D      256
#define FATTN_LISTED_CH     32
#define FATTN_LISTED_NT     128
#define FATTN_LISTED_STRIDE (FATTN_LISTED_D + 8) // halves per staged row: fragment loads spread over the banks

struct fattn_listed_smem {
    half  q[16][FATTN_LISTED_STRIDE];
    half  k[FATTN_LISTED_CH][FATTN_LISTED_STRIDE];
    half  v[FATTN_LISTED_CH][FATTN_LISTED_STRIDE];
    float s[16][FATTN_LISTED_CH + 1];
    float mrow[16];
    float lsum[16];
    float alpha[16];
    float mval[FATTN_LISTED_CH]; // the mask value of the chunk's cells, base 2; -inf for a hidden or empty slot
    int   cell[FATTN_LISTED_CH];
};

// D = A B + D, A 16x16 row major, B 16x8 column major, f16 inputs and f32 accumulators
static __device__ __forceinline__ void fattn_listed_mma(float * c, const uint32_t * a, const uint32_t * b) {
#if defined(GGML_USE_HIP) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < GGML_CUDA_CC_TURING
    GGML_UNUSED_VARS(c, a, b);
    NO_DEVICE_CODE;
#elif __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    // Turing has m16n8k8 with the same fragment layout: the k = 16 step is two k = 8 steps
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

static __device__ __forceinline__ uint32_t fattn_listed_pack(const float lo, const float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}

static __device__ __forceinline__ uint32_t fattn_listed_ld2(const half * p) {
    return *reinterpret_cast<const uint32_t *>(p);
}

// stages the K or V rows of the chunk's cells as f16; an empty slot gets zeros
template <ggml_type type>
static __device__ __forceinline__ void fattn_listed_stage(
        half (*dst)[FATTN_LISTED_STRIDE], const char * base, const int64_t stride_cell, const int * cell, const int tid) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        constexpr int nblk = FATTN_LISTED_D / QK8_0;
        for (int i = tid; i < FATTN_LISTED_CH*nblk; i += FATTN_LISTED_NT) {
            const int c = i / nblk;
            const int b = i % nblk;
            half2 * out = reinterpret_cast<half2 *>(&dst[c][b*QK8_0]);
            if (cell[c] < 0) {
#pragma unroll
                for (int j = 0; j < QK8_0/2; ++j) {
                    out[j] = make_half2(0.0f, 0.0f);
                }
                continue;
            }
            const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(base + cell[c]*stride_cell) + b;
            const float d = __half2float(blk->d);
            const uint16_t * qs = reinterpret_cast<const uint16_t *>(blk->qs); // block_q8_0 is 2-byte aligned
#pragma unroll
            for (int j = 0; j < QK8_0/2; ++j) {
                const uint16_t x = qs[j];
                out[j] = __floats2half2_rn(d*(float) (int8_t) (x & 0xFF), d*(float) (int8_t) (x >> 8));
            }
        }
    } else {
        static_assert(type == GGML_TYPE_F16, "f16 or q8_0 K/V");
        constexpr int npieces = FATTN_LISTED_D*sizeof(half)/sizeof(uint4);
        for (int i = tid; i < FATTN_LISTED_CH*npieces; i += FATTN_LISTED_NT) {
            const int c = i / npieces;
            const int p = i % npieces;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (cell[c] >= 0) {
                x = reinterpret_cast<const uint4 *>(base + cell[c]*stride_cell)[p];
            }
            reinterpret_cast<uint4 *>(&dst[c][0])[p] = x;
        }
    }
}

template <ggml_type type_KV>
__launch_bounds__(FATTN_LISTED_NT, 2)
static __global__ void flash_attn_ext_listed(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, const int32_t * __restrict__ cells, float * __restrict__ dst,
        const float scale_log2, const int n_list, const int n_kv, const int ne01, const int ne02, const int gqa,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int ne31, const int ne33, const int64_t nb31, const int64_t nb33, const int rk3) {
#if defined(GGML_USE_HIP) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < GGML_CUDA_CC_TURING
    GGML_UNUSED_VARS(Q, K, V, mask, cells, dst, scale_log2, n_list, n_kv, ne01, ne02, gqa, nb01, nb02, nb03,
        nb11, nb12, nb13, nb21, nb22, nb23, ne31, ne33, nb31, nb33, rk3);
    NO_DEVICE_CODE;
#else
    __shared__ fattn_listed_smem S;

    const int qi   = blockIdx.x;
    const int kvh  = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int gid  = lane / 4;
    const int tig  = lane % 4;

    const int     sm   = seq % ne33;
    const half  * mrow = reinterpret_cast<const half *>(mask + sm*nb33 + qi*nb31);
    const int32_t * list = cells + (int64_t(sm)*ne31 + qi)*n_list;

    const char * Kb = K + (seq/rk3)*nb13 + kvh*nb12;
    const char * Vb = V + (seq/rk3)*nb23 + kvh*nb22;

    // the query heads of the K/V head, scaled into base 2; rows past gqa stay zero
    for (int i = tid; i < 16*FATTN_LISTED_D; i += FATTN_LISTED_NT) {
        const int h = i / FATTN_LISTED_D;
        const int d = i % FATTN_LISTED_D;
        float x = 0.0f;
        if (h < gqa) {
            x = reinterpret_cast<const float *>(Q + seq*nb03 + qi*nb01 + (int64_t(kvh)*gqa + h)*nb02)[d] * scale_log2;
        }
        S.q[h][d] = __float2half_rn(x);
    }
    if (tid < 16) {
        S.mrow[tid] = -INFINITY;
        S.lsum[tid] = 0.0f;
    }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    }

    for (int c0 = 0; c0 < n_list; c0 += FATTN_LISTED_CH) {
        // the cells and their mask values (read only by this chunk's gather and scores, so no barrier before)
        if (tid < FATTN_LISTED_CH) {
            int   c = c0 + tid < n_list ? list[c0 + tid] : -1;
            float m = -INFINITY;
            if (c >= 0 && c < n_kv) {
                m = __half2float(mrow[c]) * 1.4426950408889634f;
            }
            if (m == -INFINITY) {
                c = -1;
            }
            S.cell[tid] = c;
            S.mval[tid] = m;
        }
        __syncthreads(); // also: the previous chunk's P V is done with k, v and s

        fattn_listed_stage<type_KV>(S.k, Kb, nb11, S.cell, tid);
        fattn_listed_stage<type_KV>(S.v, Vb, nb21, S.cell, tid);
        __syncthreads();

        // scores: warp w takes cells 8w..8w+7 over all dims
        {
            const int cb = warp*8;
            float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k0 = 0; k0 < FATTN_LISTED_D; k0 += 16) {
                uint32_t a[4];
                uint32_t b[2];
                a[0] = fattn_listed_ld2(&S.q[gid    ][k0 + 2*tig    ]);
                a[1] = fattn_listed_ld2(&S.q[gid + 8][k0 + 2*tig    ]);
                a[2] = fattn_listed_ld2(&S.q[gid    ][k0 + 2*tig + 8]);
                a[3] = fattn_listed_ld2(&S.q[gid + 8][k0 + 2*tig + 8]);
                b[0] = fattn_listed_ld2(&S.k[cb + gid][k0 + 2*tig    ]);
                b[1] = fattn_listed_ld2(&S.k[cb + gid][k0 + 2*tig + 8]);
                fattn_listed_mma(sc, a, b);
            }
            const int c = cb + 2*tig;
            S.s[gid    ][c    ] = sc[0] + S.mval[c    ];
            S.s[gid    ][c + 1] = sc[1] + S.mval[c + 1];
            S.s[gid + 8][c    ] = sc[2] + S.mval[c    ];
            S.s[gid + 8][c + 1] = sc[3] + S.mval[c + 1];
        }
        __syncthreads();

        // online softmax: 8 threads per row, 4 cells each
        {
            constexpr int per = FATTN_LISTED_CH/8;
            const int r   = tid / 8;
            const int sub = tid % 8;
            float x[per];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < per; ++j) {
                x[j] = S.s[r][sub*per + j];
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, o));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < per; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub*per + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, o);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r]  = S.lsum[r]*a + sum;
                S.mrow[r]  = m_new;
            }
        }
        __syncthreads();

        // P V: warp w owns dims 64w..64w+63
        {
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
            }
#pragma unroll
            for (int ks = 0; ks < FATTN_LISTED_CH/16; ++ks) {
                const int cA = ks*16 + 2*tig;
                const int cB = cA + 8;
                uint32_t a[4];
                a[0] = fattn_listed_pack(S.s[gid    ][cA], S.s[gid    ][cA + 1]);
                a[1] = fattn_listed_pack(S.s[gid + 8][cA], S.s[gid + 8][cA + 1]);
                a[2] = fattn_listed_pack(S.s[gid    ][cB], S.s[gid    ][cB + 1]);
                a[3] = fattn_listed_pack(S.s[gid + 8][cB], S.s[gid + 8][cB + 1]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = warp*64 + j*8 + gid;
                    const half2 v0 = __halves2half2(S.v[cA][d], S.v[cA + 1][d]);
                    const half2 v1 = __halves2half2(S.v[cB][d], S.v[cB + 1][d]);
                    const uint32_t b[2] = { *reinterpret_cast<const uint32_t *>(&v0), *reinterpret_cast<const uint32_t *>(&v1) };
                    fattn_listed_mma(tmp[j], a, b);
                }
            }
            const float a0 = S.alpha[gid];
            const float a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = acc[j][0]*a0 + tmp[j][0];
                acc[j][1] = acc[j][1]*a0 + tmp[j][1];
                acc[j][2] = acc[j][2]*a1 + tmp[j][2];
                acc[j][3] = acc[j][3]*a1 + tmp[j][3];
            }
        }
    }
    __syncthreads();

    const float l0 = S.lsum[gid];
    const float l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f/l0 : 0.0f;
    const float i1 = l1 > 0.0f ? 1.0f/l1 : 0.0f;
    float * out = dst + ((int64_t(seq)*ne01 + qi)*ne02 + int64_t(kvh)*gqa)*FATTN_LISTED_D;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp*64 + j*8 + 2*tig;
        if (gid < gqa) {
            *reinterpret_cast<float2 *>(out + gid*FATTN_LISTED_D + d) = make_float2(acc[j][0]*i0, acc[j][1]*i0);
        }
        if (gid + 8 < gqa) {
            *reinterpret_cast<float2 *>(out + (gid + 8)*FATTN_LISTED_D + d) = make_float2(acc[j][2]*i1, acc[j][3]*i1);
        }
    }
#endif
}

bool ggml_cuda_flash_attn_ext_listed_supported(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * cells = dst->src[5];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[device].cc;

    return cells != nullptr && GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        Q->type == GGML_TYPE_F32 && Q->ne[0] == FATTN_LISTED_D && K->ne[0] == FATTN_LISTED_D && V->ne[0] == FATTN_LISTED_D &&
        K->type == V->type && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0) &&
        Q->ne[2] % K->ne[2] == 0 && Q->ne[2] / K->ne[2] <= 16 && Q->ne[3] % K->ne[3] == 0 &&
        mask != nullptr && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && sinks == nullptr &&
        max_bias == 0.0f && logit_softcap == 0.0f && dst->type == GGML_TYPE_F32 &&
        (K->type != GGML_TYPE_F16 || (K->nb[1] % sizeof(uint4) == 0 && K->nb[2] % sizeof(uint4) == 0 && K->nb[3] % sizeof(uint4) == 0 &&
                                      V->nb[1] % sizeof(uint4) == 0 && V->nb[2] % sizeof(uint4) == 0 && V->nb[3] % sizeof(uint4) == 0));
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

void ggml_cuda_flash_attn_ext_listed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * cells = dst->src[5];

    GGML_ASSERT(ggml_is_contiguous(cells) && cells->ne[1] == mask->ne[1] && cells->ne[3] == mask->ne[3]);
    GGML_ASSERT(ggml_is_contiguous(dst));

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const dim3 blocks(Q->ne[1], K->ne[2], Q->ne[3]);
    const dim3 threads(FATTN_LISTED_NT, 1, 1);

    auto launch = [&](auto kernel) {
        kernel<<<blocks, threads, 0, ctx.stream()>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (const int32_t *) cells->data, (float *) dst->data, scale*1.4426950408889634f,
            int(cells->ne[0]), int(K->ne[1]), int(Q->ne[1]), int(Q->ne[2]), int(Q->ne[2] / K->ne[2]),
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            int(mask->ne[1]), int(mask->ne[3]), mask->nb[1], mask->nb[3], int(Q->ne[3] / K->ne[3]));
    };

    switch (K->type) {
        case GGML_TYPE_Q8_0: launch(flash_attn_ext_listed<GGML_TYPE_Q8_0>); break;
        case GGML_TYPE_F16:  launch(flash_attn_ext_listed<GGML_TYPE_F16>);  break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}
