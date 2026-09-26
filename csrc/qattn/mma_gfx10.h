#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#else
#error "mma_gfx10.h is only intended for ROCm/HIP."
#endif

#include <cstdint>
#include <type_traits>

namespace sageattn_gfx10 {

typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int sdot4_i32_i8(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

__device__ __forceinline__ float fdot2_f32_f16(unsigned a, unsigned b, float c) {
    return __builtin_amdgcn_fdot2(
        *reinterpret_cast<const v2h*>(&a), *reinterpret_cast<const v2h*>(&b), c, false);
}

__device__ __forceinline__ int load_i8_quad(const int8_t* p) {
    return *reinterpret_cast<const int*>(p);
}

__device__ __forceinline__ int8_t gfx10_q8_round(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int i = static_cast<int>(x);
    if (i > 127) i = 127;
    if (i < -128) i = -128;
    return static_cast<int8_t>(i);
}

__device__ __forceinline__ int gfx10_q8_pack(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (static_cast<int>(a) & 0xff) | ((static_cast<int>(b) & 0xff) << 8) |
           ((static_cast<int>(c) & 0xff) << 16) | ((static_cast<int>(d) & 0xff) << 24);
}

__device__ __forceinline__ float gfx10_q8_2f(unsigned raw_half, bool bf16) {
    if (bf16) {
        __hip_bfloat16 b = *reinterpret_cast<const __hip_bfloat16*>(&raw_half);
        return __bfloat162float(b);
    }
    __half h = *reinterpret_cast<const __half*>(&raw_half);
    return __half2float(h);
}

template <typename ODT> __device__ __forceinline__ ODT gfx10_out_convert(float v);
template <> __device__ __forceinline__ __half gfx10_out_convert<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __hip_bfloat16 gfx10_out_convert<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

template <int HD, bool ISC, int BN, int BM, typename ODT, int TM = 2, bool INQ = false>
__global__ __attribute__((amdgpu_num_vgpr(224))) __launch_bounds__(BM, 1) void attn_kernel_i8q_f16pv_tiled_pv(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const __half* __restrict__ v, ODT* __restrict__ out,
    const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    int64_t batch, int64_t qo_len, int64_t kv_len,
    int64_t q_heads, int64_t kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h,
    int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h,
    int64_t ks_stride_b, int64_t ks_stride_h,
    int diag,
    const void* __restrict__ q_fp, int q_src_bf16, float sm_scale_log2e) {
#if defined(__GFX10__)
    const int diag_qk = diag & 1;
    const int diag_sm = (diag >> 1) & 1;
    const int diag_pv = (diag >> 2) & 1;
    const int diag_st = (diag >> 3) & 1;
    const int diag_wb = (diag >> 4) & 1;
    constexpr int QUADS = HD / 4;
    constexpr int NTHREAD = BM;
    constexpr int K_STRIDE = HD;
    constexpr int V_STRIDE = BN;
    constexpr int KS_TOTAL = BN * K_STRIDE;
    constexpr int VS_TOTAL = HD * V_STRIDE;
    constexpr int VDSW = 8;

    constexpr int TD = HD / TM;
    constexpr int NROWG = BM / TM;
    constexpr int NDMG = HD / TD;
    static_assert(NROWG * NDMG == NTHREAD, "lane tiling must fill the block");
    static_assert(NROWG * NDMG == BM, "lane tiling must fill the block");
    static_assert(BN % 2 == 0, "BN must be even");
    static_assert(KS_TOTAL * 2 + VS_TOTAL * 2 * (int)sizeof(__half)
                  + BM * BN * (int)sizeof(__half)
                  + BM * 2 * (int)sizeof(float) <= 49152, "LDS budget");

    __shared__ __attribute__((aligned(32))) int8_t k_buf[2][KS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half v_buf[2][VS_TOTAL];
    __shared__ __attribute__((aligned(32))) __half p_buf[BN * BM];
    __shared__ __attribute__((aligned(32))) float alpha_buf[BM];
    __shared__ __attribute__((aligned(32))) float l_buf[BM];

    const int tid = threadIdx.x;
    const int64_t m = blockIdx.x * BM + tid;
    const int64_t b = blockIdx.z;
    const int64_t h = blockIdx.y;
    const int64_t kvh = h / (q_heads / kv_heads);
    const bool valid = (m < qo_len) && (h < q_heads);

    float qsv;
    int q_reg[QUADS];
    if constexpr (INQ) {
        #pragma unroll
        for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
        const int64_t qb = b * q_stride_b + h * q_stride_h;
        float row_amax = 1e-7f;
        if (valid) {
            const char* qrow = reinterpret_cast<const char*>(q_fp) + (qb + m * q_stride_n) * 2;
            constexpr int NW = HD / 2;
            #pragma unroll
            for (int x = 0; x < NW; ++x) {
                const unsigned u = reinterpret_cast<const unsigned*>(qrow)[x];
                const float f0 = gfx10_q8_2f(u & 0xffffu, q_src_bf16 != 0);
                const float f1 = gfx10_q8_2f(u >> 16, q_src_bf16 != 0);
                row_amax = fmaxf(row_amax, fabsf(f0));
                row_amax = fmaxf(row_amax, fabsf(f1));
            }
        }

        #pragma unroll
        for (int sh = 1; sh < 32; sh <<= 1)
            row_amax = fmaxf(row_amax, __shfl_xor(row_amax, sh));
        qsv = valid ? (row_amax * (1.0f / 127.0f) * sm_scale_log2e) : 0.0f;
        if (valid) {
            const char* qrow = reinterpret_cast<const char*>(q_fp) + (qb + m * q_stride_n) * 2;
            constexpr int NW = HD / 2;
            const float iscale = 127.0f / row_amax;
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) {
                const unsigned a = reinterpret_cast<const unsigned*>(qrow)[2 * dq];
                const unsigned b_ = reinterpret_cast<const unsigned*>(qrow)[2 * dq + 1];
                const float f0 = gfx10_q8_2f(a & 0xffffu, q_src_bf16 != 0);
                const float f1 = gfx10_q8_2f(a >> 16, q_src_bf16 != 0);
                const float f2 = gfx10_q8_2f(b_ & 0xffffu, q_src_bf16 != 0);
                const float f3 = gfx10_q8_2f(b_ >> 16, q_src_bf16 != 0);
                q_reg[dq] = gfx10_q8_pack(
                    gfx10_q8_round(f0 * iscale), gfx10_q8_round(f1 * iscale),
                    gfx10_q8_round(f2 * iscale), gfx10_q8_round(f3 * iscale));
            }
        } else {
            qsv = 0.0f;
        }
    } else {
        qsv = valid
            ? q_scale[b * qs_stride_b + h * qs_stride_h + static_cast<int>(m / MIN_BLK_Q)] : 0.0f;
        if (valid) {
            const int64_t qb = b * q_stride_b + h * q_stride_h;
            const int8_t* qrow = q + qb + m * q_stride_n;
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = load_i8_quad(qrow + dq * 4);
        } else {
            #pragma unroll
            for (int dq = 0; dq < QUADS; ++dq) q_reg[dq] = 0;
        }
    }

    float acc[TM][TD];
    float row_m = -3.0e38f, row_l = 0.0f;
    #pragma unroll
    for (int u = 0; u < TM; ++u)
        #pragma unroll
        for (int dd = 0; dd < TD; ++dd) acc[u][dd] = 0.0f;

    auto stage_kv = [&](int dst, int64_t kb0) {
        #pragma unroll 1
        for (int i = tid; i < BN * QUADS; i += NTHREAD) {
            const int r = i / QUADS, ck = i % QUADS;
            const int64_t n = kb0 + r;
            reinterpret_cast<int*>(&k_buf[dst][r * K_STRIDE + ck * 4])[0] =
                (n < kv_len) ? load_i8_quad(k + b * k_stride_b + kvh * k_stride_h + n * k_stride_n + ck * 4) : 0;
        }
        #pragma unroll 1
        for (int u = 0; u < (HD * BN / VDSW) / NTHREAD; ++u) {
            const int slot = tid + u * NTHREAD;
            if (slot < HD * BN / VDSW) {

                const int n_local = slot % BN;
                const int dg = slot / BN;
                const int64_t n = kb0 + n_local;
                if ((dg * VDSW) < HD) {
                    if (n < kv_len) {
                        const __half* src = v + b * v_stride_b + kvh * v_stride_n + n * v_stride_h + dg * VDSW;
                        int4 val = *reinterpret_cast<const int4*>(src);
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] = reinterpret_cast<__half*>(&val)[jj];
                    } else {
                        #pragma unroll
                        for (int jj = 0; jj < VDSW; ++jj)
                            v_buf[dst][(dg * VDSW + jj) * V_STRIDE + n_local] = __half{0};
                    }
                }
            }
        }
    };

    stage_kv(0, 0);
    __syncthreads();

    const int64_t kb_lim = ISC ? min(kv_len, min(qo_len, blockIdx.x * static_cast<int64_t>(BM) + BM)) : kv_len;

    #pragma unroll 1
    for (int64_t kb = 0; kb < kb_lim; kb += BN) {
        const int buf = static_cast<int>((kb / BN) & 1);
        const int64_t nb = kb + BN;

        if ((nb < kb_lim) && !diag_st) {
            stage_kv(buf ^ 1, nb);
        }

        float scr[BN];
        {
            #pragma unroll
            for (int j0 = 0; j0 < BN; j0 += 4) {
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                if (!diag_qk)
                #pragma unroll
                for (int dq = 0; dq < QUADS; dq += 4) {
                    const int4 k0 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 0) * K_STRIDE + dq * 4]);
                    const int4 k1 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 1) * K_STRIDE + dq * 4]);
                    const int4 k2 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 2) * K_STRIDE + dq * 4]);
                    const int4 k3 = *reinterpret_cast<const int4*>(&k_buf[buf][(j0 + 3) * K_STRIDE + dq * 4]);
                    s0 = sdot4_i32_i8(q_reg[dq + 0], k0.x, s0); s0 = sdot4_i32_i8(q_reg[dq + 1], k0.y, s0);
                    s0 = sdot4_i32_i8(q_reg[dq + 2], k0.z, s0); s0 = sdot4_i32_i8(q_reg[dq + 3], k0.w, s0);
                    s1 = sdot4_i32_i8(q_reg[dq + 0], k1.x, s1); s1 = sdot4_i32_i8(q_reg[dq + 1], k1.y, s1);
                    s1 = sdot4_i32_i8(q_reg[dq + 2], k1.z, s1); s1 = sdot4_i32_i8(q_reg[dq + 3], k1.w, s1);
                    s2 = sdot4_i32_i8(q_reg[dq + 0], k2.x, s2); s2 = sdot4_i32_i8(q_reg[dq + 1], k2.y, s2);
                    s2 = sdot4_i32_i8(q_reg[dq + 2], k2.z, s2); s2 = sdot4_i32_i8(q_reg[dq + 3], k2.w, s2);
                    s3 = sdot4_i32_i8(q_reg[dq + 0], k3.x, s3); s3 = sdot4_i32_i8(q_reg[dq + 1], k3.y, s3);
                    s3 = sdot4_i32_i8(q_reg[dq + 2], k3.z, s3); s3 = sdot4_i32_i8(q_reg[dq + 3], k3.w, s3);
                }
                for (int tj = 0; tj < 4; ++tj) {
                    int s = (tj == 0) ? s0 : (tj == 1) ? s1 : (tj == 2) ? s2 : s3;
                    const int64_t n = kb + j0 + tj;
                    const float ksj = k_scale[b * ks_stride_b + kvh * ks_stride_h + static_cast<int>(n / MIN_BLK_K)];
                    float sc = static_cast<float>(s) * (qsv * ksj);
                    if ((!valid) || (ISC && n > m) || (n >= kv_len)) sc = -3.0e38f;
                    scr[j0 + tj] = sc;
                }
            }
        }

        float P_al = 1.0f;
        if (!diag_sm) {
            float lm = scr[0];
            #pragma unroll
            for (int j = 1; j < BN; ++j) lm = fmaxf(lm, scr[j]);
            float gm = fmaxf(row_m, lm);
            float alpha = (row_l > 0.0f) ? exp2f(row_m - gm) : 0.0f;
            P_al = alpha;
            row_m = gm;
            row_l *= alpha;
            float ps = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) { float p = exp2f(scr[j] - row_m); ps += p; }
            row_l += ps;
            #pragma unroll
            for (int j = 0; j < BN; ++j) p_buf[j * BM + (int)(m % BM)] = __float2half(exp2f(scr[j] - row_m));
        } else {
            #pragma unroll
            for (int j = 0; j < BN; ++j) p_buf[j * BM + (int)(m % BM)] = __float2half(scr[j]);
        }
        alpha_buf[m % BM] = P_al;
        l_buf[m % BM] = row_l;

        __syncthreads();

        if (!diag_pv) {
            const int rg = tid % NROWG;
            const int dgo = tid / NROWG;
            const int r0 = rg * TM;
            #pragma unroll
            for (int u = 0; u < TM; ++u) {
                const float al = alpha_buf[r0 + u];
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd) acc[u][dd] *= al;
            }
            #pragma unroll
            for (int kp = 0; kp < BN / 2; ++kp) {
                const int k = kp * 2;
                unsigned pp[TM];
                #pragma unroll
                for (int up = 0; up < TM / 2; ++up) {

                    const unsigned pk0 = *reinterpret_cast<const unsigned*>(&p_buf[k * BM + r0 + up * 2]);
                    const unsigned pk1 = *reinterpret_cast<const unsigned*>(&p_buf[(k + 1) * BM + r0 + up * 2]);
                    pp[up * 2]     = (pk0 & 0xffffu) | ((pk1 & 0xffffu) << 16);
                    pp[up * 2 + 1] = (pk0 >> 16)      | (pk1 & 0xffff0000u);
                }
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd) {
                    const int d = dgo * TD + dd;
                    const unsigned vv = *reinterpret_cast<const unsigned*>(&v_buf[buf][d * V_STRIDE + k]);
                    #pragma unroll
                    for (int u = 0; u < TM; ++u)
                        acc[u][dd] = fdot2_f32_f16(pp[u], vv, acc[u][dd]);
                }
            }
        }

        __syncthreads();
    }

    if (!diag_wb) {
        const int rg = tid % NROWG;
        const int dgo = tid / NROWG;
        const int r0 = rg * TM;
        #pragma unroll
        for (int u = 0; u < TM; ++u) {
            const int64_t row = blockIdx.x * BM + r0 + u;
            if ((row < qo_len) && (h < q_heads)) {
                const float inv = 1.0f / l_buf[r0 + u];
                const int64_t base = b * o_stride_b + row * o_stride_n + h * o_stride_h;
                #pragma unroll
                for (int dd = 0; dd < TD; ++dd)
                    out[base + dgo * TD + dd] = gfx10_out_convert<ODT>(acc[u][dd] * inv);
            }
        }
    }
#endif
}

}