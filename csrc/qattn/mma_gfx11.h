#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#error "mma_gfx11.h is only intended for ROCm/HIP."
#endif

#include <cstdint>

namespace sageattn_gfx11 {

typedef int int32_t_v4 __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

__device__ __forceinline__ float fast_exp2(float x) {
    return exp2f(x);
}

__device__ __forceinline__ _Float16 f16_from_bits(unsigned bits) {
    _Float16 r;
    __builtin_memcpy(&r, &bits, sizeof(_Float16));
    return r;
}

// v_permlanex16_b32: XOR-16 跨半波交换 (lane L <-> lane L^16)
__device__ __forceinline__ float permlanex16(float src) {
    float result;
    asm volatile("v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]"
                 : "=v"(result)
                 : "v"(src), "s"(0x76543210), "n"(0xfedcba98));
    return result;
}

// 32-bit 版本 (用于打包的 fp16 对)
__device__ __forceinline__ unsigned permlanex16_u32(unsigned src) {
    unsigned result;
    asm volatile("v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]"
                 : "=v"(result)
                 : "v"(src), "s"(0x76543210), "n"(0xfedcba98));
    return result;
}

__device__ __forceinline__ int32_t_v4 load_i8_frag(
    const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    int32_t_v4 r;
    const int* q = reinterpret_cast<const int*>(p);
    r[0] = q[0];
    r[1] = q[1];
    r[2] = q[2];
    r[3] = q[3];
    return r;
}

__device__ __forceinline__ v8i wmma_i32_iu8(int32_t_v4 a, int32_t_v4 b, v8i c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

__device__ __forceinline__ v8f wmma_f32_bf16(v16bf a, v16bf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

__device__ __forceinline__ v8f wmma_f32_f16(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// 组装 P B-fragment: 生成 P 行 (L&15) 的完整 16 列 (WMMA B operand = P^T 的列 L&15)。
// p_vals[e] = P[L&15][2e+hw] (本 lane 的 8 个值), 通过 permlanex16 与 lane^16 交换得奇/偶列。
__device__ __forceinline__ v16h assemble_p_frag(const float p_vals[8], int hw) {
    unsigned pack[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        __half lo = __float2half_rn(p_vals[2 * k]);
        __half hi = __float2half_rn(p_vals[2 * k + 1]);
        pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                  static_cast<unsigned>(__half_as_ushort(lo));
    }
    unsigned cross[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);
    unsigned even[4], odd[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        even[k] = (hw == 0) ? pack[k] : cross[k];
        odd[k] = (hw == 0) ? cross[k] : pack[k];
    }
    v16h b;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        b[4 * k] = f16_from_bits(even[k] & 0xFFFF);
        b[4 * k + 1] = f16_from_bits(odd[k] & 0xFFFF);
        b[4 * k + 2] = f16_from_bits(even[k] >> 16);
        b[4 * k + 3] = f16_from_bits(odd[k] >> 16);
    }
    return b;
}

}  // namespace sageattn_gfx11