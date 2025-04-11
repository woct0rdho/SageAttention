/*
 * Adapted from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/mma.cuh
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Modifications copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace mma{

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define MMA_F16F16F32_M16N8K16_ENABLED
#define MMA_F16F16F16_M16N8K16_ENABLED
#define MMA_S8S8S32_M16N8K32_ENABLED
#define MMA_S4S4S32_M16N8K64_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define MMA_F16F16F32_M16N8K8_ENABLED
#define MMA_F16F16F16_M16N8K8_ENABLED
#define LDMATRIX_M8N8X2_ENABLED
#define LDMATRIX_M8N8X4_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define MMA_F8F8F32_M16N8K16_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define MMA_F8F8F16_M16N8K16_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 800
#define RUNTIME_ASSERT(x) __brkpt()
#else
#define RUNTIME_ASSERT(x) do {} while(0)  // Do nothing at runtime
#endif
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x2 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x2(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X2_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  // ! only support half dtype now
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else  // MMA_F16F16F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    // SM_75 fallback for InplaceUpdate
    asm volatile(
        "{\n"
        ".reg .b32 tmp0, tmp1, tmp2, tmp3;\n" // Temp float registers for intermediate accumulation

        // First m16n8k8: Processes first half of K dimension (k=0..7)
        // Inputs: A[0], A[1] (holding first 4 f16s of A's K dim)
        //         B[0]      (holding first 2 f16s of B's K dim)
        // Accumulates from existing C values into temps
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{tmp0,  tmp1,  tmp2,  tmp3}," // Output to temps
        "{%4,  %5},"                   // Input A registers {A[0], A[1]}
        "{%8},"                        // Input B register {B[0]}
        "{%10, %11, %12, %13};\n"      // Input C accumulators {C[0], C[1], C[2], C[3]}

        // Second m16n8k8: Processes second half of K dimension (k=8..15)
        // Inputs: A[2], A[3] (holding second 4 f16s of A's K dim)
        //         B[1]      (holding second 2 f16s of B's K dim)
        // Accumulates from temps into final C registers
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"         // Output to final C registers {C[0], C[1], C[2], C[3]}
        "{%6,  %7},"                   // Input A registers {A[2], A[3]}
        "{%9},"                        // Input B register {B[1]}
        "{tmp0,  tmp1,  tmp2,  tmp3};\n" // Input C accumulators (temps from previous step)
        "}\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])   // %0..3: Output registers (final C)
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),     // %4..7: Input registers for A
          "r"(B[0]), "r"(B[1]),                           // %8..9: Input registers for B
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));   // %10..13: Input registers for initial C (read-only)
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    // SM_75 fallback for Init
    asm volatile(
        "{\n"
        ".reg .b32 tmp0, tmp1, tmp2, tmp3;\n" // Temp float registers

        // First m16n8k8: Processes first half of K dimension (k=0..7)
        // Inputs: A[0], A[1], B[0]
        // Accumulates from 0.f into temps
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{tmp0,  tmp1,  tmp2,  tmp3}," // Output to temps
        "{%4,  %5},"                   // Input A registers {A[0], A[1]}
        "{%8},"                        // Input B register {B[0]}
        "{%10, %11, %12, %13};\n"      // Input C accumulators {0.f, 0.f, 0.f, 0.f}

        // Second m16n8k8: Processes second half of K dimension (k=8..15)
        // Inputs: A[2], A[3], B[1]
        // Accumulates from temps into final C registers
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"         // Output to final C registers {C[0], C[1], C[2], C[3]}
        "{%6,  %7},"                   // Input A registers {A[2], A[3]}
        "{%9},"                        // Input B register {B[1]}
        "{tmp0,  tmp1,  tmp2,  tmp3};\n" // Input C accumulators (temps from previous step)
        "}\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])   // %0..3: Output registers (final C)
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),     // %4..7: Input registers for A
          "r"(B[0]), "r"(B[1]),                           // %8..9: Input registers for B
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));       // %10..13: Input registers for initial C (zeros)
  }
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  // ! only support half dtype now
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else  // MMA_F16F16F32_M16N8K16_ENABLED
  // Define temporary registers for the accumulator chaining
  float t0, t1, t2, t3;

  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
      // --- First m16n8k16 replacement (uses B[0], B[1]) ---
      // First m16n8k8 step (K=0..7)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // Temporaries
          "{%4, %5},"          // A[0], A[1]
          "{%6},"              // B[0]
          "{%7, %8, %9, %10};" // C[0], C[1], C[2], C[3] initial
          : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
          : "r"(A[0]), "r"(A[1]), "r"(B[0]),
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
      );
      // Second m16n8k8 step (K=8..15), accumulating into temps
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // C[0], C[1], C[2], C[3] final
          "{%4, %5},"          // A[2], A[3]
          "{%6},"              // B[1]
          "{%7, %8, %9, %10};" // Temporaries from previous step
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]),
            "f"(t0), "f"(t1), "f"(t2), "f"(t3)
      );

      // --- Second m16n8k16 replacement (uses B[2], B[3]) ---
       // First m16n8k8 step (K=0..7)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // Temporaries
          "{%4, %5},"          // A[0], A[1]
          "{%6},"              // B[2]
          "{%7, %8, %9, %10};" // C[4], C[5], C[6], C[7] initial
          : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
          : "r"(A[0]), "r"(A[1]), "r"(B[2]),
            "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
      );
      // Second m16n8k8 step (K=8..15), accumulating into temps
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // C[4], C[5], C[6], C[7] final
          "{%4, %5},"          // A[2], A[3]
          "{%6},"              // B[3]
          "{%7, %8, %9, %10};" // Temporaries from previous step
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]),
            "f"(t0), "f"(t1), "f"(t2), "f"(t3)
      );
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
      // --- First m16n8k16 replacement (uses B[0], B[1]) ---
      // First m16n8k8 step (K=0..7)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // Temporaries
          "{%4, %5},"          // A[0], A[1]
          "{%6},"              // B[0]
          "{%7, %8, %9, %10};" // Zero initializers
          : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
          : "r"(A[0]), "r"(A[1]), "r"(B[0]),
            "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
      );
      // Second m16n8k8 step (K=8..15), accumulating into temps
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // C[0], C[1], C[2], C[3] final
          "{%4, %5},"          // A[2], A[3]
          "{%6},"              // B[1]
          "{%7, %8, %9, %10};" // Temporaries from previous step
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]),
            "f"(t0), "f"(t1), "f"(t2), "f"(t3)
      );

      // --- Second m16n8k16 replacement (uses B[2], B[3]) ---
       // First m16n8k8 step (K=0..7)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // Temporaries
          "{%4, %5},"          // A[0], A[1]
          "{%6},"              // B[2]
          "{%7, %8, %9, %10};" // Zero initializers
          : "=f"(t0), "=f"(t1), "=f"(t2), "=f"(t3)
          : "r"(A[0]), "r"(A[1]), "r"(B[2]),
            "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
      );
      // Second m16n8k8 step (K=8..15), accumulating into temps
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0, %1, %2, %3}," // C[4], C[5], C[6], C[7] final
          "{%4, %5},"          // A[2], A[3]
          "{%6},"              // B[3]
          "{%7, %8, %9, %10};" // Temporaries from previous step
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]),
            "f"(t0), "f"(t1), "f"(t2), "f"(t3)
      );
  }
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  }
#else  // MMA_F16F16F16_M16N8K16_ENABLED
// Preload initial accumulator values (or zero if kInit)
  uint32_t c0_in = (mma_mode == MMAMode::kInit) ? 0 : C[0];
  uint32_t c1_in = (mma_mode == MMAMode::kInit) ? 0 : C[1];
  uint32_t c2_in = (mma_mode == MMAMode::kInit) ? 0 : C[2];
  uint32_t c3_in = (mma_mode == MMAMode::kInit) ? 0 : C[3];

  // Fragment mapping:
  // A: {A[0], A[1]} for k=0..7, {A[2], A[3]} for k=8..15
  // B: {B[0]} k=0..7,n=0..7 | {B[1]} k=8..15,n=0..7 | {B[2]} k=0..7,n=8..15 | {B[3]} k=8..15,n=8..15
  // C: {C[0], C[1]} n=0..7 | {C[2], C[3]} n=8..15

  asm volatile(
    "{\n"
    // Allocate 4 temporary registers (.b32) for intermediate f16 pairs
    "  .reg .b32 t0, t1, t2, t3;\n\n"

    // --- Compute first 16x8 tile (C0: n=0..7) ---
    // 1. First K-half: tmp{0,1} = A[k=0..7] * B[k=0..7, n=0..7] + C_in{0,1}
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{t0, t1}, {%4, %5}, {%8}, {%12, %13};\n" // Inputs: A[0,1], B[0], C_in[0,1]
    // 2. Second K-half: C{0,1} = A[k=8..15] * B[k=8..15, n=0..7] + tmp{0,1}
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{%0, %1}, {%6, %7}, {%9}, {t0, t1};\n\n" // Inputs: A[2,3], B[1], tmp[0,1] -> Output: C[0,1]

    // --- Compute second 16x8 tile (C1: n=8..15) ---
    // 1. First K-half: tmp{2,3} = A[k=0..7] * B[k=0..7, n=8..15] + C_in{2,3}
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{t2, t3}, {%4, %5}, {%10}, {%14, %15};\n" // Inputs: A[0,1], B[2], C_in[2,3]
    // 2. Second K-half: C{2,3} = A[k=8..15] * B[k=8..15, n=8..15] + tmp{2,3}
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{%2, %3}, {%6, %7}, {%11}, {t2, t3};\n" // Inputs: A[2,3], B[3], tmp[2,3] -> Output: C[2,3]
    "}\n"
    : // Output Constraints: Final accumulator values C[0]..C[3]
      "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
    : // Input Constraints: A[0]..A[3], B[0]..B[3], initial C_in[0]..C_in[3]
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),     // %4 , %5 , %6 , %7
      "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),     // %8 , %9 , %10, %11
      "r"(c0_in), "r"(c1_in), "r"(c2_in), "r"(c3_in)  // %12, %13, %14, %15
    // No explicit clobbers needed for .reg allocated temps
  );
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major int8 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S8S8S32_M16N8K32_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else  // MMA_S8S8S32_M16N8K32_ENABLED

  // Emulate m16n8k32.s8.s8.s32 using four m8n8k16.s8.s8.s32 operations.
  // SM_75 supports m8n8k16 for s8 inputs.
  // A[0..3] holds the M=16, K=32 fragment components.
  // B[0..1] holds the K=32, N=8 fragment components.
  // C[0..3] holds the M=16, N=8 accumulator components.
  //   C[0], C[1] -> M=0..7 part
  //   C[2], C[3] -> M=8..15 part

  int32_t tmp0, tmp1, tmp2, tmp3; // Temporary accumulators for first K-half

  // === K = 0..15 part ===

  // MMA 1: M = 0..7, N = 0..7, K = 0..15
  // Inputs: A[0], B[0]
  // Accumulator: C[0], C[1] (or 0 if kInit)
  // Output: tmp0, tmp1
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
      : "=r"(tmp0), "=r"(tmp1)
      : "r"(A[0]),
        "r"(B[0]),
        "r"( (mma_mode == MMAMode::kInit) ? 0 : C[0] ),
        "r"( (mma_mode == MMAMode::kInit) ? 0 : C[1] )
  );

  // MMA 2: M = 8..15, N = 0..7, K = 0..15
  // Inputs: A[2], B[0]
  // Accumulator: C[2], C[3] (or 0 if kInit)
  // Output: tmp2, tmp3
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
      : "=r"(tmp2), "=r"(tmp3)
      : "r"(A[2]), // This operand corresponds to rows M=8..15
        "r"(B[0]),
        "r"( (mma_mode == MMAMode::kInit) ? 0 : C[2] ),
        "r"( (mma_mode == MMAMode::kInit) ? 0 : C[3] )
  );


  // === K = 16..31 part ===
  // Accumulate onto results from first K-half (stored in tmp0..3)

  // MMA 3: M = 0..7, N = 0..7, K = 16..31
  // Inputs: A[1], B[1]
  // Accumulator: tmp0, tmp1
  // Output: C[0], C[1]
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
      : "=r"(C[0]), "=r"(C[1]) // Write final result directly to C
      : "r"(A[1]), // Corresponds to K=16..31 part
        "r"(B[1]),
        "r"(tmp0),  // Accumulate with first K-half result
        "r"(tmp1)
  );

  // MMA 4: M = 8..15, N = 0..7, K = 16..31
  // Inputs: A[3], B[1]
  // Accumulator: tmp2, tmp3
  // Output: C[2], C[3]
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
      : "=r"(C[2]), "=r"(C[3]) // Write final result directly to C
      : "r"(A[3]), // Corresponds to M=8..15, K=16..31 part
        "r"(B[1]),
        "r"(tmp2),  // Accumulate with first K-half result
        "r"(tmp3)
  );
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major int8 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S8S8S32_M16N8K32_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else  // SM_75 fallback path
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction"); //this
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major int4 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k64_row_col_s4s4s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S4S4S32_M16N8K64_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k64 instruction for row major and column major int4 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k64_row_col_s4s4s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S4S4S32_M16N8K64_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major fp8 e4m3 matrix
 *   multiplication, accumulated in fp32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major fp8 matrix
 *   multiplication, accumulated in fp16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f16(uint32_t* C_uint32, uint32_t* A,
                                                                   uint32_t* B) {
 //uint32_t* C_uint32 = reinterpret_cast<uint32_t*>(C);
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C_uint32[0]), "=r"(C_uint32[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C_uint32[0]), "r"(C_uint32[1]));

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C_uint32[2]), "=r"(C_uint32[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C_uint32[2]), "r"(C_uint32[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C_uint32[0]), "=r"(C_uint32[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C_uint32[2]), "=r"(C_uint32[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}



/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major fp8 matrix
 *   multiplication, accumulated in fp32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
          "f"(C[6]), "f"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
__device__ __forceinline__ void rowsum_f16f16f32(float* d, uint32_t* s) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  asm volatile(
      "{\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      "}\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(1006648320), // 1006648320 packs two 1.0f in half precision
        "r"(1006648320), "f"(d[0]), "f"(d[1]));
#else
  // SM75 fallback using m16n8k8 instruction for k8 instead of k16
  // We need to do the same operation in two parts

  // First half of the input (k=0..7)
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3},"
      "{%4},"
      "{%5,  0.,  %6,  0.};\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(1006648320),  // 1006648320 packs two 1.0f in half precision
        "f"(d[0]), "f"(d[1]));

  // Second half of the input (k=8..15)
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3},"
      "{%4},"
      "{%5,  0.,  %6,  0.};\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[2]), "r"(s[3]), "r"(1006648320),  // 1006648320 packs two 1.0f in half precision
        "f"(d[0]), "f"(d[1]));
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
__device__ __forceinline__ void rowsum_f8f8f32(float* d, uint32_t* s) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(943208504), "r"(943208504), // 943208504 packs four 1.0f in e4m3
        "f"(d[0]), "f"(d[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

} // namespace mma
