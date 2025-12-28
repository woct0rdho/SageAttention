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
 * \brief [SM75 Optimized] Step 0 of m16n8k16 (Computes K=0..7)
 * Usage: Call this, then issue ldmatrix, then call Step 1.
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_f16f16f32_step0(float* C, uint32_t* A, uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K8_ENABLED
  if constexpr (mma_mode == MMAMode::kInit) {
    C[0] = 0.0f; C[1] = 0.0f; C[2] = 0.0f; C[3] = 0.0f;
  }
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3}, {%4,  %5}, {%6}, {%0,  %1,  %2,  %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0])
  );
#endif
}

/*!
 * \brief [SM75 Optimized] Step 1 of m16n8k16 (Computes K=8..15)
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_f16f16f32_step1(float* C, uint32_t* A, uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K8_ENABLED
  // Always inplace update on top of step0 result
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3}, {%4,  %5}, {%6}, {%0,  %1,  %2,  %3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[2]), "r"(A[3]), "r"(B[1]) // Note: Uses A[2/3] and B[1]
  );
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

  
  if constexpr (mma_mode == MMAMode::kInit) {
     C[0] = 0.0f; C[1] = 0.0f; C[2] = 0.0f; C[3] = 0.0f;
  }

  asm volatile(
      "{\n"
      // K=0..7: Accumulate directly into C
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3}, {%4,  %5}, {%8}, {%0,  %1,  %2,  %3};\n"
      
      // K=8..15: Accumulate directly into C
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3}, {%6,  %7}, {%9}, {%0,  %1,  %2,  %3};\n"
      "}\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), // %4-%7
        "r"(B[0]), "r"(B[1])                        // %8-%9
  );
  

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

  if constexpr (mma_mode == MMAMode::kInit) {
      C[0]=0.f; C[1]=0.f; C[2]=0.f; C[3]=0.f;
      C[4]=0.f; C[5]=0.f; C[6]=0.f; C[7]=0.f;
  }

  // Interleave L/R instructions to hide latency
  asm volatile(
      "{\n"
      // K=0..7 Left
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%8, %9}, {%12}, {%0, %1, %2, %3};\n"
      // K=0..7 Right
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%4, %5, %6, %7}, {%8, %9}, {%14}, {%4, %5, %6, %7};\n"
      
      // K=8..15 Left
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%10, %11}, {%13}, {%0, %1, %2, %3};\n"
      // K=8..15 Right
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%4, %5, %6, %7}, {%10, %11}, {%15}, {%4, %5, %6, %7};\n"
      "}\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]),
        "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), // %8-%11
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3])  // %12-%15
  );

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

  asm volatile(
    "{\n"
    "  .reg .b32 t0, t1, t2, t3;\n\n"
    // Left K0
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{t0, t1}, {%4, %5}, {%8}, {%12, %13};\n"
    // Right K0
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{t2, t3}, {%4, %5}, {%10}, {%14, %15};\n"
    // Left K1
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{%0, %1}, {%6, %7}, {%9}, {t0, t1};\n\n" 
    // Right K1
    "  mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
    "{%2, %3}, {%6, %7}, {%11}, {t2, t3};\n"
    "}\n"
    : 
      "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
    : 
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),     // %4 , %5 , %6 , %7
      "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),     // %8 , %9 , %10, %11
      "r"(c0_in), "r"(c1_in), "r"(c2_in), "r"(c3_in)  // %12, %13, %14, %15
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

  if constexpr (mma_mode == MMAMode::kInit) {
      C[0]=0; C[1]=0; C[2]=0; C[3]=0;
  }

  asm volatile(
      "{\n"
      // K Chunk 0 (0-15)
      // Top-Left: C0,C1 += A0, B0
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%4}, {%8}, {%0, %1};\n"
      // Bottom-Left: C2,C3 += A1, B0
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2, %3}, {%5}, {%8}, {%2, %3};\n"

      // K Chunk 1 (16-31)
      // Top-Left: C0,C1 += A2, B1
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%6}, {%9}, {%0, %1};\n"
      // Bottom-Left: C2,C3 += A3, B1
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2, %3}, {%7}, {%9}, {%2, %3};\n"
      "}\n"
      : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), // %4-%7
        "r"(B[0]), "r"(B[1])                        // %8-%9
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

  if constexpr (mma_mode == MMAMode::kInit) {
      C[0]=0; C[1]=0; C[2]=0; C[3]=0;
      C[4]=0; C[5]=0; C[6]=0; C[7]=0;
  }

  // Interleave L/R instructions to hide latency
  asm volatile(
      "{\n"
      // 1. Left Tile TL K0
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%8}, {%12}, {%0, %1};\n"
      // 2. Right Tile TR K0
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%4, %5}, {%8}, {%14}, {%4, %5};\n" 
      // 3. Left Tile BL K0
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2, %3}, {%9}, {%12}, {%2, %3};\n" 
      // 4. Right Tile BR K0
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%6, %7}, {%9}, {%14}, {%6, %7};\n" 

      // 5. Left Tile TL K1
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%10}, {%13}, {%0, %1};\n" 
      // 6. Right Tile TR K1
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%4, %5}, {%10}, {%15}, {%4, %5};\n" 
      // 7. Left Tile BL K1
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%2, %3}, {%11}, {%13}, {%2, %3};\n" 
      // 8. Right Tile BR K1
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%6, %7}, {%11}, {%15}, {%6, %7};\n" 
      "}\n"
      : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]),
        "+r"(C[4]), "+r"(C[5]), "+r"(C[6]), "+r"(C[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), // %8-%11
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3])  // %12-%15
  );

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

  

  float zero = 0.0f;
  
  // First half of the input (k=0..7)
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _}, {%2,  %3}, {%4}, {%0,  0.,  %1,  0.};\n"
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _}, {%5,  %6}, {%4}, {%0,  0.,  %1,  0.};\n"
      : "+f"(d[0]), "+f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(1006648320),
        "r"(s[2]), "r"(s[3])
  );
  

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
