/*
 * Copyright (c) 2024 by SageAttention team.
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

#include <vector>

#include <cuda_runtime_api.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/util/Exception.h>

inline torch::stable::accelerator::DeviceGuard make_device_guard(const torch::stable::Tensor &tensor) {
  return torch::stable::accelerator::DeviceGuard(tensor.get_device_index());
}

inline cudaStream_t get_current_cuda_stream(const torch::stable::Tensor &tensor) {
  // We rely on the raw shim API to get the current CUDA stream.
  // This will be improved in a future release of PyTorch, see https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
  // This is needed because torch.compile may launch kernels on non-default streams.
  // Use this while a tensor-device guard is alive for multi-GPU correctness.
  const auto device_index = tensor.get_device_index();
  void *stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

#define CHECK_CUDA(x) \
  STD_TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype)     \
  STD_TORCH_CHECK(x.scalar_type() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim)    \
  STD_TORCH_CHECK(x.dim() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_NUMEL(x, minimum)     \
  STD_TORCH_CHECK(x.numel() >= minimum, \
              "Tensor " #x " must have at last " #minimum " elements")
// https://github.com/Dao-AILab/flash-attention/blob/add175637c5d54b74bc25372e49ce282d6f236fc/hopper/flash_api_stable.cpp#L98
#define CHECK_SHAPE(x, ...) \
    do { \
        auto expected_dims = std::vector<int64_t>{__VA_ARGS__}; \
        STD_TORCH_CHECK(x.dim() == static_cast<int64_t>(expected_dims.size()), #x " must have " + std::to_string(expected_dims.size()) + " dimensions, got " + std::to_string(x.dim())); \
        for (size_t i = 0; i < expected_dims.size(); ++i) { \
            STD_TORCH_CHECK(x.size(i) == expected_dims[i], #x " dimension " + std::to_string(i) + " must have size " + std::to_string(expected_dims[i]) + ", got " + std::to_string(x.size(i))); \
        } \
    } while (0)
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.stride(-1) == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")
