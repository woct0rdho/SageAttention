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

#include <torch/all.h>

void quant_per_block_int8_scale_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                double sm_scale,
                int64_t block_size,
                int64_t tensor_layout);

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int64_t block_size,
                int64_t tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                torch::Tensor scale,
                int64_t block_size,
                int64_t tensor_layout);

void quant_per_warp_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int64_t block_size,
                int64_t warp_block_size,
                int64_t tensor_layout);

void sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                int64_t tensor_layout);

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int64_t tensor_layout);

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int64_t num_tokens,
                double scale_max,
                int64_t tensor_layout);

void mean_scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor mean,
                torch::Tensor scale,
                int64_t num_tokens,
                double scale_max,
                int64_t tensor_layout);
