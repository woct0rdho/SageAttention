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

// #include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor qk_int8_sv_f16_accum_f32_attn(Tensor query,
                    Tensor key,
                    Tensor value,
                    Tensor output,
                    Tensor query_scale,
                    Tensor key_scale,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_lse);

Tensor qk_int8_sv_f16_accum_f16_attn(Tensor query,
                    Tensor key,
                    Tensor value,
                    Tensor output,
                    Tensor query_scale,
                    Tensor key_scale,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_lse);

Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(Tensor query,
                    Tensor key,
                    Tensor value,
                    Tensor output,
                    Tensor query_scale,
                    Tensor key_scale,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_lse);

Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(Tensor query,
                    Tensor key,
                    Tensor value,
                    Tensor output,
                    Tensor query_scale,
                    Tensor key_scale,
                    Tensor value_mean,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_lse);
