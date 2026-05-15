/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

Tensor qk_int8_sv_f16_d64_native_attn_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor query_scale,
    Tensor key_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t valid_kv_len = 0,
    int64_t value_transposed_hnd = -1);

Tensor qk_rawq_int8_sv_f8_native_attn_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor key_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t valid_kv_len = 0,
    int64_t value_transposed_hnd = -1);

Tensor qk_int8_sv_f8_scaled_native_attn_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor query_scale,
    Tensor key_scale,
    Tensor value_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t valid_kv_len = 0);

Tensor qk_rawq_int8_sv_f8_scaled_native_attn_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor key_scale,
    Tensor value_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t valid_kv_len = 0,
    int64_t value_transposed_hnd = -1);

Tensor qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    int64_t is_causal,
    int64_t value_is_fp8,
    int64_t use_raw_f16_value,
    double sm_scale,
    int64_t valid_kv_len = 0);

std::vector<Tensor> quant_q_nhd_per_warp_gfx12(Tensor query);

Tensor transpose_value_fp8_hnd_gfx12(Tensor value);

Tensor transpose_value_fp8_scaled_hnd_gfx12(Tensor value, Tensor value_scale);

Tensor transpose_value_f16_hnd_gfx12(Tensor value);

Tensor convert_f16_to_bf16_gfx12(Tensor input);
