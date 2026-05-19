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
    int64_t value_transposed_hnd = -1,
    int64_t pv_accum_mode = -1);

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
    int64_t value_transposed_hnd = -1,
    int64_t key_hnd_layout = 0);

Tensor qk_rawq_int8_sv_f16_native_attn_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor key_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t valid_kv_len = 0,
    int64_t pv_accum_mode = -1);

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
    int64_t value_transposed_hnd = -1,
    int64_t key_hnd_layout = 0);

Tensor sage_fp8_nhd_short_mha_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    int64_t is_causal,
    double sm_scale,
    double scale_max);

Tensor qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    int64_t is_causal,
    int64_t value_is_fp8,
    int64_t use_raw_f16_value,
    double sm_scale,
    int64_t valid_kv_len = 0,
    int64_t pv_accum_mode = -1);

std::vector<Tensor> quant_q_nhd_per_warp_gfx12(Tensor query);

Tensor transpose_value_fp8_hnd_gfx12(Tensor value);

Tensor transpose_value_fp8_scaled_hnd_gfx12(Tensor value, Tensor value_scale);

std::vector<Tensor> fp8_value_nhd_short_gfx12(
    Tensor value,
    double scale_max);

Tensor mean_nhd_gfx12(Tensor input);

Tensor mean_nhd_d64_seq32_gfx12(Tensor input);

Tensor mean_hnd_gfx12(Tensor input);

std::vector<Tensor> prepare_qkv_hnd_smooth_f16_gfx12(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor key_mean);

std::vector<Tensor> mean_and_fp8_value_nhd_short_gfx12(
    Tensor key,
    Tensor value,
    double scale_max);

Tensor transpose_value_f16_hnd_gfx12(Tensor value);

Tensor convert_f16_to_bf16_gfx12(Tensor input);
