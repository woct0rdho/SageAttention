#pragma once

#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

// gfx110x (RDNA3) 转置布局 attention kernel 的 host 分发入口 (由 pybind 映射到 op)。
// q_fp / skip_q / v_scale 参数: gfx110x kernel 未实现 INQ 与 int8-V, 不读取;
// 保留以匹配 core.py 调用约定 (两扩展 schema 必须一致)。
Tensor qk_int8_sv_bf16_attn_gfx110x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor q_scale,
    Tensor k_scale,
    Tensor v_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    Tensor q_fp);

Tensor fp16_attn_gfx110x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t bm_sel);

Tensor bf16_attn_gfx110x_t(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    int64_t tensor_layout,
    int64_t is_causal,
    double sm_scale,
    int64_t bm_sel);

std::vector<Tensor> quant_qk_int8_gfx110x(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale,
    int64_t skip_q);

Tensor mean_seq_gfx110x(Tensor input, int64_t tensor_layout);

// V [B,N,H,D] -> V_T [B,H,D,N] (contiguous, n padded 到 64 倍数)
Tensor v_transpose_gfx110x(Tensor value, Tensor value_t, int64_t tensor_layout);