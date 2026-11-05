#pragma once

#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

// gfx103x (RDNA2) int8 attention kernels 的 host 分发入口 (由 pybind 映射到 op)
Tensor qk_int8_sv_bf16_attn_gfx103x_t(
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

std::vector<Tensor> quant_qk_int8_gfx103x(
    Tensor query,
    Tensor key,
    Tensor key_mean,
    int64_t tensor_layout,
    double sm_scale,
    int64_t skip_q);

Tensor mean_seq_gfx103x(Tensor input, int64_t tensor_layout);
