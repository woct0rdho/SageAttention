/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "attn_gfx12_native.h"

PyMODINIT_FUNC PyInit__qattn_gfx12_native(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_gfx12_native",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

STABLE_TORCH_LIBRARY(sageattention_qattn_gfx12_native, m) {
    m.def("qk_int8_sv_f16_d64_native_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor query_scale, Tensor key_scale, int tensor_layout, "
            "int is_causal, float sm_scale, int valid_kv_len=0, "
            "int value_transposed_hnd=-1, int pv_accum_mode=-1"
          ") -> Tensor");
    m.def("qk_rawq_int8_sv_f8_native_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor key_scale, int tensor_layout, int is_causal, "
            "float sm_scale, int valid_kv_len=0, int value_transposed_hnd=-1, "
            "int key_hnd_layout=0"
          ") -> Tensor");
    m.def("qk_rawq_int8_sv_f16_native_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor key_scale, int tensor_layout, int is_causal, "
            "float sm_scale, int valid_kv_len=0, int pv_accum_mode=-1"
          ") -> Tensor");
    m.def("qk_int8_sv_f8_scaled_native_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor query_scale, Tensor key_scale, Tensor value_scale, "
            "int tensor_layout, int is_causal, float sm_scale, "
            "int valid_kv_len=0"
          ") -> Tensor");
    m.def("qk_rawq_int8_sv_f8_scaled_native_attn("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor key_scale, Tensor value_scale, int tensor_layout, "
            "int is_causal, float sm_scale, int valid_kv_len=0, "
            "int value_transposed_hnd=-1, int key_hnd_layout=0"
          ") -> Tensor");
    m.def("qk_int8_sv_f16_d64_prepare_attn_hnd("
            "Tensor query, Tensor key, Tensor value, int is_causal, "
            "int value_is_fp8, int use_raw_f16_value, float sm_scale, "
            "int valid_kv_len=0, int pv_accum_mode=-1"
          ") -> Tensor");
    m.def("quant_q_nhd_per_warp(Tensor query) -> Tensor[]");
    m.def("transpose_value_fp8_hnd(Tensor value) -> Tensor");
    m.def("transpose_value_fp8_scaled_hnd(Tensor value, Tensor value_scale) -> Tensor");
    m.def("fp8_value_nhd_short(Tensor value, float scale_max) -> Tensor[]");
    m.def("mean_nhd(Tensor input) -> Tensor");
    m.def("mean_and_fp8_value_nhd_short(Tensor key, Tensor value, float scale_max) -> Tensor[]");
    m.def("transpose_value_f16_hnd(Tensor value) -> Tensor");
    m.def("convert_f16_to_bf16(Tensor input) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(sageattention_qattn_gfx12_native, CUDA, m) {
    m.impl("qk_int8_sv_f16_d64_native_attn", TORCH_BOX(qk_int8_sv_f16_d64_native_attn_gfx12));
    m.impl("qk_rawq_int8_sv_f8_native_attn", TORCH_BOX(qk_rawq_int8_sv_f8_native_attn_gfx12));
    m.impl("qk_rawq_int8_sv_f16_native_attn", TORCH_BOX(qk_rawq_int8_sv_f16_native_attn_gfx12));
    m.impl("qk_int8_sv_f8_scaled_native_attn", TORCH_BOX(qk_int8_sv_f8_scaled_native_attn_gfx12));
    m.impl("qk_rawq_int8_sv_f8_scaled_native_attn", TORCH_BOX(qk_rawq_int8_sv_f8_scaled_native_attn_gfx12));
    m.impl("qk_int8_sv_f16_d64_prepare_attn_hnd", TORCH_BOX(qk_int8_sv_f16_d64_prepare_attn_hnd_gfx12));
    m.impl("quant_q_nhd_per_warp", TORCH_BOX(quant_q_nhd_per_warp_gfx12));
    m.impl("transpose_value_fp8_hnd", TORCH_BOX(transpose_value_fp8_hnd_gfx12));
    m.impl("transpose_value_fp8_scaled_hnd", TORCH_BOX(transpose_value_fp8_scaled_hnd_gfx12));
    m.impl("fp8_value_nhd_short", TORCH_BOX(fp8_value_nhd_short_gfx12));
    m.impl("mean_nhd", TORCH_BOX(mean_nhd_gfx12));
    m.impl("mean_and_fp8_value_nhd_short", TORCH_BOX(mean_and_fp8_value_nhd_short_gfx12));
    m.impl("transpose_value_f16_hnd", TORCH_BOX(transpose_value_f16_hnd_gfx12));
    m.impl("convert_f16_to_bf16", TORCH_BOX(convert_f16_to_bf16_gfx12));
}
