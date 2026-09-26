#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "attn_gfx110x.h"

PyMODINIT_FUNC PyInit__qattn_gfx110x(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_gfx110x",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

STABLE_TORCH_LIBRARY(sageattention_qattn_gfx110x, m) {
    m.def("qk_int8_sv_bf16_attn_t("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, "
            "int is_causal, float sm_scale, Tensor q_fp"
          ") -> Tensor");
    m.def("fp16_attn_t("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "int tensor_layout, int is_causal, float sm_scale, int bm_sel"
          ") -> Tensor");
    m.def("bf16_attn_t("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "int tensor_layout, int is_causal, float sm_scale, int bm_sel"
          ") -> Tensor");
    m.def("quant_qk_int8("
            "Tensor query, Tensor key, Tensor key_mean, int tensor_layout, "
            "float sm_scale, int skip_q"
          ") -> Tensor[]");
    m.def("mean_seq(Tensor input, int tensor_layout) -> Tensor");
    m.def("v_transpose(Tensor value, Tensor(a!) value_t, int tensor_layout) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(sageattention_qattn_gfx110x, CUDA, m) {
    m.impl("qk_int8_sv_bf16_attn_t", TORCH_BOX(qk_int8_sv_bf16_attn_gfx110x_t));
    m.impl("fp16_attn_t", TORCH_BOX(fp16_attn_gfx110x_t));
    m.impl("bf16_attn_t", TORCH_BOX(bf16_attn_gfx110x_t));
    m.impl("quant_qk_int8", TORCH_BOX(quant_qk_int8_gfx110x));
    m.impl("mean_seq", TORCH_BOX(mean_seq_gfx110x));
    m.impl("v_transpose", TORCH_BOX(v_transpose_gfx110x));
}
