#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "attn_gfx103x.h"

PyMODINIT_FUNC PyInit__qattn_gfx103x(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_qattn_gfx103x",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

STABLE_TORCH_LIBRARY(sageattention_qattn_gfx103x, m) {
    m.def("qk_int8_sv_bf16_attn_t("
            "Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
            "Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, "
            "int is_causal, float sm_scale, Tensor q_fp"
          ") -> Tensor");
    m.def("quant_qk_int8("
            "Tensor query, Tensor key, Tensor key_mean, int tensor_layout, "
            "float sm_scale, int skip_q"
          ") -> Tensor[]");
    m.def("mean_seq(Tensor input, int tensor_layout) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(sageattention_qattn_gfx103x, CUDA, m) {
    m.impl("qk_int8_sv_bf16_attn_t", TORCH_BOX(qk_int8_sv_bf16_attn_gfx103x_t));
    m.impl("quant_qk_int8", TORCH_BOX(quant_qk_int8_gfx103x));
    m.impl("mean_seq", TORCH_BOX(mean_seq_gfx103x));
}