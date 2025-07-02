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

#include <Python.h>
#include <torch/library.h>

#include "attn_cuda_sm80.h"

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__qattn_sm80(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_qattn_sm80",  /* name of module */
            NULL,           /* module documentation, may be NULL */
            -1,             /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
            NULL,           /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

// Defines the operators
TORCH_LIBRARY(sageattention_qattn_sm80, m) {
    m.def("qk_int8_sv_f16_accum_f32_attn("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_lse"
          ") -> Tensor");
    m.def("qk_int8_sv_f16_accum_f16_attn("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_lse"
          ") -> Tensor");
    m.def("qk_int8_sv_f16_accum_f16_attn_inst_buf("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_lse"
          ") -> Tensor");
    m.def("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_mean, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_lse"
          ") -> Tensor");
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(sageattention_qattn_sm80, CUDA, m) {
    m.impl("qk_int8_sv_f16_accum_f32_attn", &qk_int8_sv_f16_accum_f32_attn);
    m.impl("qk_int8_sv_f16_accum_f16_attn", &qk_int8_sv_f16_accum_f16_attn);
    m.impl("qk_int8_sv_f16_accum_f16_attn_inst_buf", &qk_int8_sv_f16_accum_f16_attn_inst_buf);
    m.impl("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", &qk_int8_sv_f16_accum_f16_fuse_v_mean_attn);
}
