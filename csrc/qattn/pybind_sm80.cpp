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
#include <torch/csrc/stable/library.h>

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

void boxed_qk_int8_sv_f16_accum_f32_attn(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto query = to<Tensor>(stack[0]);
    auto key = to<Tensor>(stack[1]);
    auto value = to<Tensor>(stack[2]);
    auto output = to<Tensor>(stack[3]);
    auto query_scale = to<Tensor>(stack[4]);
    auto key_scale = to<Tensor>(stack[5]);
    auto tensor_layout = to<int64_t>(stack[6]);
    auto is_causal = to<int64_t>(stack[7]);
    auto qk_quant_gran = to<int64_t>(stack[8]);
    auto sm_scale = to<double>(stack[9]);
    auto return_lse = to<int64_t>(stack[10]);

    auto lse = qk_int8_sv_f16_accum_f32_attn(query, key, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);

    stack[0] = from(lse);
}

void boxed_qk_int8_sv_f16_accum_f16_attn(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto query = to<Tensor>(stack[0]);
    auto key = to<Tensor>(stack[1]);
    auto value = to<Tensor>(stack[2]);
    auto output = to<Tensor>(stack[3]);
    auto query_scale = to<Tensor>(stack[4]);
    auto key_scale = to<Tensor>(stack[5]);
    auto tensor_layout = to<int64_t>(stack[6]);
    auto is_causal = to<int64_t>(stack[7]);
    auto qk_quant_gran = to<int64_t>(stack[8]);
    auto sm_scale = to<double>(stack[9]);
    auto return_lse = to<int64_t>(stack[10]);

    auto lse = qk_int8_sv_f16_accum_f16_attn(query, key, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);

    stack[0] = from(lse);
}

void boxed_qk_int8_sv_f16_accum_f16_attn_inst_buf(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto query = to<Tensor>(stack[0]);
    auto key = to<Tensor>(stack[1]);
    auto value = to<Tensor>(stack[2]);
    auto output = to<Tensor>(stack[3]);
    auto query_scale = to<Tensor>(stack[4]);
    auto key_scale = to<Tensor>(stack[5]);
    auto tensor_layout = to<int64_t>(stack[6]);
    auto is_causal = to<int64_t>(stack[7]);
    auto qk_quant_gran = to<int64_t>(stack[8]);
    auto sm_scale = to<double>(stack[9]);
    auto return_lse = to<int64_t>(stack[10]);

    auto lse = qk_int8_sv_f16_accum_f16_attn_inst_buf(query, key, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);

    stack[0] = from(lse);
}

void boxed_qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto query = to<Tensor>(stack[0]);
    auto key = to<Tensor>(stack[1]);
    auto value = to<Tensor>(stack[2]);
    auto output = to<Tensor>(stack[3]);
    auto query_scale = to<Tensor>(stack[4]);
    auto key_scale = to<Tensor>(stack[5]);
    auto value_mean = to<Tensor>(stack[6]);
    auto tensor_layout = to<int64_t>(stack[7]);
    auto is_causal = to<int64_t>(stack[8]);
    auto qk_quant_gran = to<int64_t>(stack[9]);
    auto sm_scale = to<double>(stack[10]);
    auto return_lse = to<int64_t>(stack[11]);

    auto lse = qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(query, key, value, output, query_scale, key_scale, value_mean, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);

    stack[0] = from(lse);
}

// Defines the operators
STABLE_TORCH_LIBRARY(sageattention_qattn_sm80, m) {
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
STABLE_TORCH_LIBRARY_IMPL(sageattention_qattn_sm80, CUDA, m) {
    m.impl("qk_int8_sv_f16_accum_f32_attn", &boxed_qk_int8_sv_f16_accum_f32_attn);
    m.impl("qk_int8_sv_f16_accum_f16_attn", &boxed_qk_int8_sv_f16_accum_f16_attn);
    m.impl("qk_int8_sv_f16_accum_f16_attn_inst_buf", &boxed_qk_int8_sv_f16_accum_f16_attn_inst_buf);
    m.impl("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", &boxed_qk_int8_sv_f16_accum_f16_fuse_v_mean_attn);
}
