"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import subprocess
import re

from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
from .triton.quant_per_block_varlen import per_block_int8 as per_block_int8_varlen_triton
from .triton.attn_qk_int8_per_block import forward as attn_false
from .triton.attn_qk_int8_per_block_causal import forward as attn_true
from .triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
from .triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

from .triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

try:
    from .sm80_compile import _qattn_sm80
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    from .sm89_compile import _qattn_sm89
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    from .sm90_compile import _qattn_sm90
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

try:
    from .gfx12_native_compile import _qattn_gfx12_native
    _qattn_gfx12_prepare_attn_hnd = _qattn_gfx12_native.qk_int8_sv_f16_d64_prepare_attn_hnd
    GFX12_NATIVE_ENABLED = True
except Exception:
    _qattn_gfx12_native = None
    _qattn_gfx12_prepare_attn_hnd = None
    GFX12_NATIVE_ENABLED = False

from .quant import per_block_int8 as per_block_int8_cuda
from .quant import per_warp_int8 as per_warp_int8_cuda
from .quant import sub_mean
from .quant import per_channel_fp8
from .quant import _fused as _quant_fused

from typing import Any, List, Literal, Optional, Tuple, Union
import warnings


def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major, minor
    except Exception as e:
        print("Failed to get CUDA version:", e)
    return None, None


def get_cuda_arch_versions():
    cuda_archs = []
    if torch.version.hip is not None:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            arch = getattr(props, "gcnArchName", "")
            cuda_archs.append(arch.split(":", 1)[0] if arch else "")
    else:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


def _get_gfx12_native_extension():
    global _qattn_gfx12_native, _qattn_gfx12_prepare_attn_hnd, GFX12_NATIVE_ENABLED
    if _qattn_gfx12_native is None:
        from .gfx12_native_compile import _qattn_gfx12_native as ops
        _qattn_gfx12_native = ops
        _qattn_gfx12_prepare_attn_hnd = _qattn_gfx12_native.qk_int8_sv_f16_d64_prepare_attn_hnd
        GFX12_NATIVE_ENABLED = True
    return _qattn_gfx12_native


def _get_gfx12_prepare_attn_hnd():
    _get_gfx12_native_extension()
    return _qattn_gfx12_prepare_attn_hnd


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_gfx12_hnd_sequence(
    q_hnd: torch.Tensor,
    k_hnd: torch.Tensor,
    v_hnd: torch.Tensor,
    q_len: int,
    kv_len: int,
    is_causal: bool = False,
    k_pad_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_padded_len = _round_up_to_multiple(q_len, 128)
    kv_padded_len = q_padded_len if is_causal else _round_up_to_multiple(kv_len, 64)
    q_pad_len = q_padded_len - q_len
    kv_pad_len = kv_padded_len - kv_len
    if q_pad_len > 0:
        q_hnd = F.pad(q_hnd, (0, 0, 0, q_pad_len))
    if kv_pad_len > 0:
        if k_pad_value is None:
            k_hnd = F.pad(k_hnd, (0, 0, 0, kv_pad_len))
        else:
            k_hnd = torch.cat([k_hnd, k_pad_value.expand(-1, -1, kv_pad_len, -1)], dim=2)
        v_hnd = F.pad(v_hnd, (0, 0, 0, kv_pad_len))
    return q_hnd, k_hnd, v_hnd


def _pad_gfx12_nhd_sequence(
    q_nhd: torch.Tensor,
    k_nhd: torch.Tensor,
    v_nhd: torch.Tensor,
    q_len: int,
    kv_len: int,
    is_causal: bool = False,
    k_pad_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_padded_len = _round_up_to_multiple(q_len, 128)
    kv_padded_len = q_padded_len if is_causal else _round_up_to_multiple(kv_len, 64)
    q_pad_len = q_padded_len - q_len
    kv_pad_len = kv_padded_len - kv_len
    if q_pad_len > 0:
        q_nhd = F.pad(q_nhd, (0, 0, 0, 0, 0, q_pad_len))
    if kv_pad_len > 0:
        if k_pad_value is None:
            k_nhd = F.pad(k_nhd, (0, 0, 0, 0, 0, kv_pad_len))
        else:
            k_nhd = torch.cat([k_nhd, k_pad_value.expand(-1, kv_pad_len, -1, -1)], dim=1)
        v_nhd = F.pad(v_nhd, (0, 0, 0, 0, 0, kv_pad_len))
    return q_nhd, k_nhd, v_nhd


_GFX12_FP8_VALUE_SCALE_MAX_FP32_FP16 = 2.25


def _gfx12_fp8_value_scale_hnd(v_hnd: torch.Tensor, scale_max: float) -> torch.Tensor:
    return v_hnd.abs().amax(dim=2).to(torch.float32).div(scale_max).contiguous()


def _gfx12_fp8_value_native(
    gfx12_native: Any,
    value: torch.Tensor,
    scale_max: float,
    tensor_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value_hnd = value if tensor_layout == "HND" else value.transpose(1, 2).contiguous()
    value_scale = _gfx12_fp8_value_scale_hnd(value_hnd, scale_max)
    value_native = gfx12_native.transpose_value_fp8_scaled_hnd(value_hnd, value_scale)
    return value_native, value_scale


def _gfx12_normalize_v2_options(
    value_dtype: str,
    pv_accum_dtype: Optional[str],
    smooth_v: bool,
) -> Tuple[str, str, bool, float]:
    value_dtype_normalized = value_dtype.lower()
    if value_dtype_normalized == "auto":
        value_dtype_normalized = "fp8"
    if value_dtype_normalized not in {"fp16", "fp8"}:
        raise ValueError("gfx12 native value_dtype must be 'auto', 'fp16', or 'fp8'.")
    if pv_accum_dtype is None:
        pv_accum_dtype = "fp32+fp16" if value_dtype_normalized == "fp8" else "fp32"
    if value_dtype_normalized == "fp8":
        if pv_accum_dtype not in {"fp32+fp16", "fp32", "fp32+fp32"}:
            raise ValueError("gfx12 fp8 value path supports pv_accum_dtype 'fp32+fp16', 'fp32', or 'fp32+fp32'.")
        if smooth_v and pv_accum_dtype in {"fp32+fp16", "fp32+fp32"}:
            warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
            smooth_v = False
        return value_dtype_normalized, pv_accum_dtype, smooth_v, (
            _GFX12_FP8_VALUE_SCALE_MAX_FP32_FP16 if pv_accum_dtype == "fp32+fp16" else 448.0
        )
    if pv_accum_dtype not in {"fp32", "fp16", "fp16+fp32"}:
        raise ValueError("gfx12 fp16 value path supports pv_accum_dtype 'fp32', 'fp16', or 'fp16+fp32'.")
    if smooth_v and pv_accum_dtype in {"fp32", "fp16+fp32"}:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False
    return value_dtype_normalized, pv_accum_dtype, smooth_v, _GFX12_FP8_VALUE_SCALE_MAX_FP32_FP16


def _gfx12_pv_accum_mode(value_dtype: str, pv_accum_dtype: str) -> int:
    if value_dtype != "fp16":
        return -1
    return 1 if pv_accum_dtype == "fp16" else 0


def _gfx12_apply_smooth_v(
    v: torch.Tensor,
    tensor_layout: str,
    q_heads: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    seq_dim = 1 if tensor_layout == "NHD" else 2
    head_dim = 2 if tensor_layout == "NHD" else 1
    vm = v.mean(dim=seq_dim)
    centered = (v - vm.unsqueeze(seq_dim)).to(torch.float16)
    kv_heads = v.size(head_dim)
    if q_heads % kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads.")
    if q_heads != kv_heads:
        vm = torch.repeat_interleave(vm, q_heads // kv_heads, dim=1)
    return centered, vm


def _gfx12_add_smooth_v_mean(
    out: torch.Tensor,
    vm: Optional[torch.Tensor],
    tensor_layout: str,
) -> torch.Tensor:
    if vm is None:
        return out
    if tensor_layout == "NHD":
        return out + vm.unsqueeze(1).to(out.dtype)
    return out + vm.unsqueeze(2).to(out.dtype)


def _attention_lse_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: float,
    block_q: int = 128,
    max_score_elems: int = 8 * 1024 * 1024,
) -> torch.Tensor:
    if tensor_layout == "NHD":
        q_hnd = q.transpose(1, 2)
        k_hnd = k.transpose(1, 2)
    else:
        q_hnd = q
        k_hnd = k

    bsz, num_q_heads, q_len, _ = q_hnd.shape
    _, num_kv_heads, kv_len, _ = k_hnd.shape
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads.")

    heads_per_kv = num_q_heads // num_kv_heads
    block_q = max(1, min(block_q, max_score_elems // max(1, bsz * heads_per_kv * kv_len)))
    lse = torch.empty((bsz, num_q_heads, q_len), device=q.device, dtype=torch.float32)
    q_float = q_hnd.to(torch.float32)
    k_float = k_hnd.to(torch.float32)

    for hkv in range(num_kv_heads):
        h_start = hkv * heads_per_kv
        h_stop = h_start + heads_per_kv
        k_head = k_float[:, hkv]
        for q_start in range(0, q_len, block_q):
            q_stop = min(q_start + block_q, q_len)
            scores = torch.einsum(
                "bhsd,btd->bhst",
                q_float[:, h_start:h_stop, q_start:q_stop],
                k_head,
            ).mul_(sm_scale)
            if is_causal:
                q_idx = torch.arange(q_start, q_stop, device=q.device)[:, None]
                k_idx = torch.arange(kv_len, device=q.device)[None, :]
                scores.masked_fill_(k_idx > q_idx, float("-inf"))
            lse[:, h_start:h_stop, q_start:q_stop] = torch.logsumexp(scores, dim=-1)
    return lse


def sageattn_qk_int8_pv_gfx12_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: Optional[str] = None,
    value_dtype: str = "fp8",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    ROCm gfx12 native SageAttention path.

    Supports fixed-length attention. The default smooth-K path follows the
    CUDA quantization flow; NHD inputs use native NHD quantization to avoid an
    extra layout conversion when possible.

    Current gfx12 constraints:
    - q, k, and v must be fp16 or bf16.
    - value_dtype="fp8" supports head_dim 16, 64, or 128.
    - value_dtype="fp16" supports head_dim 16, 64, or 128.
    - Causal masking requires q_len == kv_len.
    - smooth_k is enabled by default to match the CUDA and Triton paths.
    - return_lse uses an exact PyTorch logsumexp side computation and does
      not affect the default return_lse=False fast path.
    """

    if qk_quant_gran not in {"per_warp", "per_thread"}:
        raise ValueError("qk_quant_gran must be either 'per_warp' or 'per_thread'.")
    value_dtype_normalized, pv_accum_dtype, smooth_v, fp8_value_scale_max = (
        _gfx12_normalize_v2_options(value_dtype, pv_accum_dtype, smooth_v)
    )
    pv_accum_mode = _gfx12_pv_accum_mode(value_dtype_normalized, pv_accum_dtype)
    gfx12_native = _get_gfx12_native_extension()
    gfx12_prepare_attn_hnd = _qattn_gfx12_prepare_attn_hnd

    assert q.is_cuda, "Input tensors must be on cuda/HIP."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    assert q.dtype in [torch.float16, torch.bfloat16], "gfx12 native path supports fp16/bf16 inputs."
    assert tensor_layout in ["HND", "NHD"], "tensor_layout must be either 'HND' or 'NHD'."
    input_dtype = q.dtype

    if smooth_v:
        q_heads = q.size(2) if tensor_layout == "NHD" else q.size(1)
        v, smooth_v_mean = _gfx12_apply_smooth_v(v, tensor_layout, q_heads)
    else:
        smooth_v_mean = None

    lse_q = q
    lse_k = k
    lse_sm_scale = float(sm_scale if sm_scale is not None else q.size(-1) ** -0.5)

    def _with_lse(out: torch.Tensor):
        out = _gfx12_add_smooth_v_mean(out, smooth_v_mean, tensor_layout)
        if not return_lse:
            return out
        return out, _attention_lse_reference(
            lse_q, lse_k, tensor_layout, bool(is_causal), lse_sm_scale
        )

    torch.cuda.set_device(v.device)

    assert v.dtype in [torch.float16, torch.bfloat16], "gfx12 native path supports fp16/bf16 value inputs."
    value_dtype = value_dtype_normalized
    if sm_scale is None and q.dim() == 4:
        sm_scale = q.size(-1) ** -0.5

    if (
        tensor_layout == "HND"
        and not smooth_k
        and q.dim() == 4
        and k.dim() == 4
        and v.dim() == 4
        and q.dtype == k.dtype == v.dtype
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and q.size(-1) in (16, 64, 128)
        and value_dtype == "fp16"
        and q.size(-1) in (16, 64)
        and q.size(2) % 64 == 0
        and k.size(2) % 64 == 0
    ):
        use_raw_f16_value = (
            value_dtype == "fp16"
            and input_dtype == torch.float16
            and is_causal
            and q.size(-1) == 64
            and q.size(2) <= 512
        )
        out = gfx12_prepare_attn_hnd(
            q,
            k,
            v,
            int(is_causal),
            int(value_dtype == "fp8"),
            int(use_raw_f16_value),
            float(sm_scale),
            0,
            pv_accum_mode,
        )
        if input_dtype == torch.bfloat16:
            out = out if out.dtype == torch.bfloat16 else gfx12_native.convert_f16_to_bf16(out)
        return _with_lse(out)

    if tensor_layout == "NHD" and smooth_k and qk_quant_gran == "per_warp" and not (
        value_dtype == "fp16" and q.size(-1) > 64
    ):
        q_nhd = q.contiguous()
        k_nhd = k.contiguous()
        v_nhd = v.contiguous()

        _, qo_len, h_qo, head_dim_og = q_nhd.shape
        _, kv_len, h_kv, _ = k_nhd.shape
        if h_qo % h_kv != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads.")
        if is_causal and qo_len != kv_len:
            raise ValueError("gfx12 causal path currently requires q_len == kv_len.")

        head_dim = head_dim_og
        if head_dim < 64:
            pad = 64 - head_dim
            q_nhd = F.pad(q_nhd, (0, pad))
            k_nhd = F.pad(k_nhd, (0, pad))
            v_nhd = F.pad(v_nhd, (0, pad))
            head_dim = 64
        elif 64 < head_dim < 128:
            pad = 128 - head_dim
            q_nhd = F.pad(q_nhd, (0, pad))
            k_nhd = F.pad(k_nhd, (0, pad))
            v_nhd = F.pad(v_nhd, (0, pad))
            head_dim = 128

        if value_dtype == "fp16" and head_dim not in (16, 64, 128):
            raise ValueError("gfx12 fp16 value path currently supports head_dim 16, 64, or 128.")
        if value_dtype == "fp8" and head_dim not in (16, 64, 128):
            raise ValueError("gfx12 fp8 value path currently supports head_dim 16, 64, or 128.")

        use_short_nhd_fp8_prep = (
            value_dtype == "fp8"
            and not is_causal
            and input_dtype == torch.float16
            and qo_len == kv_len
            and kv_len in (512, 1024)
            and head_dim in (64, 128)
        )
        value_native = None
        value_scale = None
        if use_short_nhd_fp8_prep:
            k_mean_flat, value_native, value_scale = (
                gfx12_native.mean_and_fp8_value_nhd_short(
                    k_nhd, v_nhd, float(fp8_value_scale_max)
                )
            )
            k_mean = k_mean_flat.unsqueeze(1)
        else:
            k_mean = k_nhd.mean(dim=1, keepdim=True)
            k_mean_flat = k_mean.squeeze(1)
        use_rawq_tail = value_dtype == "fp8" and not is_causal and head_dim == 128
        use_mixed_key_hnd = value_dtype == "fp8" and (
            (
                is_causal
                and (
                    (head_dim == 64 and qo_len >= 8192)
                    or (head_dim == 128 and qo_len >= 4096)
                )
            )
        )
        use_rawq_f16_value = (
            value_dtype == "fp16"
            and not is_causal
            and head_dim == 64
            and qk_quant_gran == "per_warp"
        )
        if use_rawq_tail or use_rawq_f16_value:
            q_attn = q_nhd
            q_out_len = ((qo_len + 127) // 128) * 128 if use_rawq_tail else qo_len
            kv_pad_len = ((kv_len + 63) // 64) * 64 - kv_len
            if kv_pad_len > 0:
                k_nhd = torch.cat([k_nhd, k_mean.expand(-1, kv_pad_len, -1, -1)], dim=1)
                v_nhd = F.pad(v_nhd, (0, 0, 0, 0, 0, kv_pad_len))
        else:
            q_nhd, k_nhd, v_nhd = _pad_gfx12_nhd_sequence(
                q_nhd, k_nhd, v_nhd, qo_len, kv_len, bool(is_causal), k_mean
            )
            q_attn = q_nhd
            q_out_len = q_nhd.size(1)
        if use_mixed_key_hnd:
            k_attn = k_nhd.transpose(1, 2).contiguous()
            k_mean_attn = k_mean.transpose(1, 2).contiguous()
            k_int8 = torch.empty_like(k_attn, dtype=torch.int8)
            k_scale = torch.empty(
                (k_attn.size(0), k_attn.size(1), (k_attn.size(2) + 63) // 64),
                device=k_attn.device,
                dtype=torch.float32,
            )
            _quant_fused.quant_per_block_int8_fuse_sub_mean_cuda(
                k_attn, k_mean_attn.squeeze(2), k_int8, k_scale, 64, 1
            )
        else:
            k_int8 = torch.empty_like(k_nhd, dtype=torch.int8)
            k_scale = torch.empty(
                (k_nhd.size(0), k_nhd.size(2), (k_nhd.size(1) + 63) // 64),
                device=k_nhd.device,
                dtype=torch.float32,
            )
            _quant_fused.quant_per_block_int8_fuse_sub_mean_cuda(
                k_nhd, k_mean_flat, k_int8, k_scale, 64, 0
            )
        if value_dtype == "fp8":
            if value_native is None:
                value_native, value_scale = _gfx12_fp8_value_native(
                    gfx12_native, v_nhd, fp8_value_scale_max, "NHD"
                )
        else:
            value_native = v_nhd if input_dtype == torch.float16 else v_nhd.to(torch.float16)
        out = torch.empty(
            (q_nhd.size(0), q_out_len, q_nhd.size(2), q_nhd.size(3)),
            device=q_nhd.device,
            dtype=torch.float16,
        )
        if value_dtype == "fp8":
            gfx12_native.qk_rawq_int8_sv_f8_scaled_native_attn(
                q_attn,
                k_int8,
                value_native,
                out,
                k_scale,
                value_scale,
                0,
                int(is_causal),
                float(sm_scale),
                kv_len,
                1,
                int(use_mixed_key_hnd),
            )
        else:
            if head_dim == 64 and qk_quant_gran == "per_warp":
                gfx12_native.qk_rawq_int8_sv_f16_native_attn(
                    q_attn,
                    k_int8,
                    value_native,
                    out,
                    k_scale,
                    0,
                    int(is_causal),
                    float(sm_scale),
                    kv_len,
                    pv_accum_mode,
                )
            else:
                q_int8, q_scale = gfx12_native.quant_q_nhd_per_warp(q_attn)
                gfx12_native.qk_int8_sv_f16_d64_native_attn(
                    q_int8,
                    k_int8,
                    value_native,
                    out,
                    q_scale,
                    k_scale,
                    0,
                    int(is_causal),
                    float(sm_scale),
                    kv_len,
                    0,
                    pv_accum_mode,
                )
        out = out[:, :qo_len, :, :head_dim_og]
        if input_dtype == torch.bfloat16 and out.dtype != torch.bfloat16:
            out = gfx12_native.convert_f16_to_bf16(out.contiguous() if not out.is_contiguous() else out)
        elif input_dtype != torch.float16:
            out = out.to(input_dtype)
        return _with_lse(out)

    if tensor_layout == "NHD":
        q_hnd = q.transpose(1, 2).contiguous()
        k_hnd = k.transpose(1, 2).contiguous()
        v_hnd = v.transpose(1, 2).contiguous()
    else:
        q_hnd = q.contiguous()
        k_hnd = k.contiguous()
        v_hnd = v.contiguous()

    _, h_qo, qo_len, head_dim_og = q_hnd.shape
    _, h_kv, kv_len, _ = k_hnd.shape
    if h_qo % h_kv != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads.")
    if is_causal and qo_len != kv_len:
        raise ValueError("gfx12 causal path currently requires q_len == kv_len.")

    head_dim = head_dim_og
    if head_dim < 64 and (
        smooth_k or head_dim != 16 or value_dtype == "fp8" or q_hnd.dtype != v_hnd.dtype
    ):
        pad = 64 - head_dim
        q_hnd = F.pad(q_hnd, (0, pad))
        k_hnd = F.pad(k_hnd, (0, pad))
        v_hnd = F.pad(v_hnd, (0, pad))
        head_dim = 64
    elif 64 < head_dim < 128:
        pad = 128 - head_dim
        q_hnd = F.pad(q_hnd, (0, pad))
        k_hnd = F.pad(k_hnd, (0, pad))
        v_hnd = F.pad(v_hnd, (0, pad))
        head_dim = 128

    if value_dtype == "fp16" and head_dim not in (16, 64, 128):
        raise ValueError("gfx12 fp16 value path currently supports head_dim 16, 64, or 128.")
    if value_dtype == "fp8" and head_dim not in (16, 64, 128):
        raise ValueError("gfx12 fp8 value path currently supports head_dim 16, 64, or 128.")

    k_mean = k_hnd.mean(dim=2, keepdim=True) if smooth_k else None
    q_hnd, k_hnd, v_hnd = _pad_gfx12_hnd_sequence(
        q_hnd, k_hnd, v_hnd, qo_len, kv_len, bool(is_causal), k_mean)
    padded_qo_len = q_hnd.size(2)

    use_raw_f16_value = (
        value_dtype == "fp16"
        and input_dtype == torch.float16
        and is_causal
        and head_dim == 64
        and padded_qo_len <= 512
    )

    def _quant_qk_hnd(q_src: torch.Tensor, k_src: torch.Tensor, km_src: Optional[torch.Tensor]):
        if qk_quant_gran == "per_thread":
            return per_thread_int8_triton(
                q_src, k_src, km_src, BLKQ=128,
                WARPQ=(16 if (head_dim == 128 and pv_accum_dtype == "fp16+fp32") else 32),
                BLKK=64, WARPK=64, tensor_layout="HND"
            )
        return per_warp_int8_cuda(
            q_src, k_src, km_src, BLKQ=128, WARPQ=32, BLKK=64, tensor_layout="HND"
        )

    if not smooth_k:
        if value_dtype == "fp8":
            q_int8, q_scale, k_int8, k_scale = _quant_qk_hnd(q_hnd, k_hnd, None)
            value_native, value_scale = _gfx12_fp8_value_native(
                gfx12_native, v_hnd, fp8_value_scale_max, "HND"
            )
            out = torch.empty_like(q_hnd, dtype=torch.float16)
            gfx12_native.qk_int8_sv_f8_scaled_native_attn(
                q_int8, k_int8, value_native, out, q_scale, k_scale, value_scale,
                1, int(is_causal), float(sm_scale), kv_len
            )
        else:
            if qk_quant_gran == "per_warp" and q_hnd.dtype == k_hnd.dtype == v_hnd.dtype:
                out = gfx12_prepare_attn_hnd(
                    q_hnd,
                    k_hnd,
                    v_hnd,
                    int(is_causal),
                    0,
                    int(use_raw_f16_value),
                    float(sm_scale),
                    kv_len,
                    pv_accum_mode,
                )
            else:
                q_int8, q_scale, k_int8, k_scale = _quant_qk_hnd(q_hnd, k_hnd, None)
                value_native = gfx12_native.transpose_value_f16_hnd(v_hnd)
                out = torch.empty_like(q_hnd, dtype=torch.float16)
                gfx12_native.qk_int8_sv_f16_d64_native_attn(
                    q_int8, k_int8, value_native, out, q_scale, k_scale,
                    1, int(is_causal), float(sm_scale), kv_len, 1,
                    pv_accum_mode
                )
    else:
        use_rawq_hnd_fp8 = (
            value_dtype == "fp8"
            and head_dim in (64, 128)
            and (
                not is_causal
                or head_dim == 64
                or padded_qo_len <= 1024
                or padded_qo_len >= 8192
            )
        )
        if use_rawq_hnd_fp8 and qk_quant_gran == "per_warp":
            k_int8 = torch.empty_like(k_hnd, dtype=torch.int8)
            k_scale = torch.empty(
                (k_hnd.size(0), k_hnd.size(1), (k_hnd.size(2) + 63) // 64),
                device=k_hnd.device,
                dtype=torch.float32,
            )
            _quant_fused.quant_per_block_int8_fuse_sub_mean_cuda(
                k_hnd, k_mean.squeeze(2), k_int8, k_scale, 64, 1
            )
            value_native, value_scale = _gfx12_fp8_value_native(
                gfx12_native, v_hnd, fp8_value_scale_max, "HND"
            )
            out = torch.empty_like(
                q_hnd,
                dtype=torch.bfloat16 if input_dtype == torch.bfloat16 else torch.float16,
            )
            gfx12_native.qk_rawq_int8_sv_f8_scaled_native_attn(
                q_hnd, k_int8, value_native, out, k_scale, value_scale,
                1, int(is_causal), float(sm_scale), kv_len, 1
            )
            out = out[..., :qo_len, :head_dim_og]
            if input_dtype != torch.float16 and out.dtype != input_dtype:
                out = out.to(input_dtype)
            if tensor_layout == "NHD":
                out = out.transpose(1, 2).contiguous()
            return _with_lse(out)

        q_int8, q_scale, k_int8, k_scale = _quant_qk_hnd(q_hnd, k_hnd, k_mean)
        out = torch.empty_like(q_hnd, dtype=torch.float16)
        if value_dtype == "fp8":
            value_native, value_scale = _gfx12_fp8_value_native(
                gfx12_native, v_hnd, fp8_value_scale_max, "HND"
            )
            gfx12_native.qk_int8_sv_f8_scaled_native_attn(
                q_int8, k_int8, value_native, out, q_scale, k_scale, value_scale,
                1, int(is_causal), float(sm_scale), kv_len
            )
        else:
            value_native = gfx12_native.transpose_value_f16_hnd(v_hnd)
            gfx12_native.qk_int8_sv_f16_d64_native_attn(
                q_int8, k_int8, value_native, out, q_scale, k_scale,
                1, int(is_causal), float(sm_scale), kv_len, 1,
                pv_accum_mode
            )
    out = out[..., :qo_len, :head_dim_og]
    if input_dtype == torch.bfloat16 and out.dtype != torch.bfloat16:
        out = gfx12_native.convert_f16_to_bf16(out.contiguous() if not out.is_contiguous() else out)
    elif input_dtype != torch.float16:
        out = out.to(input_dtype)
    if tensor_layout == "NHD":
        out = out.transpose(1, 2).contiguous()
    return _with_lse(out)


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """

    arch = get_cuda_arch_versions()[q.device.index]
    if arch.startswith("gfx12"):
        return sageattn_qk_int8_pv_gfx12_native(
            q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
            sm_scale=sm_scale, return_lse=return_lse, **kwargs)
    if arch == "sm80":
        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif arch == "sm86":
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse)
    elif arch == "sm89":
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16")
    elif arch == "sm90":
        return sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    elif arch == "sm120":
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16") # sm120 has accurate fp32 accumulator for fp8 mma and triton kernel is currently not usable on sm120.
    elif arch == "sm121":
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp16") # sm121 has accurate fp32 accumulator for fp8 mma and triton kernel is currently not usable on sm121.
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool =False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 quantization for Q and K, FP16 PV with FP16 accumulation, implemented using Triton.
    The FP16 accumulator is added to a FP32 buffer immediately after each iteration.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    quantization_backend : str
        The quantization backend, either "triton" or "cuda".
        "cuda" backend offers better performance due to kernel fusion.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    attn_mask : Optional[torch.Tensor]
        The attention mask tensor, of dtype bool or float32.
        Should be able to broadcast to the shape of the matrix qk^T.
        Default: None.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype, "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."

    # FIXME(DefTruth): make sage attention work compatible with distributed
    # env, for example, xDiT which launch by torchrun. Without this workaround,
    # sage attention will run into illegal memory access error after first
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    elif quantization_backend == "cuda":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")
    if is_causal:
        assert attn_mask is None, "Mask should be None for causal attention."
        o, lse = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse)
    else:
        if attn_mask is not None:
            if tensor_layout == "HND":
                target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
            elif tensor_layout == "NHD":
                target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
            else:
                raise ValueError(f"tensor_layout {tensor_layout} not supported")
            try:
                attn_mask = attn_mask.expand(target_shape)
            except Exception:
                raise AssertionError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}")
        o, lse = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=attn_mask, return_lse=return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`.
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`.
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.

    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed
    # env, for example, xDiT which launch by torchrun. Without this workaround,
    # sage attention will run into illegal memory access error after first
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.mean(dim=0, keepdim=True) # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen_triton(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale)

    if is_causal:
        o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    else:
        o = attn_false_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)

    o = o[..., :head_dim_og]

    return o


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP16 PV with FP16/FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp16", "fp16+fp32" or "fp32".
        - "fp16": PV accumulation is done in fully in FP16. This is the fastest option but may lead to numerical instability. `smooth_v` option will increase the accuracy in cases when the value tensor has a large bias (like in CogVideoX-2b).
        - "fp32": PV accumulation is done in FP32. This is the most accurate option but may be slower than "fp16" due to CUDA core overhead.
        - "fp16+fp32": PV accumulation is done in FP16, but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32" or "fp16+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert SM80_ENABLED, "SM80 kernel is not available. make sure you GPUs with compute capability 8.0 or higher."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed
    # env, for example, xDiT which launch by torchrun. Without this workaround,
    # sage attention will run into illegal memory access error after first
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64, WARPK=64)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32':
        v = v.to(torch.float16)
        lse = sm80_compile.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = sm80_compile.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            v = v.to(torch.float16)
            lse = sm80_compile.qk_int8_sv_f16_accum_f16_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16+fp32":
        v = v.to(torch.float16)
        lse = sm80_compile.qk_int8_sv_f16_accum_f16_attn_inst_buf(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # cuda_major_version, cuda_minor_version = get_cuda_version()
    # if(cuda_major_version, cuda_minor_version) < (12, 8) and pv_accum_dtype == 'fp32+fp16':
    #     warnings.warn("cuda version < 12.8, change pv_accum_dtype to 'fp32+fp32'")
    #     pv_accum_dtype = 'fp32+fp32'

    # FIXME(DefTruth): make sage attention work compatible with distributed
    # env, for example, xDiT which launch by torchrun. Without this workaround,
    # sage attention will run into illegal memory access error after first
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32+fp16' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored.")
        smooth_v = False

    quant_v_scale_max = 448.0
    if pv_accum_dtype == 'fp32+fp16':
        quant_v_scale_max = 2.25

    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v)

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp16":
        lse = sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert SM90_ENABLED, "SM90 kernel is not available. Make sure you GPUs with compute capability 9.0."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    # pad v to multiple of 128
    # TODO: modify per_channel_fp8 kernel to handle this
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
        else:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    if pv_accum_dtype == "fp32":
        raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
        lse = sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
