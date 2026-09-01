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

import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from .quant import _fused as _quant_fused
from .quant import per_warp_int8 as per_warp_int8_cuda
from .triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

try:
    from .gfx12_native_compile import _qattn_gfx12_native
    _qattn_gfx12_prepare_attn_hnd = _qattn_gfx12_native.qk_int8_sv_f16_d64_prepare_attn_hnd
except Exception:
    _qattn_gfx12_native = None
    _qattn_gfx12_prepare_attn_hnd = None


def get_gfx12_arch_versions():
    if torch.version.hip is None:
        return []
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        arch = getattr(props, "gcnArchName", "")
        cuda_archs.append(arch.split(":", 1)[0] if arch else "")
    return cuda_archs


def _get_gfx12_native_extension():
    if _qattn_gfx12_native is None:
        raise RuntimeError(
            "The gfx12 native extension is unavailable. Rebuild SageAttention "
            "with a gfx12 ROCm target selected."
        )
    return _qattn_gfx12_native


def _try_gfx12_fp8_nhd_short_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
    fp8_value_scale_max: float,
) -> Optional[torch.Tensor]:
    if not (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype == torch.float16
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and q.dim() == 4
        and k.dim() == 4
        and v.dim() == 4
        and q.shape == k.shape == v.shape
        and q.size(1) in (512, 1024, 2048, 4096, 8192)
        and q.size(3) in (64, 128)
    ):
        return None

    gfx12_native = _get_gfx12_native_extension()
    return gfx12_native.sage_fp8_nhd_short_mha(
        q, k, v, int(is_causal), float(sm_scale), float(fp8_value_scale_max)
    )


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
    - value_dtype="fp8" supports head_dim 16, 64, 128, or 256.
    - value_dtype="fp16" supports head_dim 16, 64, 128, or 256.
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


    assert v.dtype in [torch.float16, torch.bfloat16], "gfx12 native path supports fp16/bf16 value inputs."
    value_dtype = value_dtype_normalized
    if sm_scale is None and q.dim() == 4:
        sm_scale = q.size(-1) ** -0.5

    if tensor_layout == "HND" and q.dim() == 4 and 128 < q.size(-1) <= 256:
        out_nhd = sageattn_qk_int8_pv_gfx12_native(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            tensor_layout="NHD",
            is_causal=is_causal,
            qk_quant_gran=qk_quant_gran,
            sm_scale=sm_scale,
            pv_accum_dtype=pv_accum_dtype,
            value_dtype=value_dtype,
            smooth_k=smooth_k,
            smooth_v=False,
            return_lse=False,
            **kwargs,
        )
        return _with_lse(out_nhd.transpose(1, 2).contiguous())

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

    if tensor_layout == "NHD" and smooth_k and qk_quant_gran == "per_warp":
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
        elif 128 < head_dim < 256:
            pad = 256 - head_dim
            q_nhd = F.pad(q_nhd, (0, pad))
            k_nhd = F.pad(k_nhd, (0, pad))
            v_nhd = F.pad(v_nhd, (0, pad))
            head_dim = 256

        if value_dtype == "fp16" and head_dim not in (16, 64, 128, 256):
            raise ValueError("gfx12 fp16 value path currently supports head_dim 16, 64, 128, or 256.")
        if value_dtype == "fp8" and head_dim not in (16, 64, 128, 256):
            raise ValueError("gfx12 fp8 value path currently supports head_dim 16, 64, 128, or 256.")

        use_gfx12_fp8_nhd_mha_wrapper = (
            value_dtype == "fp8"
            and input_dtype == torch.float16
            and qo_len == kv_len
            and kv_len in (512, 1024, 2048, 4096, 8192)
            and head_dim in (64, 128)
        )
        use_short_nhd_fp8_prep = (
            value_dtype == "fp8"
            and input_dtype == torch.float16
            and qo_len == kv_len
            and kv_len in (512, 1024)
            and head_dim in (64, 128)
        )
        if use_gfx12_fp8_nhd_mha_wrapper and head_dim_og in (64, 128) and h_qo == h_kv:
            out = _try_gfx12_fp8_nhd_short_mha(
                q_nhd, k_nhd, v_nhd, is_causal, float(sm_scale), fp8_value_scale_max
            )
            if out is not None:
                return _with_lse(out)
        value_native = None
        value_scale = None
        if use_short_nhd_fp8_prep:
            k_mean_flat, value_native, value_scale = (
                gfx12_native.mean_and_fp8_value_nhd_short(
                    k_nhd, v_nhd, float(fp8_value_scale_max)
                )
            )
            k_mean = k_mean_flat.unsqueeze(1)
        elif value_dtype == "fp16" and head_dim in (64, 128, 256):
            use_d64_causal_seq32_mean = (
                input_dtype == torch.float16
                and is_causal
                and head_dim == 64
                and qo_len == kv_len
                and kv_len in (2048, 4096, 8192)
            )
            if use_d64_causal_seq32_mean:
                k_mean_flat = gfx12_native.mean_nhd_d64_seq32(k_nhd)
            else:
                k_mean_flat = gfx12_native.mean_nhd(k_nhd)
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
            and input_dtype == torch.float16
            and head_dim in (64, 128, 256)
            and qk_quant_gran == "per_warp"
            and (
                not is_causal
                or (
                    qo_len == kv_len
                    and (head_dim == 256 or (qo_len % 64 == 0 and kv_len % 64 == 0))
                )
            )
        )
        if use_rawq_tail or use_rawq_f16_value:
            if is_causal and (qo_len % 64 != 0 or kv_len % 64 != 0):
                q_nhd, k_nhd, v_nhd = _pad_gfx12_nhd_sequence(
                    q_nhd, k_nhd, v_nhd, qo_len, kv_len, True, k_mean
                )
                q_attn = q_nhd
                q_out_len = q_nhd.size(1)
            else:
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
            if use_rawq_f16_value:
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
        if q_out_len != qo_len or head_dim != head_dim_og:
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
    elif 128 < head_dim < 256:
        pad = 256 - head_dim
        q_hnd = F.pad(q_hnd, (0, pad))
        k_hnd = F.pad(k_hnd, (0, pad))
        v_hnd = F.pad(v_hnd, (0, pad))
        head_dim = 256

    if value_dtype == "fp16" and head_dim not in (16, 64, 128, 256):
        raise ValueError("gfx12 fp16 value path currently supports head_dim 16, 64, 128, or 256.")
    if value_dtype == "fp8" and head_dim not in (16, 64, 128, 256):
        raise ValueError("gfx12 fp8 value path currently supports head_dim 16, 64, 128, or 256.")

    k_mean = None
    if smooth_k:
        if value_dtype == "fp16" and qk_quant_gran == "per_warp" and head_dim in (64, 128):
            k_mean = gfx12_native.mean_hnd(k_hnd).unsqueeze(2)
        else:
            k_mean = k_hnd.mean(dim=2, keepdim=True)
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

        use_rawq_hnd_f16 = (
            value_dtype == "fp16"
            and input_dtype == torch.float16
            and qk_quant_gran == "per_warp"
            and head_dim in (64, 128)
            and qo_len == kv_len
            and is_causal
            and qo_len == 512
            and q_hnd.dtype == k_hnd.dtype == v_hnd.dtype
        )
        if use_rawq_hnd_f16:
            k_int8 = torch.empty_like(k_hnd, dtype=torch.int8)
            k_scale = torch.empty(
                (k_hnd.size(0), k_hnd.size(1), (k_hnd.size(2) + 63) // 64),
                device=k_hnd.device,
                dtype=torch.float32,
            )
            _quant_fused.quant_per_block_int8_fuse_sub_mean_cuda(
                k_hnd, k_mean.squeeze(2).contiguous(), k_int8, k_scale, 64, 1
            )
            out = torch.empty_like(q_hnd, dtype=torch.float16)
            gfx12_native.qk_rawq_int8_sv_f16_native_attn(
                q_hnd, k_int8, v_hnd, out, k_scale,
                1, int(is_causal), float(sm_scale), kv_len, pv_accum_mode
            )
            out = out[..., :qo_len, :head_dim_og]
            if input_dtype != torch.float16 and out.dtype != input_dtype:
                out = out.to(input_dtype)
            if tensor_layout == "NHD":
                out = out.transpose(1, 2).contiguous()
            return _with_lse(out)

        use_smooth_hnd_f16_prep = (
            value_dtype == "fp16"
            and qk_quant_gran == "per_warp"
            and head_dim in (64, 128)
            and not is_causal
            and qo_len == kv_len
            and qo_len in (512, 1024)
            and q_hnd.dtype == k_hnd.dtype == v_hnd.dtype
        )
        value_native = None
        if use_smooth_hnd_f16_prep:
            q_int8, q_scale, k_int8, k_scale, value_native = (
                gfx12_native.prepare_qkv_hnd_smooth_f16(
                    q_hnd, k_hnd, v_hnd, k_mean.squeeze(2).contiguous()
                )
            )
        else:
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
            if value_native is None:
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


def gfx12_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: Optional[float],
    return_lse: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    fast_path_keys = {"value_dtype", "smooth_k", "qk_quant_gran", "pv_accum_dtype", "smooth_v"}
    value_dtype = kwargs.get("value_dtype", "auto")
    value_dtype = value_dtype.lower() if isinstance(value_dtype, str) else value_dtype
    gfx12_fast_common = (
        not return_lse
        and tensor_layout == "NHD"
        and set(kwargs).issubset(fast_path_keys)
        and kwargs.get("smooth_k", True)
        and kwargs.get("qk_quant_gran", "per_warp") == "per_warp"
        and not kwargs.get("smooth_v", False)
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype == torch.float16
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and q.dim() == 4
        and k.dim() == 4
        and v.dim() == 4
        and q.size(0) == k.size(0) == v.size(0)
        and q.size(1) == k.size(1) == v.size(1)
        and q.size(2) == k.size(2) == v.size(2)
        and q.size(3) == k.size(3) == v.size(3)
        and q.size(1) in (512, 1024, 2048, 4096, 8192)
        and q.size(3) in (64, 128)
    )
    if (
        gfx12_fast_common
        and value_dtype in {"auto", "fp8"}
        and kwargs.get("pv_accum_dtype", None) in {None, "fp32+fp16"}
    ):
        fast_sm_scale = float(sm_scale if sm_scale is not None else q.size(-1) ** -0.5)
        out = _try_gfx12_fp8_nhd_short_mha(
            q, k, v, is_causal, fast_sm_scale, _GFX12_FP8_VALUE_SCALE_MAX_FP32_FP16
        )
        if out is not None:
            return out
    return sageattn_qk_int8_pv_gfx12_native(
        q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
        sm_scale=sm_scale, return_lse=return_lse, **kwargs
    )
