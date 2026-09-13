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

import os
from typing import Any, Optional, Tuple, Union

import torch

# Backend selection via environment variable:
#   SAGEATTN_BACKEND=native - HIP native WMMA kernel (default for gfx103x/gfx110x)
#   SAGEATTN_BACKEND=triton - Triton autotune kernel (upstream Triton path)
_BACKEND = os.getenv("SAGEATTN_BACKEND", "native").lower()

# The two native extensions (_qattn_gfx110x / _qattn_gfx103x) are registered
# under separate torch.ops namespaces, so unlike the downstream fork both can
# be built into the same wheel without schema clashes. We still lazily import
# only the module that matches the current HIP device so that importing the
# package does not eagerly pull in an extension built for another architecture.
_qattn_ops = None
GFX_NATIVE_ENABLED = False
GFX_ARCH_LOADED = None
_import_error = None


def _detect_gfx_arch():
    """Return 'gfx110x' (RDNA3) or 'gfx103x' (RDNA2) for the current HIP device, or None."""
    try:
        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev)
        name = getattr(prop, "gcnArchName", None) or getattr(prop, "name", "")
        if isinstance(name, str) and name.startswith("gfx"):
            if name.startswith("gfx11"):
                return "gfx110x"
            if name.startswith("gfx103"):
                return "gfx103x"
            return None
        mj = getattr(prop, "major", None)
        if mj == 11:
            return "gfx110x"
        if mj == 10:
            return "gfx103x"
        return None
    except Exception:
        return None


def _get_native_ops():
    """Import the native extension for the current arch and return its ops namespace."""
    global _qattn_ops, GFX_NATIVE_ENABLED, GFX_ARCH_LOADED, _import_error
    if _qattn_ops is not None:
        return _qattn_ops
    arch = _detect_gfx_arch()
    if arch is None:
        raise RuntimeError(
            "sageattention native extension: cannot determine AMD gfx arch of current device. "
            "Supported: gfx110x (RDNA3), gfx103x (RDNA2). "
            "Use backend='triton' or set SAGEATTN_BACKEND=triton."
        )
    try:
        if arch == "gfx110x":
            from . import gfx11x_native_compile  # noqa: F401
            _qattn_ops = torch.ops.sageattention_qattn_gfx110x
        else:
            from . import gfx10x_native_compile  # noqa: F401
            _qattn_ops = torch.ops.sageattention_qattn_gfx103x
        GFX_NATIVE_ENABLED = True
        GFX_ARCH_LOADED = arch
    except Exception as e:
        _import_error = e
        raise RuntimeError(
            f"sageattention native extension (_qattn_{arch}) is not available. "
            f"Build with GPU_ARCHS containing an {arch[:-1]}* arch, e.g.\n"
            "  GPU_ARCHS=gfx1103 pip install -e . --no-build-isolation\n"
            "on a ROCm/HIP system.\nOriginal error: {e}"
        ) from e
    return _qattn_ops


def gfx10x_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    sm_scale: Optional[float],
    return_lse: bool,
    kwargs: dict[str, Any],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """ROCm gfx103x / gfx110x native SageAttention path (ported from the RDNA3 fork)."""
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on CUDA/HIP device."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "Input tensors must be fp16, bf16, or fp32."
    )
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim in [64, 128], f"head_dim must be 64 or 128, got {headdim}."

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, (
        "Last dim of qkv must be contiguous."
    )

    if sm_scale is None:
        sm_scale = headdim ** -0.5

    if _BACKEND == "triton":
        from .core import sageattn_qk_int8_pv_fp16_triton
        return sageattn_qk_int8_pv_fp16_triton(
            q, k, v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
            **kwargs,
        )

    input_dtype = dtype
    if dtype == torch.float32:
        v = v.to(torch.float16)
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        dtype = torch.float16

    ops = _get_native_ops()

    # Write-back uses 16B vector stores (head_dim is always a multiple of 8), so
    # q's batch/head/seq strides must be multiples of 8 halfs. Non-contiguous q
    # (permute/slice) would cause misaligned 16B stores (UB). k/v only need the
    # head_dim stride == 1 (asserted above).
    assert q.stride(0) % 8 == 0 and q.stride(1) % 8 == 0 and q.stride(2) % 8 == 0, (
        "native backend requires q strides that are multiples of 8 halfs "
        "(16B aligned write-back). "
        f"Got strides={q.stride()}. Use contiguous tensors."
    )

    layout_code = 1 if tensor_layout == "HND" else 0

    # bm_sel controls the direct kernel's BM (0=default, 1=32, 2=128); only used on gfx1103.
    bm_sel = int(kwargs.get("bm_sel", 0) or os.getenv("SAGEATTN_BM_SEL", "0"))

    o = torch.empty_like(q)

    if tensor_layout == "HND":
        kv_len_actual = k.size(2)
        q_len = q.size(2)
    else:
        kv_len_actual = k.size(1)
        q_len = q.size(1)

    # gfx1035 (RDNA2) always uses int8 (direct is 4-65x slower than int8 on all
    # shapes); gfx1103 keeps the direct/int8 balance.
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", None)
    is_gfx103 = isinstance(arch, str) and arch.startswith("gfx103")
    if headdim == 64:
        if is_gfx103:
            use_direct = False
        else:
            d64_default = 2048 if tensor_layout == "HND" else 3072
            thr_d64 = int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64", str(d64_default)) or d64_default)
            if is_causal:
                use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64_CAUSAL", "6144") or 6144))
            elif q_len < kv_len_actual:
                use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D64_CROSS", "6144") or 6144))
            else:
                use_direct = (kv_len_actual <= thr_d64)
    else:
        if is_gfx103:
            use_direct = False
        else:
            thr_d128 = int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D128", "2048") or 2048)
            if q_len * 2 < kv_len_actual:
                use_direct = (kv_len_actual <= int(os.getenv("SAGEATTN_DIRECT_THRESHOLD_D128_CROSS", "4096") or 4096))
            else:
                use_direct = (kv_len_actual <= thr_d128)

    if use_direct:
        # V is handed directly to v_transpose: bf16 is converted to fp16 inside the
        # kernel (saves a standalone cast kernel). V_T [B,H,D,N] has its n dimension
        # padded to a multiple of 64 with zeros to prevent the attn kernel's 32B
        # v_frag_t read from going out of bounds.
        v_attn = v
        kv_heads_n = k.size(1) if tensor_layout == "HND" else k.size(2)
        padded_n = ((kv_len_actual + 63) // 64) * 64
        v_t = torch.empty(
            q.size(0), kv_heads_n, headdim, padded_n,
            device=q.device, dtype=torch.float16
        )
        ops.v_transpose(v_attn, v_t, layout_code)
        v_attn = v_t
        if input_dtype == torch.bfloat16:
            ops.bf16_attn_t(
                q, k, v_attn, o,
                layout_code, int(is_causal), sm_scale, bm_sel
            )
        else:
            ops.fp16_attn_t(
                q, k, v_attn, o,
                layout_code, int(is_causal), sm_scale, bm_sel
            )
    else:
        # int8 path: gfx103x reads the native [B,H,N,D] V directly; gfx110x reads
        # V_T [B,H,D,N] and needs a global transpose first.
        v_native = is_gfx103
        kv_heads_n = k.size(1) if tensor_layout == "HND" else k.size(2)
        if v_native:
            v_for_attn = v
            if tensor_layout == "NHD":
                v16 = v if v.dtype == torch.float16 else v.to(torch.float16)
                if v16.is_contiguous():
                    b_, n_, h_, d_ = v16.shape
                    v_for_attn = v16.as_strided((b_, h_, n_, d_), (n_ * h_ * d_, d_, h_ * d_, 1))
                else:
                    v_for_attn = v16.permute(0, 2, 1, 3).contiguous()
            elif v.dtype != torch.float16:
                v_for_attn = v.to(torch.float16)
            # v_scale placeholder (gfx103x reads native fp16 V and does not read
            # v_scale; it only satisfies the pybind signature).
            v_scale_t = torch.empty(
                q.size(0), kv_heads_n, (kv_len_actual + 31) // 32,
                device=q.device, dtype=torch.float32
            )
        else:
            # gfx110x: bf16 input is converted to fp16 inside v_transpose; pass bf16
            # directly (saves a .to(fp16) cast kernel).
            padded_n = ((kv_len_actual + 63) // 64) * 64
            v_t = torch.empty(
                q.size(0), kv_heads_n, headdim, padded_n,
                device=q.device, dtype=torch.float16
            )
            ops.v_transpose(v, v_t, layout_code)
            v_for_attn = v_t
            # v_scale placeholder (kernel does not read it); only satisfies the pybind signature.
            v_scale_t = torch.empty(
                q.size(0), kv_heads_n, (kv_len_actual + 31) // 32,
                device=q.device, dtype=torch.float32
            )
        o_int8 = o

        # smooth_k: K is mean-subtracted before quantization; default False.
        smooth_k = kwargs.get("smooth_k", False)
        if smooth_k:
            k_mean = ops.mean_seq(k, layout_code)
        else:
            k_mean = torch.empty(0, device=q.device, dtype=q.dtype)
        # v10 INQ (in-kernel Q int8 quant): for small kv the prepass Q-quantization
        # round-trip dominates end-to-end time; the main kernel quantizes Q from the
        # fp16/bf16 source in-kernel and the prepass skips Q. Only gfx103x supports
        # this (gfx110x's main kernel ignores q_fp and must use the prepass q_int8).
        # Disabling INQ worsens non-causal small-kv accuracy.
        q_skip_inq = False
        q_fp = q
        if is_gfx103:
            v10_on = True
            ipv_on = False
            inq_wanted = (kv_len_actual <= 1024)
            q_skip_inq = bool(
                headdim in (64, 128) and not is_causal and v10_on and not ipv_on and inq_wanted
                and q.is_contiguous()
            )
            if q_skip_inq and tensor_layout == "NHD":
                b_, s_, h_, d_ = q.shape
                q_fp = q.as_strided((b_, h_, s_, d_), (s_ * h_ * d_, d_, h_ * d_, 1))
        q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
            q, k, k_mean, layout_code, sm_scale, int(q_skip_inq)
        )
        # Signature includes v_scale and q_fp; both extensions rebuild against this
        # signature (schema must match).
        ops.qk_int8_sv_bf16_attn_t(
            q_int8, k_int8, v_for_attn, o_int8,
            q_scale, k_scale, v_scale_t,
            layout_code, int(is_causal), sm_scale,
            q_fp if is_gfx103 else torch.empty(0, device=q.device, dtype=q.dtype)
        )

    if input_dtype == torch.float32:
        o = o.to(torch.float32)

    if return_lse:
        # LSE is not computed by these kernels; return a zero placeholder.
        seq_dim = 2 if tensor_layout == "HND" else 1
        seq_len = q.size(seq_dim)
        lse = torch.zeros(
            (q.size(0), q.size(1 if tensor_layout == "HND" else 2), seq_len),
            dtype=torch.float32, device=q.device
        )
        return o, lse

    return o


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return gfx10x_sageattn(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        **kwargs,
    )