import torch

from . import _qattn_gfx12_native
_qattn_gfx12_native = torch.ops.sageattention_qattn_gfx12_native


def _empty_lse(query: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=query.device)


@torch.library.register_fake("sageattention_qattn_gfx12_native::qk_int8_sv_f16_d64_native_attn")
def qk_int8_sv_f16_d64_native_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    sm_scale: float,
    valid_kv_len: int = 0,
    value_transposed_hnd: int = -1,
    pv_accum_mode: int = -1,
) -> torch.Tensor:
    return _empty_lse(query)


@torch.library.register_fake("sageattention_qattn_gfx12_native::qk_rawq_int8_sv_f8_native_attn")
def qk_rawq_int8_sv_f8_native_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    sm_scale: float,
    valid_kv_len: int = 0,
    value_transposed_hnd: int = -1,
    key_hnd_layout: int = 0,
) -> torch.Tensor:
    return output


@torch.library.register_fake("sageattention_qattn_gfx12_native::qk_rawq_int8_sv_f16_native_attn")
def qk_rawq_int8_sv_f16_native_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    sm_scale: float,
    valid_kv_len: int = 0,
    pv_accum_mode: int = -1,
) -> torch.Tensor:
    return _empty_lse(query)


@torch.library.register_fake("sageattention_qattn_gfx12_native::qk_int8_sv_f8_scaled_native_attn")
def qk_int8_sv_f8_scaled_native_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    sm_scale: float,
    valid_kv_len: int = 0,
) -> torch.Tensor:
    return _empty_lse(query)


@torch.library.register_fake("sageattention_qattn_gfx12_native::qk_rawq_int8_sv_f8_scaled_native_attn")
def qk_rawq_int8_sv_f8_scaled_native_attn_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    sm_scale: float,
    valid_kv_len: int = 0,
    value_transposed_hnd: int = -1,
    key_hnd_layout: int = 0,
) -> torch.Tensor:
    return output


@torch.library.register_fake("sageattention_qattn_gfx12_native::qk_int8_sv_f16_d64_prepare_attn_hnd")
def qk_int8_sv_f16_d64_prepare_attn_hnd_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: int,
    value_is_fp8: int,
    use_raw_f16_value: int,
    sm_scale: float,
    valid_kv_len: int = 0,
    pv_accum_mode: int = -1,
) -> torch.Tensor:
    dtype = torch.bfloat16 if value_is_fp8 and query.dtype == torch.bfloat16 else torch.float16
    return torch.empty(query.shape, dtype=dtype, device=query.device)


@torch.library.register_fake("sageattention_qattn_gfx12_native::quant_q_nhd_per_warp")
def quant_q_nhd_per_warp_fake_impl(query: torch.Tensor) -> list[torch.Tensor]:
    batch, q_len, q_heads, _ = query.shape
    q_scale_groups = ((q_len + 127) // 128) * 4
    return [
        torch.empty_like(query, dtype=torch.int8),
        torch.empty((batch, q_heads, q_scale_groups), dtype=torch.float32, device=query.device),
    ]


@torch.library.register_fake("sageattention_qattn_gfx12_native::transpose_value_fp8_hnd")
def transpose_value_fp8_hnd_fake_impl(value: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (value.size(0), value.size(1), value.size(3), value.size(2)),
        dtype=torch.uint8,
        device=value.device,
    )


@torch.library.register_fake("sageattention_qattn_gfx12_native::transpose_value_fp8_scaled_hnd")
def transpose_value_fp8_scaled_hnd_fake_impl(value: torch.Tensor, value_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (value.size(0), value.size(1), value.size(3), value.size(2)),
        dtype=torch.uint8,
        device=value.device,
    )


@torch.library.register_fake("sageattention_qattn_gfx12_native::fp8_value_nhd_short")
def fp8_value_nhd_short_fake_impl(value: torch.Tensor, scale_max: float) -> list[torch.Tensor]:
    batch, seq_len, heads, head_dim = value.shape
    return [
        torch.empty((batch, heads, head_dim, seq_len), dtype=torch.uint8, device=value.device),
        torch.empty((batch, heads, head_dim), dtype=torch.float32, device=value.device),
    ]


@torch.library.register_fake("sageattention_qattn_gfx12_native::mean_nhd")
def mean_nhd_fake_impl(input: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (input.size(0), input.size(2), input.size(3)),
        dtype=input.dtype,
        device=input.device,
    )


@torch.library.register_fake("sageattention_qattn_gfx12_native::mean_and_fp8_value_nhd_short")
def mean_and_fp8_value_nhd_short_fake_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    scale_max: float,
) -> list[torch.Tensor]:
    batch, seq_len, heads, head_dim = value.shape
    return [
        torch.empty((batch, heads, head_dim), dtype=key.dtype, device=key.device),
        torch.empty((batch, heads, head_dim, seq_len), dtype=torch.uint8, device=value.device),
        torch.empty((batch, heads, head_dim), dtype=torch.float32, device=value.device),
    ]


@torch.library.register_fake("sageattention_qattn_gfx12_native::transpose_value_f16_hnd")
def transpose_value_f16_hnd_fake_impl(value: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (value.size(0), value.size(1), value.size(3), value.size(2)),
        dtype=torch.float16,
        device=value.device,
    )


@torch.library.register_fake("sageattention_qattn_gfx12_native::convert_f16_to_bf16")
def convert_f16_to_bf16_fake_impl(input: torch.Tensor) -> torch.Tensor:
    return torch.empty(input.shape, dtype=torch.bfloat16, device=input.device)
