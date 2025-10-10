import torch

from . import _qattn_sm89
_qattn_sm89 = torch.ops.sageattention_qattn_sm89


def sm89_qk_with_key_value(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    output: torch.Tensor, 
    query_scale: torch.Tensor, 
    key_scale: torch.Tensor, 
    value_scale: torch.Tensor, 
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    batch_size = query.size(0)

    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    else:
        lse = torch.empty((0,), dtype=torch.float32, device=query.device)
    return lse


torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn")(sm89_qk_with_key_value)
torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf")(sm89_qk_with_key_value)
torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf")(sm89_qk_with_key_value)


@torch.library.register_fake("sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn")
def sm89_qk_with_key_value_mean(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    output: torch.Tensor, 
    query_scale: torch.Tensor, 
    key_scale: torch.Tensor, 
    value_scale: torch.Tensor, 
    value_mean: torch.Tensor,
    tensor_layout: int, 
    is_causal: int, 
    qk_quant_gran: int, 
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return sm89_qk_with_key_value(
        query, key, value, output, query_scale, key_scale, value_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )
