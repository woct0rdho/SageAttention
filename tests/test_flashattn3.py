#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from flash_attn_interface import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_sageattn import get_rtol_atol


def main():
    batch_size = 4
    head_num = 32
    seq_len = 64
    head_dim = 128
    dtype = torch.float16

    q = torch.randn(batch_size, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    print("q", tuple(q.shape), q.device, q.dtype)

    # 'Mathematically correct' implementation
    torch.backends.cuda.enable_math_sdp(True)
    with sdpa_kernel(SDPBackend.MATH):
        out_math = F.scaled_dot_product_attention(q, k, v)

    # FlashAttention expects (batch_size, seq_len, head_num, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out_fa = flash_attn_func(q, k, v)
    # Transpose back to (batch_size, head_num, seq_len, head_dim)
    out_fa = out_fa.transpose(1, 2)
    print("fa3 vs math:", get_rtol_atol(out_fa, out_math))
    print("The above (except max_rtol) should be < 0.002")


if __name__ == "__main__":
    main()
