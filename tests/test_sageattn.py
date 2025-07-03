#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from sageattention import sageattn
from torch.nn.attention import SDPBackend, sdpa_kernel


def get_rtol_atol(actual, expect):
    actual = actual.float()
    expect = expect.float()
    diff = (actual - expect).abs()
    eps = torch.tensor(
        torch.finfo(actual.dtype).eps, device=actual.device, dtype=actual.dtype
    )
    rdiff = diff / torch.maximum(torch.maximum(actual.abs(), expect.abs()), eps)
    return (
        f"mean_rtol={rdiff.mean().item():.3g} "
        f"max_rtol={rdiff.max().item():.3g} "
        f"mean_atol={diff.max().item():.3g} "
        f"max_atol={diff.max().item():.3g}"
    )


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

    out_sage = sageattn(q, k, v)
    print("sage vs math:", get_rtol_atol(out_sage, out_math))
    print("The above (except max_rtol) should be < 0.05 (on RTX 20xx/30xx) or < 0.1 (on RTX 40xx/50xx)")


if __name__ == "__main__":
    main()
