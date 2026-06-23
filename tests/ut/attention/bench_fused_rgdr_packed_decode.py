# SPDX-License-Identifier: Apache-2.0
"""Performance benchmark for fused_rgdr_packed_decode."""

import time
import torch
from vllm_ascend.utils import enable_custom_op
enable_custom_op()


def bench(name, mixed_qkv, a, b, a_log, dt_bias, state, si, scale=1.0,
          warmup=5, repeat=50):
    """Run kernel and return avg microseconds."""
    mq = mixed_qkv.npu()
    an = a.npu(); bn = b.npu()
    al = a_log.npu(); db = dt_bias.npu()
    si_n = si.npu()
    st0 = state.npu().clone()

    # Warmup
    for _ in range(warmup):
        st_warm = st0.clone()
        torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
            mq, an, bn, al, db, st_warm, si_n, scale)
    torch.npu.synchronize()

    # Timed runs -- clone once, then reuse by filling with original
    st_work = st0.clone()
    start = time.perf_counter()
    for _ in range(repeat):
        st_work.copy_(st0)
        torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
            mq, an, bn, al, db, st_work, si_n, scale)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    avg_us = elapsed / repeat * 1e6

    print(f"  {name:40s} avg={avg_us:8.1f} us  ({repeat} runs)")
    return avg_us


def make_inputs(B, HV, DV, DK, N=8, seed=42, state_dtype=torch.float32):
    torch.manual_seed(seed)
    HK = HV
    L = 2 * HK * DK + HV * DV
    return (
        torch.randn(B, L, dtype=torch.bfloat16),
        torch.randn(B, HV, dtype=torch.bfloat16),
        torch.randn(B, HV, dtype=torch.bfloat16),
        torch.randn(HV, dtype=torch.float32),
        torch.randn(HV, dtype=torch.float32),
        torch.randn(N, HV, DV, DK, dtype=state_dtype),
        torch.randperm(N)[:B].to(torch.int32),
    )


def main():
    print("=== Single-op Performance Benchmark ===\n")

    # ---- Model-like dimensions (Qwen3.5-0.8B) ----
    # DV=DK=128, HV=HK=16  (the real decode scenario)
    print("--- Model-like (DV=DK=128, HV=16) ---")
    for B in [1, 2, 4, 8, 10]:
        mq, a, b, al, db, st, si = make_inputs(B, 16, 128, 128, N=max(16, B))
        bench(f"B={B:3d}, HV=16, DV=DK=128", mq, a, b, al, db, st, si)

    # ---- Batch scaling with large state ----
    print("\n--- Batch scaling (HV=32, DV=DK=128) ---")
    for B in [1, 4, 16, 32, 64]:
        mq, a, b, al, db, st, si = make_inputs(B, 32, 128, 128, N=max(64, B))
        bench(f"B={B:3d}, HV=32, DV=DK=128", mq, a, b, al, db, st, si)

    # ---- HV scaling ----
    print("\n--- HV scaling (B=1, DV=DK=128) ---")
    for HV in [8, 16, 24, 32]:
        mq, a, b, al, db, st, si = make_inputs(1, HV, 128, 128)
        bench(f"B=1, HV={HV:3d}, DV=DK=128", mq, a, b, al, db, st, si)

    # ---- bf16 state ----
    print("\n--- bf16 state ---")
    mq, a, b, al, db, st_bf16, si = make_inputs(
        1, 16, 128, 128, state_dtype=torch.bfloat16)
    bench(f"B=1, HV=16, DV=DK=128 (bf16 state)", mq, a, b, al, db, st_bf16, si)

    print("\nDone.")


if __name__ == "__main__":
    main()
