# SPDX-License-Identifier: Apache-2.0
"""Performance benchmark for fused_rgdr_packed_decode.

Test cases aligned with test_fused_rgdr_packed_decode.py correctness tests.
"""

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

    # Timed runs
    st_work = st0.clone()
    start = time.perf_counter()
    for _ in range(repeat):
        st_work.copy_(st0)
        torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
            mq, an, bn, al, db, st_work, si_n, scale)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    avg_us = elapsed / repeat * 1e6

    print(f"  {name:45s} avg={avg_us:8.1f} us  ({repeat} runs)")
    return avg_us


def make_inputs_e2e(HK, HV, DV, DK, B=1, N=8, seed=42, state_dtype=torch.float32):
    """Build inputs with independent HK/HV (matching _make_inputs_e2e in test)."""
    torch.manual_seed(seed)
    L = 2 * HK * DK + HV * DV
    mixed_qkv = torch.randn(B, L, dtype=torch.bfloat16)
    a = torch.randn(B, HV, dtype=torch.bfloat16)
    b = torch.randn(B, HV, dtype=torch.bfloat16)
    a_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    state = torch.randn(N, HV, DV, DK, dtype=state_dtype)
    si = torch.randperm(N)[:B].to(torch.int32)
    return mixed_qkv, a, b, a_log, dt_bias, state, si


def main():
    print("=== Single-op Performance Benchmark ===\n")

    # ---- Basic correctness-aligned cases (DV=DK=16) ----
    print("--- Basic cases (DV=DK=16, matching test_hv8_b1/hv8_b4/hv16) ---")
    for B, HV in [(1, 8), (4, 8), (1, 16)]:
        HK = HV  # G=1
        mq, a, b, al, db, st, si = make_inputs_e2e(HK, HV, 8, 16, B=B, N=max(8, B))
        bench(f"G=1, B={B:2d}, HK={HK:2d}, HV={HV:2d}", mq, a, b, al, db, st, si)
        import gc; gc.collect()
        torch.npu.empty_cache()

    # ---- E2E combos (DV=DK=128, matching _E2E_COMBOS in test) ----
    _COMBOS = [(8, 8), (8, 16), (8, 24), (8, 32), (8, 64),
               (16, 16), (16, 32), (16, 64)]
    print("\n--- E2E combos (DV=DK=128, matching test_e2e_combos) ---")
    for HK, HV in _COMBOS:
        G = HV // HK
        mq, a, b, al, db, st, si = make_inputs_e2e(HK, HV, 128, 128, N=max(8, 1))
        bench(f"G={G}, B=1, HK={HK:2d}, HV={HV:2d}", mq, a, b, al, db, st, si)
        import gc; gc.collect()
        torch.npu.empty_cache()

    # ---- Batch scaling (DV=DK=128, HK=8, HV=32, matching test_batch) ----
    print("\n--- Batch scaling (HK=8, HV=32, DV=DK=128, matching test_batch) ---")
    for B in [2, 4, 8]:
        mq, a, b, al, db, st, si = make_inputs_e2e(8, 32, 128, 128, B=B, N=max(8, B))
        bench(f"G=4, B={B:2d}, HK= 8, HV=32", mq, a, b, al, db, st, si)
        import gc; gc.collect()
        torch.npu.empty_cache()

    # ---- bf16 state (DV=DK=16, matching test_bf16_state) ----
    print("\n--- bf16 state (DV=DK=16, matching test_bf16_state) ---")
    mq, a, b, al, db, st_b16, si = make_inputs_e2e(
        16, 16, 8, 16, B=2, N=8, state_dtype=torch.bfloat16)
    bench(f"G=1, B= 2, HK=16, HV=16 (bf16 state)", mq, a, b, al, db, st_b16, si)
    import gc; gc.collect()
    torch.npu.empty_cache()

    # ---- Model-like: Qwen3.6-27B GDN config (G=3, DV=DK=128) ----
    print("\n--- Model-like (DV=DK=128, G=3, matching Qwen3.6-27B) ---")
    # TP=1: full heads (HK=16, HV=48)
    for B in [1, 4, 10]:
        mq, a, b, al, db, st, si = make_inputs_e2e(16, 48, 128, 128, B=B, N=max(48, B))
        bench(f"TP1, B={B:2d}, HK=16, HV=48", mq, a, b, al, db, st, si)
        gc.collect(); torch.npu.empty_cache()
    # TP=2: half heads per card (HK=8, HV=24), G=3 preserved
    for B in [1, 4, 10]:
        mq, a, b, al, db, st, si = make_inputs_e2e(8, 24, 128, 128, B=B, N=max(24, B))
        bench(f"TP2, B={B:2d}, HK= 8, HV=24", mq, a, b, al, db, st, si)
        gc.collect(); torch.npu.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
