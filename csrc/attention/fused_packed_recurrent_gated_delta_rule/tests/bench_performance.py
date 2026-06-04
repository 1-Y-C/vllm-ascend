#!/usr/bin/env python3
"""
Performance benchmark for fused_packed_recurrent_gated_delta_rule AscendC kernel.

Compares:
  A) New fused kernel (single AscendC kernel: L2Norm + gating + recurrence from packed qkv)
  B) Existing multi-kernel path (rearrange + l2norm_fwd*2 + fused_gdn_gating_patch + RGDR)

Usage:
    python bench_performance.py
    python bench_performance.py --warmup 10 --repeat 100
    python bench_performance.py --batch 1,4,16,64,128
"""

import argparse
import sys
import time
from typing import Callable

import torch
import torch_npu  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_custom_op() -> bool:
    try:
        _ = torch.ops._C_ascend.npu_fused_packed_recurrent_gated_delta_rule
        return True
    except (AttributeError, RuntimeError):
        return False


def has_existing_ops() -> bool:
    try:
        _ = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
        return True
    except (AttributeError, RuntimeError):
        return False


def make_inputs(B: int, H: int, HV: int, K: int, V: int, N: int):
    """Create inputs for a decode batch."""
    device = torch.device("npu")
    qkv_dim = 2 * H * K + HV * V

    q = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, 1, HV, V, dtype=torch.bfloat16, device=device)

    q_flat = q.reshape(B, 1, H * K)
    k_flat = k.reshape(B, 1, H * K)
    v_flat = v.reshape(B, 1, HV * V)
    mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1).squeeze(1).contiguous()

    a = torch.randn(B, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    ssm_state = torch.randn(N, HV, V, K, dtype=torch.float32, device=device)
    ssi = torch.randint(1, N, (B,), dtype=torch.int32, device=device)
    scale = K ** -0.5

    return mixed_qkv, a, b, A_log, dt_bias, ssm_state, ssi, scale, q, k


# ---------------------------------------------------------------------------
# Path A: New fused kernel
# ---------------------------------------------------------------------------

def run_fused_kernel(mixed_qkv, a, b, A_log, dt_bias, ssm_state, ssi, scale):
    """Single fused kernel call."""
    ssm_state_copy = ssm_state.clone()
    out = torch.ops._C_ascend.npu_fused_packed_recurrent_gated_delta_rule(
        mixed_qkv, a, b, A_log, dt_bias, ssm_state_copy, ssi, scale,
    )
    # Sync to ensure timing is accurate.
    torch.npu.synchronize()
    return out


# ---------------------------------------------------------------------------
# Path B: Existing multi-kernel path (for comparison)
# ---------------------------------------------------------------------------

def run_existing_path(mixed_qkv, a, b, A_log, dt_bias, ssm_state, ssi, scale, q, k):
    """
    Replicate the existing Ascend decode flow:
      rearrange_mixed_qkv + l2norm_fwd(q) + l2norm_fwd(k)
      + fused_gdn_gating_patch + npu_recurrent_gated_delta_rule
    """
    from einops import rearrange
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    import torch_npu

    B = mixed_qkv.shape[0]
    H = q.shape[2]
    HV = a.shape[1]
    K = q.shape[3]

    # Simulate rearrange_mixed_qkv
    qkv_dim = mixed_qkv.shape[1]
    v_dim = HV * (qkv_dim - 2 * H * K) // HV  # actually just V * HV

    # Actually, let's use the pre-computed q, k and extract v from mixed_qkv
    v_start = 2 * H * K
    v_flat = mixed_qkv[:, v_start:].reshape(B, HV, -1)
    V = v_flat.shape[-1]

    # L2Norm
    q_norm = l2norm_fwd(q.squeeze(1))
    k_norm = l2norm_fwd(k.squeeze(1))

    # Gating
    g, beta = fused_gdn_gating_patch(A_log, a, b, dt_bias)

    # Actual seq lengths (each is 1 for decode)
    cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device=mixed_qkv.device)
    actual_seq_lengths = torch.ones(B, dtype=torch.int32, device=mixed_qkv.device)

    ssm_state_copy = ssm_state.clone()

    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=q_norm,
        key=k_norm,
        value=v_flat,
        state=ssm_state_copy,
        beta=beta.squeeze(0),
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssi,
        g=g.squeeze(0),
    )
    torch.npu.synchronize()
    return out


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark(fn: Callable, args: tuple, warmup: int, repeat: int, label: str) -> dict:
    """Run a benchmark and return timing stats."""
    # Warmup
    for _ in range(warmup):
        fn(*args)

    # Timing
    times = []
    for _ in range(repeat):
        # Re-create state to avoid cumulative effects
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times_t = torch.tensor(times)
    return {
        "label": label,
        "mean_ms": times_t.mean().item(),
        "std_ms": times_t.std().item(),
        "min_ms": times_t.min().item(),
        "max_ms": times_t.max().item(),
        "median_ms": times_t.median().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark for packed decode kernel")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Timing iterations")
    parser.add_argument("--batches", type=str, default="1,4,16,32,64,128,256",
                        help="Comma-separated batch sizes")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batches.split(",")]

    if not has_custom_op():
        print("Custom op not available. Compile vllm-ascend first.")
        sys.exit(1)

    # Default model dimensions (Qwen3.5-like)
    H, HV, K, V, N = 16, 32, 128, 128, 1024

    print("=" * 70)
    print("Performance Benchmark: fused_packed_recurrent_gated_delta_rule")
    print(f"Model dims: H={H}, HV={HV}, K={K}, V={V}, N={N}")
    print(f"Warmup={args.warmup}, Repeat={args.repeat}")
    print("=" * 70)

    results = []

    for B in batch_sizes:
        mq, a_, b_, al_, dt_, st_, ssi, scl, q_, k_ = make_inputs(B, H, HV, K, V, N)

        # Benchmark fused kernel
        def run_fused():
            return run_fused_kernel(mq.clone(), a_.clone(), b_.clone(),
                                     al_.clone(), dt_.clone(), st_.clone(),
                                     ssi.clone(), scl)
        r_fused = benchmark(run_fused, (), args.warmup, args.repeat, f"Fused (B={B})")
        results.append(r_fused)

        # Benchmark existing path
        if has_existing_ops():
            def run_existing():
                return run_existing_path(mq.clone(), a_.clone(), b_.clone(),
                                          al_.clone(), dt_.clone(), st_.clone(),
                                          ssi.clone(), scl, q_.clone(), k_.clone())
            r_existing = benchmark(run_existing, (), args.warmup, args.repeat,
                                    f"Existing (B={B})")
            results.append(r_existing)

    # Print results table
    print()
    print(f"{'Path':<30} {'B':>6} {'Mean(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
    print("-" * 76)
    for r in results:
        print(f"{r['label']:<30} {int(r['label'].split('=')[1].rstrip(')')):>6} "
              f"{r['mean_ms']:>10.4f} {r['std_ms']:>10.4f} "
              f"{r['min_ms']:>10.4f} {r['max_ms']:>10.4f}")

    # Speedup summary
    fused_results = {r["label"]: r for r in results if "Fused" in r["label"]}
    existing_results = {r["label"]: r for r in results if "Existing" in r["label"]}

    if existing_results:
        print()
        print("Speedup (Fused vs Existing):")
        print(f"{'B':>6} {'Fused(ms)':>10} {'Existing(ms)':>12} {'Speedup':>8}")
        print("-" * 42)
        for B in batch_sizes:
            fkey = f"Fused (B={B})"
            ekey = f"Existing (B={B})"
            if fkey in fused_results and ekey in existing_results:
                ft = fused_results[fkey]["mean_ms"]
                et = existing_results[ekey]["mean_ms"]
                sp = et / ft if ft > 0 else 0
                print(f"{B:>6} {ft:>10.4f} {et:>12.4f} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
