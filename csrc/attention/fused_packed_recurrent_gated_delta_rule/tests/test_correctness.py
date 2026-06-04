#!/usr/bin/env python3
"""
Correctness test for fused_packed_recurrent_gated_delta_rule AscendC kernel.

Compares the new fused kernel output against a PyTorch reference implementation.

Usage:
    python test_correctness.py
    python test_correctness.py --verbose
    python test_correctness.py --seed 42 --rtol 1e-2 --atol 1e-2
"""

import argparse
import sys
import time

import torch
import torch_npu  # noqa: F401 - initializes NPU device

from reference import (
    fused_packed_recurrent_gated_delta_rule_ref,
)


def has_custom_op() -> bool:
    """Check if the custom op is available (requires compilation)."""
    try:
        from vllm_ascend import vllm_ascend_C  # noqa: F401 - loads torch ops
        _ = torch.ops._C_ascend.npu_fused_packed_recurrent_gated_delta_rule
        return True
    except (AttributeError, RuntimeError, ImportError):
        return False


def pack_mixed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Pack separate Q/K/V tensors into the packed mixed_qkv format expected by the kernel.

    Q: [B, T, H, K]  (T=1 for decode)
    K: [B, T, H, K]
    V: [B, T, HV, V]
    Returns: [B, 2*H*K + HV*V]
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    # q: [B, T, H, K] -> [B, T, H*K]
    q_flat = q.reshape(B, T, H * K)
    k_flat = k.reshape(B, T, H * K)
    v_flat = v.reshape(B, T, HV * V)
    # Concat: [B, T, H*K + H*K + HV*V] -> squeeze T
    mq = torch.cat([q_flat, k_flat, v_flat], dim=-1).squeeze(1)
    return mq.contiguous()


def run_correctness_test(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    N: int,
    rtol: float,
    atol: float,
    seed: int,
    verbose: bool,
) -> bool:
    """Run a single correctness test case."""
    torch.manual_seed(seed)
    device = torch.device("npu")

    qkv_dim = 2 * H * K + HV * V

    # Generate random inputs on NPU.
    q = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, 1, HV, V, dtype=torch.bfloat16, device=device)

    mixed_qkv = pack_mixed_qkv(q, k, v)

    a = torch.randn(B, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(B, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    # State: [N, HV, V, K]
    ssm_state = torch.randn(N, HV, V, K, dtype=torch.float32, device=device)

    # ssm_state_indices: one index per batch element
    ssm_state_indices = torch.randint(1, N, (B,), dtype=torch.int32, device=device)

    scale = K ** -0.5

    # --- Reference computation ---
    # Copy state to avoid aliasing
    ssm_state_ref = ssm_state.clone()
    out_ref = fused_packed_recurrent_gated_delta_rule_ref(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state_ref,
        ssm_state_indices=ssm_state_indices,
        head_k_dim=K,
        head_v_dim=V,
        num_k_heads=H,
        num_v_heads=HV,
        scale=scale,
    )

    # --- Kernel computation ---
    if not has_custom_op():
        print("SKIP: Custom op not available. Compile vllm-ascend first.")
        return False

    ssm_state_kernel = ssm_state.clone()
    try:
        out_kernel = torch.ops._C_ascend.npu_fused_packed_recurrent_gated_delta_rule(
            mixed_qkv, a, b, A_log, dt_bias,
            ssm_state_kernel, ssm_state_indices, scale=scale,
        )
    except Exception as e:
        print(f"FAIL: Kernel invocation error: {e}")
        return False

    # --- Compare ---
    # Output: [B, HV, V]
    out_ref_2d = out_ref  # [B, HV, V]
    out_kernel_2d = out_kernel.squeeze(1)  # [B, 1, HV, V] -> [B, HV, V]

    out_match = torch.allclose(out_ref_2d.float(), out_kernel_2d.float(), rtol=rtol, atol=atol)
    max_out_diff = (out_ref_2d.float() - out_kernel_2d.float()).abs().max().item()

    state_match = torch.allclose(ssm_state_ref, ssm_state_kernel, rtol=rtol, atol=atol)
    max_state_diff = (ssm_state_ref - ssm_state_kernel).abs().max().item()

    if verbose or not (out_match and state_match):
        print(f"  Config: B={B}, H={H}, HV={HV}, K={K}, V={V}, N={N}")
        print(f"  Output match: {out_match}, max diff: {max_out_diff:.6e}")
        print(f"  State  match: {state_match}, max diff: {max_state_diff:.6e}")

    return out_match and state_match


def main():
    parser = argparse.ArgumentParser(description="Correctness test for packed decode kernel")
    parser.add_argument("--rtol", type=float, default=2e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=2e-2, help="Absolute tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not has_custom_op():
        print("=" * 60)
        print("Custom op 'npu_fused_packed_recurrent_gated_delta_rule' not found.")
        print("The kernel needs to be compiled first (e.g., via pip install -e .)")
        print("=" * 60)
        sys.exit(1)

    # Test configurations covering realistic GDN decode dimensions.
    # (B, H, HV, K, V, N)
    test_cases = [
        (1, 16, 32, 128, 128, 256),    # Single token, typical Qwen3.5
        (4, 16, 32, 128, 128, 256),    # Small batch
        (32, 16, 32, 128, 128, 1024),  # Medium batch
        (128, 16, 32, 128, 128, 1024), # Large batch
        (1, 4, 8, 64, 64, 128),        # Small heads
        (1, 8, 16, 128, 128, 128),     # Different GQA ratio
        (8, 8, 16, 256, 128, 256),     # Large K
        (8, 8, 16, 128, 256, 256),     # Large V
    ]

    print("=" * 60)
    print("Correctness test: fused_packed_recurrent_gated_delta_rule")
    print(f"rtol={args.rtol}, atol={args.atol}, seed={args.seed}")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for tc in test_cases:
        B, H, HV, K, V, N = tc
        result = run_correctness_test(
            B=B, H=H, HV=HV, K=K, V=V, N=N,
            rtol=args.rtol, atol=args.atol,
            seed=args.seed, verbose=args.verbose,
        )
        if result:
            passed += 1
            if not args.verbose:
                print(f"  PASS: B={B}, H={H}, HV={HV}, K={K}, V={V}, N={N}")
        else:
            failed += 1

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
