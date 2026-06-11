#!/usr/bin/env python3
"""
Correctness test for fused_packed_recurrent_gated_delta_rule AscendC kernel.

Compares the fused kernel output against a PyTorch reference implementation
that executes the identical math: L2Norm(q,k), gating via -exp(A_log)*softplus(a+dt_bias),
sigmoid(beta), then gated delta rule recurrence.

Usage:
    python test_correctness.py
    python test_correctness.py --verbose
    python test_correctness.py --seed 42 --rtol 2e-2 --atol 2e-2
    python test_correctness.py --list   # list test cases without running
"""

import argparse
import sys
import time

import torch
import torch_npu  # noqa: F401 - initializes NPU device


# ============================================================================
# Reference implementation
# ============================================================================

def fused_packed_recurrent_gated_delta_rule_ref(
    mixed_qkv: torch.Tensor,       # [B, qkv_dim] bf16, packed Q|K|V
    a: torch.Tensor,               # [B, HV] bf16
    b: torch.Tensor,               # [B, HV] bf16
    A_log: torch.Tensor,           # [HV] float32
    dt_bias: torch.Tensor,         # [HV] float32
    ssm_state: torch.Tensor,       # [N, HV, V, K] float32 (updated in-place)
    ssm_state_indices: torch.Tensor,  # [B] int32, unique, in [0, N); negative = PAD_SLOT_ID
    head_k_dim: int,
    head_v_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    scale: float | None = None,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """
    Reference implementation of fused packed recurrent gated delta rule.

    Per token, per value-head:
      1. Unpack q, k, v from mixed_qkv layout
      2. L2Norm q and k (eps=1e-6)
      3. Scale q by scale (default K ** -0.5)
      4. g = -exp(A_log) * softplus(a + dt_bias)
      5. beta = sigmoid(b)
      6. h *= exp(g)
      7. delta = v - sum(h * k, dim=1)
      8. delta *= beta
      9. h += outer(delta, k)
      10. out = sum(h * q, dim=1)

    ssm_state_indices < 0 are treated as PAD_SLOT_ID and skipped
    (output stays zero, state not read/written).

    Returns: out [B, HV, V] bf16
    """
    B = mixed_qkv.shape[0]
    HV = num_v_heads
    K = head_k_dim
    V = head_v_dim
    H = num_k_heads

    if scale is None:
        scale = K ** -0.5

    device = mixed_qkv.device
    out = torch.empty(B, HV, V, dtype=torch.bfloat16, device=device)

    # Work in fp32 for accuracy
    mq = mixed_qkv.float()
    a_f = a.float()
    b_f = b.float()
    al_f = A_log.float()
    dt_f = dt_bias.float()

    for bi in range(B):
        state_idx = ssm_state_indices[bi].item()
        if state_idx < 0:
            # PAD_SLOT_ID: skip padding tokens.
            continue

        for hvi in range(HV):
            hi = hvi // (HV // H)  # GQA: map value head -> key head

            # Unpack from mixed_qkv layout: [Q | K | V]
            q_off = hi * K
            k_off = H * K + hi * K
            v_off = 2 * H * K + hvi * V

            q = mq[bi, q_off:q_off + K]
            k = mq[bi, k_off:k_off + K]
            v = mq[bi, v_off:v_off + V]

            # L2Norm with eps
            q = q / (torch.sqrt(torch.sum(q * q) + 1e-6))
            k = k / (torch.sqrt(torch.sum(k * k) + 1e-6))
            q = q * scale

            # Gating
            x = a_f[bi, hvi] + dt_f[hvi]
            if x <= softplus_threshold:
                sp = torch.log(1.0 + torch.exp(x))
            else:
                sp = x
            g = -torch.exp(al_f[hvi]) * sp
            beta = torch.sigmoid(b_f[bi, hvi])

            # Recurrence
            h = ssm_state[state_idx, hvi].clone()
            h *= torch.exp(g)
            delta = v - (h * k.unsqueeze(0)).sum(dim=1)
            delta *= beta
            h += delta.unsqueeze(1) * k.unsqueeze(0)
            o = (h * q.unsqueeze(0)).sum(dim=1)

            out[bi, hvi] = o.to(torch.bfloat16)
            ssm_state[state_idx, hvi] = h

    return out


# ============================================================================
# Test infrastructure
# ============================================================================

def has_custom_op() -> bool:
    """Check if the custom op is available (requires compilation)."""
    try:
        from vllm_ascend.utils import enable_custom_op
        enable_custom_op()
        _ = torch.ops._C_ascend.npu_fused_packed_recurrent_gated_delta_rule
        return True
    except (AttributeError, RuntimeError, ImportError):
        return False


def _make_unique_indices(B: int, N: int, seed: int) -> torch.Tensor:
    """Generate B unique state indices in [0, N)."""
    assert N > B, f"N={N} must be > B={B} to guarantee unique indices"
    g = torch.Generator()
    g.manual_seed(seed * 10007 + B * 31337)
    perm = torch.randperm(N, generator=g)[:B]
    return perm.int()


def pack_mixed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Pack separate Q/K/V tensors into the packed mixed_qkv format.
    Q: [B, T, H, K]  (T=1 for decode)
    K: [B, T, H, K]
    V: [B, T, HV, V]
    Returns: [B, 2*H*K + HV*V]
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    q_flat = q.reshape(B, T, H * K)
    k_flat = k.reshape(B, T, H * K)
    v_flat = v.reshape(B, T, HV * V)
    mq = torch.cat([q_flat, k_flat, v_flat], dim=-1).squeeze(1)
    return mq.contiguous()


def run_correctness_test(
    *,
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
) -> dict:
    """Run a single correctness test case. Returns dict with pass/fail details."""
    device = torch.device("npu")

    # --- Generate inputs ---
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    q = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device, generator=g)
    k = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device, generator=g)
    v = torch.randn(B, 1, HV, V, dtype=torch.bfloat16, device=device, generator=g)
    mixed_qkv = pack_mixed_qkv(q, k, v)

    a = torch.randn(B, HV, dtype=torch.bfloat16, device=device, generator=g)
    b = torch.randn(B, HV, dtype=torch.bfloat16, device=device, generator=g)
    A_log = torch.randn(HV, dtype=torch.float32, device=device, generator=g)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device, generator=g)

    ssm_state_initial = torch.randn(N, HV, V, K, dtype=torch.float32, device=device, generator=g)
    # Unique indices per batch element: each sequence owns its state slot
    ssm_state_indices = _make_unique_indices(B, N, seed).to(device)

    scale = K ** -0.5

    # --- Reference ---
    ssm_state_ref = ssm_state_initial.clone()
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    out_ref = fused_packed_recurrent_gated_delta_rule_ref(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        ssm_state=ssm_state_ref, ssm_state_indices=ssm_state_indices,
        head_k_dim=K, head_v_dim=V, num_k_heads=H, num_v_heads=HV,
        scale=scale,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    ref_time = time.time() - t0

    # --- Kernel ---
    ssm_state_kernel = ssm_state_initial.clone()
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    try:
        out_kernel = torch.ops._C_ascend.npu_fused_packed_recurrent_gated_delta_rule(
            mixed_qkv, a, b, A_log, dt_bias,
            ssm_state_kernel, ssm_state_indices, scale=scale,
        )
    except Exception as e:
        return {"passed": False, "error": f"Kernel invocation failed: {e}"}
    torch.cuda.synchronize() if device.type == "cuda" else None
    kernel_time = time.time() - t0

    # --- Compare ---
    # Kernel output: [B, 1, HV, V]; reference: [B, HV, V]
    out_k = out_kernel.squeeze(1).float()
    out_r = out_ref.float()

    out_match = torch.allclose(out_r, out_k, rtol=rtol, atol=atol)
    max_out_diff = (out_r - out_k).abs().max().item()

    state_match = torch.allclose(ssm_state_ref, ssm_state_kernel, rtol=rtol, atol=atol)
    max_state_diff = (ssm_state_ref - ssm_state_kernel).abs().max().item()

    return {
        "passed": out_match and state_match,
        "out_match": out_match,
        "max_out_diff": max_out_diff,
        "state_match": state_match,
        "max_state_diff": max_state_diff,
        "ref_time": ref_time,
        "kernel_time": kernel_time,
    }


# ============================================================================
# Test case definitions
# ============================================================================

def get_test_cases() -> list[dict]:
    """Define all correctness test configurations.

    Format: {name, B, H, HV, K, V, N, desc}
    """
    return [
        # --- Core decode scenarios ---
        {"name": "single_token_typical",  "B": 1,   "H": 16, "HV": 32, "K": 128, "V": 128, "N": 256,
         "desc": "Single token, typical Qwen3.5 GDN config"},
        {"name": "small_batch",           "B": 4,   "H": 16, "HV": 32, "K": 128, "V": 128, "N": 256,
         "desc": "Small decode batch"},
        {"name": "medium_batch",          "B": 32,  "H": 16, "HV": 32, "K": 128, "V": 128, "N": 1024,
         "desc": "Medium decode batch"},
        {"name": "large_batch",           "B": 128, "H": 16, "HV": 32, "K": 128, "V": 128, "N": 1024,
         "desc": "Large decode batch"},

        # --- Dimension variants ---
        {"name": "small_heads",           "B": 1,  "H": 4,  "HV": 8,  "K": 64,  "V": 64,  "N": 128,
         "desc": "Small head dimensions"},
        {"name": "gqa_ratio_2",           "B": 1,  "H": 8,  "HV": 16, "K": 128, "V": 128, "N": 128,
         "desc": "GQA ratio 2 (HV=2*H)"},
        {"name": "gqa_ratio_8",           "B": 1,  "H": 2,  "HV": 16, "K": 128, "V": 128, "N": 128,
         "desc": "Extreme GQA ratio 8 (HV=8*H)"},
        {"name": "mha",                   "B": 1,  "H": 16, "HV": 16, "K": 128, "V": 128, "N": 128,
         "desc": "MHA (HV==H)"},
        {"name": "large_K",               "B": 8,  "H": 8,  "HV": 16, "K": 256, "V": 128, "N": 256,
         "desc": "Large head_k_dim"},
        {"name": "large_V",               "B": 8,  "H": 8,  "HV": 16, "K": 128, "V": 256, "N": 256,
         "desc": "Large head_v_dim"},
        {"name": "large_both",            "B": 4,  "H": 8,  "HV": 16, "K": 256, "V": 256, "N": 256,
         "desc": "Both K and V large"},

        # --- Edge cases ---
        {"name": "minimal",               "B": 1,  "H": 1,  "HV": 2,  "K": 32,  "V": 32,  "N": 16,
         "desc": "Minimal dimensions"},
        {"name": "single_state_slot",     "B": 1,  "H": 4,  "HV": 8,  "K": 64,  "V": 64,  "N": 2,
         "desc": "Minimal state slots (N=2)"},
    ]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correctness test for fused_packed_recurrent_gated_delta_rule kernel"
    )
    parser.add_argument("--rtol", type=float, default=2e-2,
                        help="Relative tolerance (default: 2e-2)")
    parser.add_argument("--atol", type=float, default=2e-2,
                        help="Absolute tolerance (default: 2e-2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output with per-test timing")
    parser.add_argument("--list", action="store_true",
                        help="List all test cases and exit")
    parser.add_argument("--filter", type=str, default="",
                        help="Run only test cases whose name contains this substring")
    args = parser.parse_args()

    test_cases = get_test_cases()

    if args.list:
        print(f"{'Name':<30} {'B':>4} {'H':>3} {'HV':>3} {'K':>4} {'V':>4} {'N':>5}  Description")
        print("-" * 90)
        for tc in test_cases:
            print(f"{tc['name']:<30} {tc['B']:>4} {tc['H']:>3} {tc['HV']:>3} "
                  f"{tc['K']:>4} {tc['V']:>4} {tc['N']:>5}  {tc['desc']}")
        return

    # Single check at the top level
    if not has_custom_op():
        print("=" * 60)
        print("Custom op 'npu_fused_packed_recurrent_gated_delta_rule' not found.")
        print("The kernel needs to be compiled first:")
        print("  pip install --no-build-isolation -e .")
        print("or for faster iteration on op_host/op_kernel changes only:")
        print("  cd csrc && cmake --build build -j$(nproc) && cd build && \\")
        print("  cpack -G External && cd .. && \\")
        print("  find build -name 'cann-ops-transformer*.run' -exec {} --install-path=../vllm_ascend/_cann_ops_custom \\;")
        print("=" * 60)
        sys.exit(1)

    # Filter if requested
    if args.filter:
        test_cases = [tc for tc in test_cases if args.filter in tc["name"]]
        if not test_cases:
            print(f"No test cases match filter '{args.filter}'")
            sys.exit(1)

    print("=" * 60)
    print("Correctness test: fused_packed_recurrent_gated_delta_rule")
    print(f"rtol={args.rtol}, atol={args.atol}, seed={args.seed}")
    print(f"Test cases: {len(test_cases)}")
    print("=" * 60)

    passed = 0
    failed = 0
    results = []

    for tc in test_cases:
        result = run_correctness_test(
            B=tc["B"], H=tc["H"], HV=tc["HV"], K=tc["K"], V=tc["V"], N=tc["N"],
            rtol=args.rtol, atol=args.atol,
            seed=args.seed, verbose=args.verbose,
        )

        name = tc["name"]
        if result["passed"]:
            passed += 1
            timing = ""
            if args.verbose:
                timing = (f"  ref={result['ref_time']*1000:.2f}ms"
                          f" kernel={result['kernel_time']*1000:.2f}ms")
            print(f"  PASS: {name:<30} ({tc['desc']}){timing}")
        else:
            failed += 1
            if "error" in result:
                print(f"  FAIL: {name:<30} - {result['error']}")
            else:
                print(f"  FAIL: {name:<30} ({tc['desc']})")
                print(f"    out_match={result['out_match']}, max_out_diff={result['max_out_diff']:.6e}")
                print(f"    state_match={result['state_match']}, max_state_diff={result['max_state_diff']:.6e}")

        results.append((name, result))

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(test_cases)} total")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
