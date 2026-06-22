# SPDX-License-Identifier: Apache-2.0
"""Unit test for fused_rgdr_packed_decode AscendC kernel.

Validate NPU output against CPU golden reference for known-working
parameter combinations. Run with:

  python tests/ut/attention/test_fused_rgdr_packed_decode.py
"""

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


# ======================== CPU Golden Reference ==============================

def golden(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale_val: float = 1.0,
):
    """CPU reference implementation of fused_rgdr_packed_decode."""
    B = mixed_qkv.shape[0]
    L = mixed_qkv.shape[1]
    HV = a.shape[1]
    DV = state.shape[2]
    DK = state.shape[3]

    HK = (L - HV * DV) // (2 * DK)
    assert HK > 0, f"Invalid HK={HK}"
    assert L == 2 * HK * DK + HV * DV, f"L mismatch: {L} != {2 * HK * DK + HV * DV}"

    G = HV // HK
    eps = 1e-6
    out = torch.zeros(B, 1, HV, DV, dtype=torch.bfloat16)
    state_out = state.clone()

    for bi in range(B):
        sid = int(ssm_state_indices[bi].item())
        if sid < 0:
            continue

        mx = mixed_qkv[bi].to(torch.float32)
        qO, kO, vO = 0, HK * DK, 2 * HK * DK
        Q = mx[qO : qO + HK * DK].reshape(HK, DK)
        K = mx[kO : kO + HK * DK].reshape(HK, DK)
        V = mx[vO : vO + HV * DV].reshape(HV, DV)

        for h in range(HK):
            Q[h] /= torch.sqrt((Q[h] ** 2).sum() + eps)
            K[h] /= torch.sqrt((K[h] ** 2).sum() + eps)

        Q *= scale_val

        a_f = a[bi].to(torch.float32)
        b_f = b[bi].to(torch.float32)
        gate = torch.exp(-torch.exp(a_log) * torch.log(1.0 + torch.exp(a_f + dt_bias)))
        beta = torch.sigmoid(b_f)

        for hvi in range(HV):
            hki = hvi // G if G > 0 else hvi
            st = state_out[sid, hvi].to(torch.float32).clone()

            if gate[hvi].item() != 1.0:
                st *= gate[hvi]

            o = V[hvi] - torch.einsum("vk,vk->v", st, K[hki].unsqueeze(0).expand(DV, DK))
            o *= beta[hvi]

            for dvi in range(DV):
                if o[dvi].item() != 0.0:
                    st[dvi] += K[hki] * o[dvi]

            o = torch.einsum("vk,vk->v", st, Q[hki].unsqueeze(0).expand(DV, DK))
            out[bi, 0, hvi] = o.to(torch.bfloat16)
            state_out[sid, hvi] = st.to(state.dtype)

    return out, state_out


# ============================ Test Helpers ==================================

def _make_inputs(B, HV, DV, DK, N=4, seed=42):
    torch.manual_seed(seed)
    HK = HV
    L = 2 * HK * DK + HV * DV
    mixed_qkv = torch.randn(B, L, dtype=torch.bfloat16)
    a = torch.randn(B, HV, dtype=torch.bfloat16)
    b = torch.randn(B, HV, dtype=torch.bfloat16)
    a_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    state = torch.randn(N, HV, DV, DK, dtype=torch.float32)
    si = torch.randperm(N)[:B].to(torch.int32)  # unique, no data race
    return mixed_qkv, a, b, a_log, dt_bias, state, si


def _run_npu(mixed_qkv, a, b, a_log, dt_bias, state, si, scale=1.0):
    state_npu = state.npu().clone()
    out = torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
        mixed_qkv.npu(), a.npu(), b.npu(),
        a_log.npu(), dt_bias.npu(),
        state_npu, si.npu(), scale,
    )
    return out.cpu(), state_npu.cpu()


def _check(actual_out, actual_state, ref_out, ref_state,
           rtol=5e-2, atol=1e-1):
    diff = (actual_out.to(torch.float32) - ref_out.to(torch.float32)).abs()
    print(f"  out max diff={diff.max():.6f} mean diff={diff.mean():.6f}")
    torch.testing.assert_close(
        actual_out.to(torch.float32), ref_out.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )
    torch.testing.assert_close(
        actual_state.to(torch.float32), ref_state.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )
    print("  PASS")


# =============================== Tests ======================================

def test_hv8_b1():
    """HV=8, B=1: alignment-friendly case."""
    print("test_hv8_b1:")
    B, HV, DV, DK = 1, 8, 8, 16
    mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
    ref_out, ref_state = golden(mixed_qkv, a, b, a_log, dt_bias, state, si)
    npu_out, npu_state = _run_npu(mixed_qkv, a, b, a_log, dt_bias, state, si)
    _check(npu_out, npu_state, ref_out, ref_state)


def test_hv8_b4():
    """HV=8, B=4: batch > 1."""
    print("test_hv8_b4:")
    B, HV, DV, DK = 4, 8, 8, 16
    mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
    ref_out, ref_state = golden(mixed_qkv, a, b, a_log, dt_bias, state, si)
    npu_out, npu_state = _run_npu(mixed_qkv, a, b, a_log, dt_bias, state, si)
    _check(npu_out, npu_state, ref_out, ref_state)


def test_negative_sid():
    """Negative ssm_state_indices → zero output, state unchanged."""
    print("test_negative_sid:")
    B, HV, DV, DK = 2, 8, 8, 16
    mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
    si[0] = -1
    ref_out, ref_state = golden(mixed_qkv, a, b, a_log, dt_bias, state, si)
    npu_out, npu_state = _run_npu(mixed_qkv, a, b, a_log, dt_bias, state, si)

    # First batch output should be all zeros
    assert (npu_out[0] == 0).all(), "skip output not zero"
    # State for skipped index should be unchanged
    torch.testing.assert_close(
        npu_state[0].to(torch.float32),
        state[0].to(torch.float32),
        rtol=0, atol=0,
    )
    # Check rest numerically
    _check(npu_out[1:], npu_state[1:], ref_out[1:], ref_state[1:])
    print("  PASS")


def test_scale_override():
    """Non-default scale value."""
    print("test_scale_override:")
    B, HV, DV, DK = 1, 8, 8, 16
    scale = 0.5
    mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
    ref_out, ref_state = golden(mixed_qkv, a, b, a_log, dt_bias, state, si,
                                scale_val=scale)
    npu_out, npu_state = _run_npu(mixed_qkv, a, b, a_log, dt_bias, state, si,
                                  scale=scale)
    _check(npu_out, npu_state, ref_out, ref_state)


def test_hv16():
    """HV=16: larger head count."""
    print("test_hv16:")
    B, HV, DV, DK = 1, 16, 8, 16
    mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
    ref_out, ref_state = golden(mixed_qkv, a, b, a_log, dt_bias, state, si)
    npu_out, npu_state = _run_npu(mixed_qkv, a, b, a_log, dt_bias, state, si)
    _check(npu_out, npu_state, ref_out, ref_state)


def _make_inputs_e2e(HK, HV, DV, DK, B=1, N=8, seed=42):
    """Build inputs with independent HK/HV (G = HV/HK)."""
    torch.manual_seed(seed)
    L = 2 * HK * DK + HV * DV
    mixed_qkv = torch.randn(B, L, dtype=torch.bfloat16)
    a = torch.randn(B, HV, dtype=torch.bfloat16)
    b = torch.randn(B, HV, dtype=torch.bfloat16)
    a_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    state = torch.randn(N, HV, DV, DK, dtype=torch.float32)
    si = torch.randperm(N)[:B].to(torch.int32)
    return mixed_qkv, a, b, a_log, dt_bias, state, si


def golden_e2e(mixed_qkv, a, b, a_log, dt_bias, state, si, scale_val=1.0):
    """Vectorized CPU golden for large DV/DK."""
    compute_dtype = torch.float32
    B = mixed_qkv.shape[0]; L = mixed_qkv.shape[1]
    HV = a.shape[1]; DV = state.shape[2]; DK = state.shape[3]
    HK = (L - HV * DV) // (2 * DK); G = HV // HK; eps = 1e-6
    a_log_f = a_log.to(compute_dtype); dt_bias_f = dt_bias.to(compute_dtype)
    out = torch.zeros(B, 1, HV, DV, dtype=torch.bfloat16)
    state_out = state.clone()
    for bi in range(B):
        sid = int(si[bi].item())
        if sid < 0: continue
        mx = mixed_qkv[bi].to(compute_dtype)
        Q = mx[:HK*DK].reshape(HK,DK); K = mx[HK*DK:2*HK*DK].reshape(HK,DK)
        V = mx[2*HK*DK:2*HK*DK+HV*DV].reshape(HV,DV)
        Q = Q / torch.sqrt((Q**2).sum(dim=-1,keepdim=True)+eps)
        K = K / torch.sqrt((K**2).sum(dim=-1,keepdim=True)+eps); Q = Q * scale_val
        a_f = a[bi].to(compute_dtype); b_f = b[bi].to(compute_dtype)
        gate = torch.exp(-torch.exp(a_log_f) * torch.log(1.0+torch.exp(a_f+dt_bias_f)))
        beta = torch.sigmoid(b_f)
        for hvi in range(HV):
            hki = hvi // G if G else hvi
            st = state_out[sid,hvi].to(compute_dtype).clone()
            if gate[hvi].item()!=1.0: st*=gate[hvi]
            o = V[hvi] - (st @ K[hki]); o *= beta[hvi]
            st += K[hki].unsqueeze(0) * o.unsqueeze(-1)
            o = st @ Q[hki]
            out[bi,0,hvi] = o.to(torch.bfloat16); state_out[sid,hvi] = st.to(state.dtype)
    return out, state_out


_E2E_COMBOS = [(8,8),(8,16),(8,24),(8,32),(8,64),(16,16),(16,32),(16,64)]

def test_e2e_combos():
    for HK, HV in _E2E_COMBOS:
        DV=DK=128
        mq,a,b,al,db,st,si = _make_inputs_e2e(HK,HV,DV,DK)
        ref_o,ref_s = golden_e2e(mq,a,b,al,db,st,si)
        npu_o,npu_s = _run_npu(mq,a,b,al,db,st,si)
        _check(npu_o,npu_s,ref_o,ref_s)
        print(f"    HK={HK},HV={HV}: PASS")
        import gc; gc.collect()
        torch.npu.empty_cache()


def test_batch():
    """Batch scaling: B=2,4,8 with DV=DK=128, NV=32."""
    for B in [2, 4, 8]:
        HK, HV, DV, DK, N = 8, 32, 128, 128, max(8, B)
        mq,a,b,al,db,st,si = _make_inputs_e2e(HK,HV,DV,DK,B=B,N=N)
        ref_o,ref_s = golden_e2e(mq,a,b,al,db,st,si)
        npu_o,npu_s = _run_npu(mq,a,b,al,db,st,si)
        _check(npu_o,npu_s,ref_o,ref_s)
        print(f"    B={B},HV={HV}: PASS  out max diff={((npu_o.to(torch.float32)-ref_o.to(torch.float32)).abs().max()):.6f}")
        import gc; gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    test_hv8_b1()
    test_hv8_b4()
    test_negative_sid()
    test_scale_override()
    test_hv16()
    print("--- e2e combos (DV=DK=128) ---")
    test_e2e_combos()
    print("--- batch scaling ---")
    test_batch()
    print("\nALL TESTS PASSED")
