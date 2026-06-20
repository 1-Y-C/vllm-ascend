# SPDX-License-Identifier: Apache-2.0
"""Correctness test for AscendC fused_rgdr_packed_decode kernel.

Validates torch.ops._C_ascend.npu_fused_rgdr_packed_decode against a
CPU golden reference across batch / head / dimension combinations.

Prerequisite: the AscendC kernel must be compiled and installed via
  bash csrc/build_aclnn.sh <ROOT_DIR> <SOC_VERSION>

Run:
  pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_rgdr_packed_decode.py -v
"""

import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42

# Reference implementation
# ---------------------------------------------------------------------------


def _golden_fused_rgdr_packed_decode(
    mixed_qkv: torch.Tensor,  # [B, L] bf16  (packed Q,K,V)
    a: torch.Tensor,           # [B, HV] bf16
    b: torch.Tensor,           # [B, HV] bf16
    a_log: torch.Tensor,       # [HV] fp32
    dt_bias: torch.Tensor,     # [HV] fp32
    state: torch.Tensor,       # [N, HV, DV, DK] float/bf16
    ssm_state_indices: torch.Tensor,  # [B] int32
    scale_val: float = 1.0,
):
    """CPU golden reference for fused_rgdr_packed_decode."""
    compute_dtype = torch.float32
    B = mixed_qkv.shape[0]
    L = mixed_qkv.shape[1]
    HV = a.shape[1]
    DV = state.shape[2]
    DK = state.shape[3]

    HK = (L - HV * DV) // (2 * DK)
    assert HK > 0, f"Invalid HK={HK}: L={L}, HV={HV}, DV={DV}, DK={DK}"
    assert L == 2 * HK * DK + HV * DV, f"L mismatch: L={L} vs 2*{HK}*{DK}+{HV}*{DV}={2*HK*DK+HV*DV}"

    G = HV // HK
    eps = 1e-6

    a_log_f = a_log.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    # Pre-compute gates (same for all batch items sharing the same a_log/dt_bias)
    # gate = exp(-exp(a_log) * softplus(a + dt_bias))
    # softplus(x) = log(1 + exp(x))
    gate_cache: dict[int, torch.Tensor] = {}

    out = torch.zeros(B, 1, HV, DV, dtype=torch.bfloat16)
    state_out = state.clone()

    for bi in range(B):
        sid = int(ssm_state_indices[bi].item())
        if sid < 0:
            continue  # output stays zero, state unchanged

        # Unpack mixed_qkv
        mx = mixed_qkv[bi].to(compute_dtype)  # [L]
        q_off = 0
        k_off = HK * DK
        v_off = 2 * HK * DK
        Q = mx[q_off : q_off + HK * DK].reshape(HK, DK)   # [HK, DK]
        K = mx[k_off : k_off + HK * DK].reshape(HK, DK)   # [HK, DK]
        V = mx[v_off : v_off + HV * DV].reshape(HV, DV)   # [HV, DV]

        # L2 normalize Q
        for h in range(HK):
            q_norm = torch.sqrt((Q[h] ** 2).sum() + eps)
            Q[h] = Q[h] / q_norm

        # L2 normalize K
        for h in range(HK):
            k_norm = torch.sqrt((K[h] ** 2).sum() + eps)
            K[h] = K[h] / k_norm

        # Scale Q
        Q = Q * scale_val

        # Gate computation
        if bi not in gate_cache:
            a_f = a[bi].to(compute_dtype)  # [HV]
            gate = torch.exp(-torch.exp(a_log_f) * torch.log(1.0 + torch.exp(a_f + dt_bias_f)))  # [HV]
            gate_cache[bi] = gate

        gate = gate_cache[bi]

        # Beta: sigmoid(b)
        beta = torch.sigmoid(b[bi].to(compute_dtype))  # [HV]

        for hvi in range(HV):
            hki = hvi // G if G > 0 else hvi

            # Read state for this head
            st = state_out[sid, hvi].to(compute_dtype).clone()  # [DV, DK]

            # Apply exponential decay
            if gate[hvi].item() != 1.0:
                st = st * gate[hvi]

            # Compute delta: o[dvi] = V[hvi][dvi] - sum(K[hki] * st[dvi])
            o_fp32 = torch.zeros(DV, dtype=compute_dtype)
            for dvi in range(DV):
                dot_val = torch.dot(K[hki], st[dvi])
                o_fp32[dvi] = V[hvi, dvi] - dot_val

            # Apply beta
            o_fp32 = o_fp32 * beta[hvi]

            # State update: st[dvi] += K[hki] * o[dvi]
            for dvi in range(DV):
                if o_fp32[dvi].item() != 0.0:
                    st[dvi] = st[dvi] + K[hki] * o_fp32[dvi]

            # Output: o[dvi] = sum(Q[hki] * st[dvi])
            for dvi in range(DV):
                o_fp32[dvi] = torch.dot(Q[hki], st[dvi])

            out[bi, 0, hvi] = o_fp32.to(torch.bfloat16)
            state_out[sid, hvi] = st.to(state.dtype)

    return out, state_out


# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    B: int,
    HV: int,
    DV: int,
    DK: int,
    N: int = 8,
    state_dtype: torch.dtype = torch.float32,
    seed: int = SEED,
):
    """Build random tensors on CPU."""
    torch.manual_seed(seed)
    HK = HV  # G=1, HK=HV
    L = 2 * HK * DK + HV * DV

    mixed_qkv = torch.randn(B, L, dtype=torch.bfloat16)
    a = torch.randn(B, HV, dtype=torch.bfloat16)
    b = torch.randn(B, HV, dtype=torch.bfloat16)
    a_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    state = torch.randn(N, HV, DV, DK, dtype=state_dtype)
    ssm_state_indices = torch.randint(0, N, (B,), dtype=torch.int32)

    return mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices


def _npu_op_exec(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale_val: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute on NPU."""
    state_npu = state.npu().clone()
    out_npu = torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
        mixed_qkv.npu(), a.npu(), b.npu(),
        a_log.npu(), dt_bias.npu(),
        state_npu, ssm_state_indices.npu(),
        scale_val,
    )
    return out_npu.cpu(), state_npu.cpu()


def _assert_close(actual, actual_state, ref, ref_state, rtol=5e-2, atol=1e-1):
    torch.testing.assert_close(
        actual.to(torch.float32), ref.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )
    torch.testing.assert_close(
        actual_state.to(torch.float32), ref_state.to(torch.float32),
        rtol=rtol, atol=atol, equal_nan=True,
    )


# Tests
# ---------------------------------------------------------------------------


class TestFusedRgdrPackedDecode:

    @pytest.mark.parametrize("HV", [4, 8, 16])
    @pytest.mark.parametrize("B", [1, 4])
    def test_small_shapes(self, B, HV):
        """Basic correctness on small shapes."""
        DV, DK = 8, 16
        mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
        ref_out, ref_state = _golden_fused_rgdr_packed_decode(
            mixed_qkv, a, b, a_log, dt_bias, state, si)
        npu_out, npu_state = _npu_op_exec(
            mixed_qkv, a, b, a_log, dt_bias, state, si)
        _assert_close(npu_out, npu_state, ref_out, ref_state)
        gc.collect()
        torch.npu.empty_cache()

    @pytest.mark.parametrize("HV", [32, 64])
    @pytest.mark.parametrize("B", [1, 8])
    def test_medium_heads(self, B, HV):
        """Medium head counts."""
        DV, DK = 8, 16
        mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
        ref_out, ref_state = _golden_fused_rgdr_packed_decode(
            mixed_qkv, a, b, a_log, dt_bias, state, si)
        npu_out, npu_state = _npu_op_exec(
            mixed_qkv, a, b, a_log, dt_bias, state, si)
        _assert_close(npu_out, npu_state, ref_out, ref_state)
        gc.collect()
        torch.npu.empty_cache()

    def test_output_shapes(self):
        """Verify output tensor shapes and dtypes."""
        B, HV, DV, DK = 2, 8, 8, 16
        mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
        state_npu = state.npu().clone()
        out_npu = torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
            mixed_qkv.npu(), a.npu(), b.npu(),
            a_log.npu(), dt_bias.npu(),
            state_npu, si.npu(), 1.0,
        )
        assert out_npu.shape == (B, 1, HV, DV), f"bad out shape: {out_npu.shape}"
        assert out_npu.dtype == torch.bfloat16, f"bad out dtype: {out_npu.dtype}"
        assert state_npu.shape == state.shape, f"bad state shape: {state_npu.shape}"
        gc.collect()
        torch.npu.empty_cache()

    def test_negative_sid_skip(self):
        """Negative ssm_state_indices should output zeros and preserve state."""
        B, HV, DV, DK = 2, 4, 8, 8
        mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
        si[0] = -1  # First item skips
        state_npu = state.npu().clone()
        out_npu = torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
            mixed_qkv.npu(), a.npu(), b.npu(),
            a_log.npu(), dt_bias.npu(),
            state_npu, si.npu(), 1.0,
        )
        out_cpu = out_npu.cpu()
        state_cpu = state_npu.cpu()
        # First batch item output should be all zeros
        assert (out_cpu[0] == 0).all(), f"skip output not zero: {out_cpu[0]}"
        # State for skipped indices should be unchanged
        torch.testing.assert_close(state_cpu[0], state[0].to(torch.float32), rtol=0, atol=0)
        gc.collect()
        torch.npu.empty_cache()

    def test_scale_override(self):
        """Non-default scale value."""
        B, HV, DV, DK = 1, 4, 8, 16
        scale = 0.5
        mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(B, HV, DV, DK)
        ref_out, ref_state = _golden_fused_rgdr_packed_decode(
            mixed_qkv, a, b, a_log, dt_bias, state, si, scale_val=scale)
        npu_out, npu_state = _npu_op_exec(
            mixed_qkv, a, b, a_log, dt_bias, state, si, scale_val=scale)
        _assert_close(npu_out, npu_state, ref_out, ref_state)
        gc.collect()
        torch.npu.empty_cache()
