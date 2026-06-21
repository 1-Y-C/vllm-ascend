# SPDX-License-Identifier: Apache-2.0
"""Correctness test for AscendC fused_rgdr_packed_decode kernel.

Validates torch.ops._C_ascend.npu_fused_rgdr_packed_decode against a
CPU golden reference.

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
DV = DK = 128
B = 1

# (HK, HV) combos where HV is a multiple of HK (valid G)
_HK_HV_COMBOS = [
    (8, 8),
    (8, 16),
    (8, 24),
    (8, 32),
    (8, 64),
    (16, 16),
    (16, 32),
    (16, 64),
]


# Reference implementation (vectorized for large DV/DK)
# ---------------------------------------------------------------------------

def _golden_fused_rgdr_packed_decode(
    mixed_qkv: torch.Tensor,  # [B, L] bf16
    a: torch.Tensor,           # [B, HV] bf16
    b: torch.Tensor,           # [B, HV] bf16
    a_log: torch.Tensor,       # [HV] fp32
    dt_bias: torch.Tensor,     # [HV] fp32
    state: torch.Tensor,       # [N, HV, DV, DK] float
    ssm_state_indices: torch.Tensor,  # [B] int32
    scale_val: float = 1.0,
):
    """CPU golden, vectorized einsum for large DV/DK."""
    compute_dtype = torch.float32
    B = mixed_qkv.shape[0]; L = mixed_qkv.shape[1]
    HV = a.shape[1]; DV = state.shape[2]; DK = state.shape[3]
    HK = (L - HV * DV) // (2 * DK)
    G = HV // HK
    eps = 1e-6

    a_log_f = a_log.to(compute_dtype); dt_bias_f = dt_bias.to(compute_dtype)
    out = torch.zeros(B, 1, HV, DV, dtype=torch.bfloat16)
    state_out = state.clone()
    gate_cache: dict[int, torch.Tensor] = {}

    for bi in range(B):
        sid = int(ssm_state_indices[bi].item())
        if sid < 0:
            continue

        mx = mixed_qkv[bi].to(compute_dtype)
        Q = mx[:HK * DK].reshape(HK, DK)
        K = mx[HK * DK : 2 * HK * DK].reshape(HK, DK)
        V = mx[2 * HK * DK : 2 * HK * DK + HV * DV].reshape(HV, DV)

        Q = Q / torch.sqrt((Q ** 2).sum(dim=-1, keepdim=True) + eps)
        K = K / torch.sqrt((K ** 2).sum(dim=-1, keepdim=True) + eps)
        Q = Q * scale_val

        if bi not in gate_cache:
            a_f = a[bi].to(compute_dtype)
            gate = torch.exp(-torch.exp(a_log_f) * torch.log(1.0 + torch.exp(a_f + dt_bias_f)))
            gate_cache[bi] = gate
        gate = gate_cache[bi]
        beta = torch.sigmoid(b[bi].to(compute_dtype))

        for hvi in range(HV):
            hki = hvi // G if G > 0 else hvi

            st = state_out[sid, hvi].to(compute_dtype).clone()
            if gate[hvi].item() != 1.0:
                st *= gate[hvi]

            o = V[hvi] - (st @ K[hki])  # [DV]
            o *= beta[hvi]

            delta = K[hki].unsqueeze(0) * o.unsqueeze(-1)  # [DV, DK]
            st += delta

            o = st @ Q[hki]  # [DV]
            out[bi, 0, hvi] = o.to(torch.bfloat16)
            state_out[sid, hvi] = st.to(state.dtype)

    return out, state_out


# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(HK: int, HV: int, N: int = 8, seed: int = SEED):
    torch.manual_seed(seed)
    L = 2 * HK * DK + HV * DV
    mixed_qkv = torch.randn(B, L, dtype=torch.bfloat16)
    a = torch.randn(B, HV, dtype=torch.bfloat16)
    b = torch.randn(B, HV, dtype=torch.bfloat16)
    a_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    state = torch.randn(N, HV, DV, DK, dtype=torch.float32)
    ssm_state_indices = torch.randperm(N)[:B].to(torch.int32)
    return mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices


def _npu_op_exec(mixed_qkv, a, b, a_log, dt_bias, state, si, scale_val=1.0):
    state_npu = state.npu().clone()
    out_npu = torch.ops._C_ascend.npu_fused_rgdr_packed_decode(
        mixed_qkv.npu(), a.npu(), b.npu(),
        a_log.npu(), dt_bias.npu(),
        state_npu, si.npu(), scale_val,
    )
    return out_npu.cpu(), state_npu.cpu()


def _assert_close(actual, actual_state, ref, ref_state, rtol=1e-2, atol=1e-1):
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

    @pytest.mark.parametrize("HK,HV", _HK_HV_COMBOS)
    def test_combos(self, HK, HV):
        G = HV // HK
        L = 2 * HK * DK + HV * DV
        mixed_qkv, a, b, a_log, dt_bias, state, si = _make_inputs(HK, HV)
        ref_out, ref_state = _golden_fused_rgdr_packed_decode(
            mixed_qkv, a, b, a_log, dt_bias, state, si)
        npu_out, npu_state = _npu_op_exec(
            mixed_qkv, a, b, a_log, dt_bias, state, si)
        _assert_close(npu_out, npu_state, ref_out, ref_state)
        gc.collect()
        torch.npu.empty_cache()
