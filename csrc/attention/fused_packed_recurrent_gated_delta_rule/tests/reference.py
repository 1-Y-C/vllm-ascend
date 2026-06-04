"""
Reference PyTorch implementation of fused packed recurrent gated delta rule.

This implements the exact same math as the AscendC kernel for correctness
verification. All computation is in float32 for maximum precision.

The math (per token, per value-head):
  1. Read q, k, v from packed mixed_qkv layout
  2. L2Norm q and k
  3. Scale q by head_k_dim ** -0.5
  4. g = -exp(A_log) * softplus(a + dt_bias)
  5. beta = sigmoid(b)
  6. h *= exp(g)
  7. delta = v - sum(h * k, dim=1)
  8. delta *= beta
  9. h += outer(delta, k)
  10. out = sum(h * q, dim=1)
"""

import torch


def fused_packed_recurrent_gated_delta_rule_ref(
    mixed_qkv: torch.Tensor,   # [B, qkv_dim] bf16
    a: torch.Tensor,           # [B, HV] bf16
    b: torch.Tensor,           # [B, HV] bf16
    A_log: torch.Tensor,       # [HV] float32
    dt_bias: torch.Tensor,     # [HV] float32
    ssm_state: torch.Tensor,   # [N, HV, V, K] float32
    ssm_state_indices: torch.Tensor,  # [B] int32/int64
    head_k_dim: int,
    head_v_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    scale: float | None = None,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """
    Reference implementation of the fused packed decode kernel.

    Returns:
        out: [B, HV, V] bf16
    Side effect:
        ssm_state is updated in-place.
    """
    B = mixed_qkv.shape[0]
    HV = num_v_heads
    K = head_k_dim
    V = head_v_dim
    H = num_k_heads
    qkv_dim = mixed_qkv.shape[1]

    if scale is None:
        scale = K ** -0.5

    device = mixed_qkv.device
    out = torch.empty(B, HV, V, dtype=torch.bfloat16, device=device)

    # Work in fp32 for accuracy.
    mq = mixed_qkv.float()
    a_f = a.float()
    b_f = b.float()
    al_f = A_log.float()
    dt_f = dt_bias.float()

    for bi in range(B):
        state_idx = ssm_state_indices[bi].item()
        if state_idx <= 0:
            continue

        for hvi in range(HV):
            hi = hvi // (HV // H)

            # Read q, k, v from packed layout.
            q_off = hi * K
            k_off = H * K + hi * K
            v_off = 2 * H * K + hvi * V

            q = mq[bi, q_off:q_off + K]
            k = mq[bi, k_off:k_off + K]
            v = mq[bi, v_off:v_off + V]

            # L2Norm.
            q = q / (torch.sqrt(torch.sum(q * q) + 1e-6))
            k = k / (torch.sqrt(torch.sum(k * k) + 1e-6))

            # Scale.
            q = q * scale

            # Gating.
            av = a_f[bi, hvi]
            bv = b_f[bi, hvi]
            al_v = al_f[hvi]
            dt_v = dt_f[hvi]

            x = av + dt_v
            if x <= softplus_threshold:
                sp = torch.log(1.0 + torch.exp(x))
            else:
                sp = x
            g = -torch.exp(al_v) * sp
            beta = torch.sigmoid(bv)

            # Load state h [V, K].
            h = ssm_state[state_idx, hvi].clone()

            # Recurrence.
            h *= torch.exp(g)
            delta = v - (h * k.unsqueeze(0)).sum(dim=1)
            delta *= beta
            h += delta.unsqueeze(1) * k.unsqueeze(0)
            o = (h * q.unsqueeze(0)).sum(dim=1)

            out[bi, hvi] = o.to(torch.bfloat16)
            ssm_state[state_idx, hvi] = h

    return out


def compute_gdn_gating_ref(
    a: torch.Tensor,      # [B, HV]
    b: torch.Tensor,      # [B, HV]
    A_log: torch.Tensor,  # [HV]
    dt_bias: torch.Tensor, # [HV]
    softplus_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference gating computation (same as fused_gdn_gating).
    Returns g [B, HV] float32, beta [B, HV] float32.
    """
    x = a.float() + dt_bias.float().unsqueeze(0)
    sp = torch.where(x <= softplus_threshold, torch.log(1.0 + torch.exp(x)), x)
    neg_exp_A = -torch.exp(A_log.float()).unsqueeze(0)
    g = neg_exp_A * sp
    beta = torch.sigmoid(b.float())
    return g, beta
