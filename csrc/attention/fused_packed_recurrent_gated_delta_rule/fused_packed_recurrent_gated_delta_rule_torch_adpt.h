/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H
#define FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_fused_packed_recurrent_gated_delta_rule(
    const at::Tensor& mixed_qkv,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    at::Tensor& state,
    const at::Tensor& ssm_state_indices,
    const c10::optional<double> scale)
{
    float s = scale.has_value() ? static_cast<float>(scale.value()) : 1.0f;

    int64_t B = mixed_qkv.size(0);
    int64_t HV = a.size(1);
    int64_t DV = state.size(2);
    auto opts = mixed_qkv.options().dtype(at::ScalarType::BFloat16);
    at::Tensor out = at::empty({B, 1, HV, DV}, opts);

    EXEC_NPU_CMD(aclnnFusedPackedRecurrentGatedDeltaRule,
                 mixed_qkv, a, b, a_log, dt_bias, state,
                 ssm_state_indices, s, out);
    return out;
}

} // namespace vllm_ascend

#endif // FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H
