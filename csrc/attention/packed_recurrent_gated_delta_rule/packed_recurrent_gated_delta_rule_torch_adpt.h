#ifndef PACKED_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H
#define PACKED_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_packed_recurrent_gated_delta_rule(
    const at::Tensor& mixed_qkv,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    at::Tensor& state,
    const at::Tensor& ssm_state_indices,
    const double scale)
{
    TORCH_CHECK(scale > 0, "scale must be positive");
    auto options = state.options().dtype(at::ScalarType::BFloat16);
    int64_t B = mixed_qkv.size(0);
    int64_t HV = a.size(1);
    int64_t DV = state.size(2);
    at::Tensor attn_out = at::empty({B, 1, HV, DV}, options);
    float scale_float = static_cast<float>(scale);
    EXEC_NPU_CMD(aclnnPackedRecurrentGatedDeltaRule,
                 mixed_qkv, a, b, a_log, dt_bias,
                 state, ssm_state_indices,
                 scale_float,
                 attn_out);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, state);
}

} // namespace vllm_ascend
#endif
