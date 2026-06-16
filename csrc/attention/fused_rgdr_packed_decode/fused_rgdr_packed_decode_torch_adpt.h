/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_RGDR_PACKED_DECODE_TORCH_ADPT_H
#define FUSED_RGDR_PACKED_DECODE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_fused_rgdr_packed_decode(
    const at::Tensor& mixed_qkv,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    at::Tensor& state,
    const at::Tensor& ssm_state_indices,
    const c10::optional<double> scale)
{
    float scale_real = scale.has_value() ? static_cast<float>(scale.value()) : 1.0f;

    auto options = mixed_qkv.options().dtype(at::ScalarType::BFloat16);
    int64_t B = mixed_qkv.size(0);
    int64_t HV = a.size(1);
    int64_t DV = state.size(2);
    at::Tensor output = at::empty({B, 1, HV, DV}, options);

    EXEC_NPU_CMD(aclnnFusedRgdrPackedDecode,
                 mixed_qkv, a, b, a_log, dt_bias,
                 state, ssm_state_indices,
                 scale_real, output);
    return output;
}

} // namespace vllm_ascend
#endif
