/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef ACLNN_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_H
#define ACLNN_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnFusedPackedRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
    const aclTensor *ssmStateIndices, float scaleValue, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFusedPackedRecurrentGatedDeltaRule(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_H
