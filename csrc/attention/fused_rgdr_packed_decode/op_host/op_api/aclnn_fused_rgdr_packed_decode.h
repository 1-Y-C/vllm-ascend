/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef ACLNN_FUSED_RGDR_PACKED_DECODE_H
#define ACLNN_FUSED_RGDR_PACKED_DECODE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnFusedRgdrPackedDecodeGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias,
    aclTensor *stateRef, const aclTensor *ssmStateIndices,
    float scaleValue, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnFusedRgdrPackedDecode(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
