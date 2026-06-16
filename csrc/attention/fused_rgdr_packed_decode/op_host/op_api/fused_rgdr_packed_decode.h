/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_RGDR_PACKED_DECODE_L0OP_H
#define FUSED_RGDR_PACKED_DECODE_L0OP_H

#include "aclnn/aclnn_base.h"

namespace l0op {

struct FusedRgdrPackedDecodeOutput {
    aclTensor *out{nullptr};
};

FusedRgdrPackedDecodeOutput FusedRgdrPackedDecode(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias,
    aclTensor *stateRef, const aclTensor *ssmStateIndices,
    float scaleValue, aclOpExecutor *executor);

} // namespace l0op

#endif
