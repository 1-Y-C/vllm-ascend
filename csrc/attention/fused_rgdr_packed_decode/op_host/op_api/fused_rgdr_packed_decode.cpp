/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "fused_rgdr_packed_decode.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedRgdrPackedDecode);

static constexpr FusedRgdrPackedDecodeOutput kNullOutput{nullptr, nullptr};

FusedRgdrPackedDecodeOutput FusedRgdrPackedDecode(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *stateRef, const aclTensor *ssmStateIndices,
    float scaleValue, aclOpExecutor *executor)
{
    L0_DFX(FusedRgdrPackedDecode, mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue);

    const DataType outDtype = DataType::DT_BF16;
    const Format format = Format::FORMAT_ND;

    auto out = executor->AllocTensor(outDtype, format, format);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."),
             return kNullOutput);

    auto sDtype = stateRef->GetDataType();
    auto stateOut = executor->AllocTensor(sDtype, format, format);
    OP_CHECK(stateOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "stateOut AllocTensor failed."),
             return kNullOutput);

    auto ret = INFER_SHAPE(FusedRgdrPackedDecode,
                           OP_INPUT(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices),
                           OP_OUTPUT(out, stateOut),
                           OP_ATTR(scaleValue));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return kNullOutput,
                        "FusedRgdrPackedDecode InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedRgdrPackedDecode,
                                      OP_INPUT(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices),
                                      OP_OUTPUT(out, stateOut),
                                      OP_ATTR(scaleValue));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return kNullOutput,
        "FusedRgdrPackedDecode ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return FusedRgdrPackedDecodeOutput{out, stateOut};
}

} // namespace l0op
