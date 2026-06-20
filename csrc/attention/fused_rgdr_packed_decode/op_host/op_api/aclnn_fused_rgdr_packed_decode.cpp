/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include <dlfcn.h>
#include "aclnn_fused_rgdr_packed_decode.h"
#include "fused_rgdr_packed_decode.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct FusedRgdrPackedDecodeParams {
    const aclTensor *mixedQkv{nullptr};
    const aclTensor *a{nullptr};
    const aclTensor *b{nullptr};
    const aclTensor *aLog{nullptr};
    const aclTensor *dtBias{nullptr};
    aclTensor *stateRef{nullptr};
    const aclTensor *ssmStateIndices{nullptr};
    float scaleValue{1.0f};
    aclTensor *out{nullptr};
};

static const std::initializer_list<op::DataType> BF16_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> FP32_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> INT32_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const FusedRgdrPackedDecodeParams &params)
{
    OP_CHECK_NULL(params.mixedQkv,         return false);
    OP_CHECK_NULL(params.a,                return false);
    OP_CHECK_NULL(params.b,                return false);
    OP_CHECK_NULL(params.aLog,             return false);
    OP_CHECK_NULL(params.dtBias,           return false);
    OP_CHECK_NULL(params.stateRef,         return false);
    OP_CHECK_NULL(params.ssmStateIndices,  return false);
    OP_CHECK_NULL(params.out,              return false);
    return true;
}

static inline bool CheckDtype(const FusedRgdrPackedDecodeParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.mixedQkv,        BF16_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a,               BF16_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.b,               BF16_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.aLog,            FP32_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBias,          FP32_TYPE_SUPPORT_LIST,  return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.stateRef,        STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ssmStateIndices, INT32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.out,             BF16_TYPE_SUPPORT_LIST,  return false);
    return true;
}

static aclnnStatus CheckParams(const FusedRgdrPackedDecodeParams &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(params),   ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnFusedRgdrPackedDecodeGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias,
    aclTensor *stateRef, const aclTensor *ssmStateIndices,
    float scaleValue, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedRgdrPackedDecode,
                   DFX_IN(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue),
                   DFX_OUT(out, stateRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedRgdrPackedDecodeParams params{mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue, out};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto mq    = l0op::Contiguous(mixedQkv,        uniqueExecutor.get());
    auto aContig    = l0op::Contiguous(a,               uniqueExecutor.get());
    auto bContig    = l0op::Contiguous(b,               uniqueExecutor.get());
    auto al    = l0op::Contiguous(aLog,            uniqueExecutor.get());
    auto db    = l0op::Contiguous(dtBias,          uniqueExecutor.get());
    auto stContig = l0op::Contiguous(stateRef,        uniqueExecutor.get());
    auto si       = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    CHECK_RET(mq != nullptr && aContig != nullptr && bContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(al != nullptr && db != nullptr && stContig != nullptr && si != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::FusedRgdrPackedDecode(mq, aContig, bContig, al, db, stContig, si,
                                              scaleValue, uniqueExecutor.get());
    CHECK_RET(result.out != nullptr && result.stateOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto vc = l0op::ViewCopy(result.out, out, uniqueExecutor.get());
    CHECK_RET(vc != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vcSt = l0op::ViewCopy(result.stateOut, stateRef, uniqueExecutor.get());
    CHECK_RET(vcSt != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedRgdrPackedDecode(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedRgdrPackedDecode);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
