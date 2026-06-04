/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include <dlfcn.h>
#include "aclnn_fused_packed_recurrent_gated_delta_rule.h"
#include "../fused_packed_recurrent_gated_delta_rule.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct Params {
    const aclTensor *mixedQkv{nullptr};
    const aclTensor *a{nullptr};
    const aclTensor *b{nullptr};
    const aclTensor *aLog{nullptr};
    const aclTensor *dtBias{nullptr};
    const aclTensor *stateRef{nullptr};
    const aclTensor *ssmStateIndices{nullptr};
    float scaleValue{1.0f};
    const aclTensor *out{nullptr};
};

static const std::initializer_list<op::DataType> BF16_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> FP32_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ST_LIST   = {op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> I32_LIST  = {op::DataType::DT_INT32};

static bool CheckNotNull(const Params &p)
{
    OP_CHECK_NULL(p.mixedQkv, return false);
    OP_CHECK_NULL(p.a, return false);
    OP_CHECK_NULL(p.b, return false);
    OP_CHECK_NULL(p.aLog, return false);
    OP_CHECK_NULL(p.dtBias, return false);
    OP_CHECK_NULL(p.stateRef, return false);
    OP_CHECK_NULL(p.ssmStateIndices, return false);
    OP_CHECK_NULL(p.out, return false);
    return true;
}

static bool CheckDtype(const Params &p)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(p.mixedQkv, BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.a, BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.b, BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.aLog, FP32_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.dtBias, FP32_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.stateRef, ST_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.ssmStateIndices, I32_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.out, BF16_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(Params &p)
{
    CHECK_RET(CheckDtype(p), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(Params &p)
{
    p.mixedQkv->SetOriginalShape(p.mixedQkv->GetViewShape());
    p.a->SetOriginalShape(p.a->GetViewShape());
    p.b->SetOriginalShape(p.b->GetViewShape());
    p.aLog->SetOriginalShape(p.aLog->GetViewShape());
    p.dtBias->SetOriginalShape(p.dtBias->GetViewShape());
    p.stateRef->SetOriginalShape(p.stateRef->GetViewShape());
    p.ssmStateIndices->SetOriginalShape(p.ssmStateIndices->GetViewShape());
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnFusedPackedRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
    const aclTensor *ssmStateIndices, float scaleValue, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedPackedRecurrentGatedDeltaRule,
                   DFX_IN(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue),
                   DFX_OUT(out, stateRef));

    auto ex = CREATE_EXECUTOR();
    CHECK_RET(ex.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    Params p{mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue, out};
    CHECK_RET(CheckNotNull(p), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(p);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto mq_  = l0op::Contiguous(mixedQkv, ex.get());
    auto a_   = l0op::Contiguous(a, ex.get());
    auto b_   = l0op::Contiguous(b, ex.get());
    auto al_  = l0op::Contiguous(aLog, ex.get());
    auto dt_  = l0op::Contiguous(dtBias, ex.get());
    auto ssi_ = l0op::Contiguous(ssmStateIndices, ex.get());
    auto o_   = l0op::Contiguous(out, ex.get());

    auto outRet = l0op::FusedPackedRecurrentGatedDeltaRule(
        mq_, a_, b_, al_, dt_, stateRef, ssi_, scaleValue, ex.get());
    if (!outRet) return ACLNN_ERR_INNER_NULLPTR;

    auto vcr = l0op::ViewCopy(outRet, o_, ex.get());
    if (!vcr) return ACLNN_ERR_INNER_NULLPTR;

    *workspaceSize = ex->GetWorkspaceSize();
    ex.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedPackedRecurrentGatedDeltaRule(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedPackedRecurrentGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
