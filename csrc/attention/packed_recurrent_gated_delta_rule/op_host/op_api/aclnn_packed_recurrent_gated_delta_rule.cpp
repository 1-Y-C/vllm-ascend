#include "aclnn_packed_recurrent_gated_delta_rule.h"
#include "../packed_recurrent_gated_delta_rule.h"

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
constexpr size_t MIXED_QKV_DIM_NUM = 2;
constexpr size_t A_DIM_NUM = 2;
constexpr size_t B_DIM_NUM = 2;
constexpr size_t A_LOG_DIM_NUM = 1;
constexpr size_t DT_BIAS_DIM_NUM = 1;
constexpr size_t STATE_DIM_NUM = 4;
constexpr size_t SSM_INDICES_DIM_NUM = 2;
constexpr size_t ATTN_OUT_DIM_NUM = 4;

struct PackedRecurrentGatedDeltaRuleParams {
    const aclTensor *mixedQkv {nullptr};
    const aclTensor *a {nullptr};
    const aclTensor *b {nullptr};
    const aclTensor *aLog {nullptr};
    const aclTensor *dtBias {nullptr};
    const aclTensor *state {nullptr};
    const aclTensor *ssmStateIndices {nullptr};
    float scale {1.0f};
    const aclTensor *attnOut {nullptr};
};

static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> FLOAT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INT32_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};

static inline bool CheckNotNull(const PackedRecurrentGatedDeltaRuleParams &params)
{
    OP_CHECK_NULL(params.mixedQkv, return false);
    OP_CHECK_NULL(params.a, return false);
    OP_CHECK_NULL(params.b, return false);
    OP_CHECK_NULL(params.aLog, return false);
    OP_CHECK_NULL(params.dtBias, return false);
    OP_CHECK_NULL(params.state, return false);
    OP_CHECK_NULL(params.ssmStateIndices, return false);
    OP_CHECK_NULL(params.attnOut, return false);
    return true;
}

static inline bool CheckDtypeVaild(const PackedRecurrentGatedDeltaRuleParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.mixedQkv, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.b, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.aLog, FLOAT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBias, FLOAT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ssmStateIndices, INT32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.attnOut, OUT_TYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(PackedRecurrentGatedDeltaRuleParams &params)
{
    CHECK_RET(CheckDtypeVaild(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("PackedRecurrentGatedDeltaRule check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(PackedRecurrentGatedDeltaRuleParams &params)
{
    params.mixedQkv->SetOriginalShape(params.mixedQkv->GetViewShape());
    params.a->SetOriginalShape(params.a->GetViewShape());
    params.b->SetOriginalShape(params.b->GetViewShape());
    params.aLog->SetOriginalShape(params.aLog->GetViewShape());
    params.dtBias->SetOriginalShape(params.dtBias->GetViewShape());
    params.state->SetOriginalShape(params.state->GetViewShape());
    params.ssmStateIndices->SetOriginalShape(params.ssmStateIndices->GetViewShape());
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnPackedRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
    const aclTensor *ssmStateIndices, float scaleValue, aclTensor *attnOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnPackedRecurrentGatedDeltaRule,
                   DFX_IN(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue),
                   DFX_OUT(attnOut, stateRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    PackedRecurrentGatedDeltaRuleParams params{mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue, attnOut};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto mixedQkv_ = l0op::Contiguous(mixedQkv, uniqueExecutor.get());
    auto a_ = l0op::Contiguous(a, uniqueExecutor.get());
    auto b_ = l0op::Contiguous(b, uniqueExecutor.get());
    auto aLog_ = l0op::Contiguous(aLog, uniqueExecutor.get());
    auto dtBias_ = l0op::Contiguous(dtBias, uniqueExecutor.get());
    auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    auto attnOut_ = l0op::Contiguous(attnOut, uniqueExecutor.get());

    auto outRet = l0op::PackedRecurrentGatedDeltaRule(
        mixedQkv_, a_, b_, aLog_, dtBias_, stateRef, ssmStateIndices_, scaleValue, uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, attnOut_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPackedRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize,
                                               aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPackedRecurrentGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
