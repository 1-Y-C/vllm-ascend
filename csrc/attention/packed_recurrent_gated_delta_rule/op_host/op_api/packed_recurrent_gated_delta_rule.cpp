#include "../packed_recurrent_gated_delta_rule.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(PackedRecurrentGatedDeltaRule);

const aclTensor *PackedRecurrentGatedDeltaRule(const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
                                               const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
                                               const aclTensor *ssmStateIndices, float scaleValue,
                                               aclOpExecutor *executor)
{
    L0_DFX(PackedRecurrentGatedDeltaRule, mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scaleValue);

    DataType outType = DataType::DT_BF16;
    Format format = Format::FORMAT_ND;

    auto out = executor->AllocTensor(outType, format, format);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."), return nullptr);

    auto ret = INFER_SHAPE(
        PackedRecurrentGatedDeltaRule,
        OP_INPUT(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices),
        OP_OUTPUT(out, stateRef), OP_ATTR(scaleValue));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "PackedRecurrentGatedDeltaRule InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        PackedRecurrentGatedDeltaRule,
        OP_INPUT(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices),
        OP_OUTPUT(out, stateRef), OP_ATTR(scaleValue));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "PackedRecurrentGatedDeltaRule ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}
} // namespace l0op
