#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_PACKED_RECURRENT_GATED_DELTA_RULE
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_PACKED_RECURRENT_GATED_DELTA_RULE

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *PackedRecurrentGatedDeltaRule(const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
                                               const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
                                               const aclTensor *ssmStateIndices, float scaleValue,
                                               aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_PACKED_RECURRENT_GATED_DELTA_RULE
