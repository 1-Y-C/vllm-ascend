#ifndef OP_API_ACLNN_PACKED_RECURRENT_GATED_DELTA_RULE_H
#define OP_API_ACLNN_PACKED_RECURRENT_GATED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnPackedRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
    const aclTensor *ssmStateIndices, float scaleValue, aclTensor *attnOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnPackedRecurrentGatedDeltaRule(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_PACKED_RECURRENT_GATED_DELTA_RULE_H
