/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
#define __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/error_log.h"
#include "../op_kernel/fused_packed_recurrent_gated_delta_rule_tiling_data.h"

namespace optiling {
using namespace FusedPackedRecurrentGatedDeltaRule;

struct FusedPackedRecurrentGatedDeltaRuleCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

class FusedPackedRecurrentGatedDeltaRuleTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FusedPackedRecurrentGatedDeltaRuleTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    }
    ~FusedPackedRecurrentGatedDeltaRuleTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void InitCompileInfo();
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus CalUbSize();
    ge::graphStatus GetScaleAttr();
    ge::graphStatus RuleCheckShape();
    ge::graphStatus RuleFillShapeData();
    ge::graphStatus RuleUpdateBlockDim();

    FusedPackedRecurrentGatedDeltaRuleCompileInfo compileInfo_;
    FusedPackedRecurrentGatedDeltaRuleTilingData tilingData_;
    int64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
};

} // namespace optiling

#endif // __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
