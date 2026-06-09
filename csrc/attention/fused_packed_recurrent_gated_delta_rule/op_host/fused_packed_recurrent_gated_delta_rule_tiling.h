/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
#define __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling_base/error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedPackedRecurrentGatedDeltaRuleTilingData)
TILING_DATA_FIELD_DEF(uint32_t, vectorCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, b);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, hv);
TILING_DATA_FIELD_DEF(uint32_t, dk);
TILING_DATA_FIELD_DEF(uint32_t, dv);
TILING_DATA_FIELD_DEF(uint32_t, qkvDim);
TILING_DATA_FIELD_DEF(uint32_t, sBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

struct FusedPackedRecurrentGatedDeltaRuleCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

REGISTER_TILING_DATA_CLASS(FusedPackedRecurrentGatedDeltaRule,
                           FusedPackedRecurrentGatedDeltaRuleTilingData)
} // namespace optiling

#endif
