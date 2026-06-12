/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
#define __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>

namespace optiling {

struct FusedPackedRecurrentGatedDeltaRuleCompileInfo {};

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForFusedPackedRecurrentGatedDeltaRule(gert::TilingParseContext *context);

} // namespace optiling

#endif // __OP_HOST_FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
