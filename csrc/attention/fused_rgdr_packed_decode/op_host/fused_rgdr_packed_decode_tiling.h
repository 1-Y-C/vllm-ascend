/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_RGDR_PACKED_DECODE_TILING_H
#define FUSED_RGDR_PACKED_DECODE_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>

namespace optiling {

struct FusedRgdrPackedDecodeCompileInfo {};

ge::graphStatus FusedRgdrPackedDecodeTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForFusedRgdrPackedDecode(gert::TilingParseContext *context);

} // namespace optiling

#endif
