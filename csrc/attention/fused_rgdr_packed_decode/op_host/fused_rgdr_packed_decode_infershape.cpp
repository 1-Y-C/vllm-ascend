/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

using namespace gert;

namespace ops {

static ge::graphStatus InferShapeFusedRgdrPackedDecode(InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto mixedQkv = context->GetInputShape(0);      // [B, L]
    auto aShape   = context->GetInputShape(1);      // [B, HV]
    auto state    = context->GetInputShape(5);      // [N, HV, DV, DK]

    if (mixedQkv == nullptr || aShape == nullptr || state == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto outShape       = context->GetOutputShape(0);
    auto stateOutShape  = context->GetOutputShape(1);
    if (outShape == nullptr || stateOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // out: [B, 1, HV, DV]
    outShape->SetDimNum(4);
    outShape->SetDim(0, mixedQkv->GetDim(0));
    outShape->SetDim(1, 1);
    outShape->SetDim(2, aShape->GetDim(1));
    outShape->SetDim(3, state->GetDim(2));

    // state_out: same as state [N, HV, DV, DK]
    stateOutShape->SetDimNum(state->GetDimNum());
    for (size_t i = 0; i < state->GetDimNum(); ++i) {
        stateOutShape->SetDim(i, state->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedRgdrPackedDecode(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, context->GetInputDataType(5));  // state_out follows state dtype
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedRgdrPackedDecode)
    .InferShape(InferShapeFusedRgdrPackedDecode)
    .InferDataType(InferDataTypeFusedRgdrPackedDecode);

} // namespace ops
