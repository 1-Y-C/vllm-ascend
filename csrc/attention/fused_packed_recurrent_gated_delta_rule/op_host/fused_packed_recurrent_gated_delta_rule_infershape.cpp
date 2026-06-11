/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;
namespace ops {

const size_t MIXED_QKV_INDEX = 0;
const size_t A_INDEX = 1;
const size_t STATE_INDEX = 5;
const size_t OUTPUT_INDEX = 0;
const size_t STATE_OUT_INDEX = 1;

// Output: [B, 1, HV, V]
const size_t OUTPUT_DIM = 4;
// State: [N, HV, V, K]
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeFusedPackedRecurrentGatedDeltaRule(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedPackedRecurrentGatedDeltaRule", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeA = context->GetInputShape(A_INDEX);
    auto shapeState = context->GetInputShape(STATE_INDEX);
    auto shapeOut = context->GetOutputShape(OUTPUT_INDEX);
    auto shapeFinalState = context->GetOutputShape(STATE_OUT_INDEX);
    if (shapeA == nullptr || shapeState == nullptr ||
        shapeOut == nullptr || shapeFinalState == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    // out: [B, 1, HV, V] — B from a, HV from a, V from state
    shapeOut->SetDimNum(OUTPUT_DIM);
    int64_t B  = shapeA->GetDim(DIM_0);
    int64_t HV = shapeA->GetDim(DIM_1);
    int64_t V  = shapeState->GetDim(DIM_2);
    shapeOut->SetDim(0, B);
    shapeOut->SetDim(1, 1);
    shapeOut->SetDim(2, HV);
    shapeOut->SetDim(3, V);

    // finalState: [N, HV, V, K] — same as input state
    shapeFinalState->SetDimNum(STATE_DIM);
    shapeFinalState->SetDim(DIM_0, shapeState->GetDim(DIM_0));
    shapeFinalState->SetDim(DIM_1, shapeState->GetDim(DIM_1));
    shapeFinalState->SetDim(DIM_2, shapeState->GetDim(DIM_2));
    shapeFinalState->SetDim(DIM_3, shapeState->GetDim(DIM_3));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedPackedRecurrentGatedDeltaRule(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedPackedRecurrentGatedDeltaRule)
    .InferShape(InferShapeFusedPackedRecurrentGatedDeltaRule)
    .InferDataType(InferDataTypeFusedPackedRecurrentGatedDeltaRule);
} // namespace ops
