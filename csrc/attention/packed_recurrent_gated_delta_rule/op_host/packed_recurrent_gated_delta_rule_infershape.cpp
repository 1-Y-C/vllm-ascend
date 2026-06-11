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
const size_t ATTNOUT_INDEX = 0;
const size_t STATE_OUT_INDEX = 1;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

const size_t ATTN_OUT_DIM_NUM = 4;

static ge::graphStatus InferShapePackedRecurrentGatedDeltaRule(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("PackedRecurrentGatedDeltaRule", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeMixedQkv = context->GetInputShape(MIXED_QKV_INDEX);
    auto shapeA = context->GetInputShape(A_INDEX);
    auto shapeInitialState = context->GetInputShape(STATE_INDEX);
    auto shapeAttnOut = context->GetOutputShape(ATTNOUT_INDEX);
    auto shapeFinalState = context->GetOutputShape(STATE_OUT_INDEX);

    if (shapeMixedQkv == nullptr || shapeA == nullptr || shapeInitialState == nullptr ||
        shapeAttnOut == nullptr || shapeFinalState == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    int64_t B = shapeMixedQkv->GetDim(DIM_0);
    int64_t HV = shapeA->GetDim(DIM_1);
    int64_t DV = shapeInitialState->GetDim(DIM_2);

    // attn_out: (B, 1, HV, DV)
    shapeAttnOut->SetDimNum(ATTN_OUT_DIM_NUM);
    shapeAttnOut->SetDim(DIM_0, B);
    shapeAttnOut->SetDim(DIM_1, 1);
    shapeAttnOut->SetDim(DIM_2, HV);
    shapeAttnOut->SetDim(DIM_3, DV);

    // final state: same as initial state (in-place)
    shapeFinalState->SetDimNum(shapeInitialState->GetDimNum());
    for (size_t i = 0; i < shapeInitialState->GetDimNum(); i++) {
        shapeFinalState->SetDim(i, shapeInitialState->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePackedRecurrentGatedDeltaRule(gert::InferDataTypeContext *context)
{
    auto stateDtype = context->GetInputDataType(STATE_INDEX);
    context->SetOutputDataType(ATTNOUT_INDEX, stateDtype);
    context->SetOutputDataType(STATE_OUT_INDEX, stateDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PackedRecurrentGatedDeltaRule)
    .InferShape(InferShapePackedRecurrentGatedDeltaRule)
    .InferDataType(InferDataTypePackedRecurrentGatedDeltaRule);
} // namespace ops
