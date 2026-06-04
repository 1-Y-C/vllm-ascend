/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "fused_packed_recurrent_gated_delta_rule_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include <cstring>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(FusedPackedRecurrentGatedDeltaRule,
                              FusedPackedRecurrentGatedDeltaRuleTiling, 0);

static constexpr size_t MQ_IDX  = 0;
static constexpr size_t A_IDX   = 1;
static constexpr size_t B_IDX   = 2;
static constexpr size_t AL_IDX  = 3;
static constexpr size_t DT_IDX  = 4;
static constexpr size_t ST_IDX  = 5;
static constexpr size_t SSI_IDX = 6;
static constexpr size_t D0 = 0, D1 = 1, D2 = 2, D3 = 3;

void FusedPackedRecurrentGatedDeltaRuleTiling::InitCompileInfo()
{
    auto *pi = context_->GetPlatformInfo();
    if (!pi) {
        OP_LOGE(context_->GetNodeName(), "platformInfo null");
        return;
    }
    platform_ascendc::PlatformAscendC ap(pi);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ap.GetCoreNumAiv();
    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.vectorCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::GetPlatformInfo()
{ return ge::GRAPH_SUCCESS; }

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    if (CheckContext() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (AnalyzeDtype() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (AnalyzeShapes() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (GetScaleAttr() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::DoOpTiling()
{
    if (CalUbSize() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::DoLibApiTiling()
{ tilingKey_ = 0; return ge::GRAPH_SUCCESS; }

uint64_t FusedPackedRecurrentGatedDeltaRuleTiling::GetTilingKey() const
{ return tilingKey_; }

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::GetWorkspaceSize()
{ workspaceSize_ = 16 * 1024 * 1024; return ge::GRAPH_SUCCESS; }

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto *raw = context_->GetRawTilingData();
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(),
                           &tilingData_, sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s tiling data failed");
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(tilingData_));
    size_t *ws = context_->GetWorkspaceSizes(1);
    if (!ws) return ge::GRAPH_FAILED;
    ws[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(MQ_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(B_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(AL_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(DT_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ST_IDX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(SSI_IDX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::AnalyzeDtype()
{
    auto mqDt = context_->GetInputDesc(MQ_IDX)->GetDataType();
    auto aDt  = context_->GetInputDesc(A_IDX)->GetDataType();
    auto bDt  = context_->GetInputDesc(B_IDX)->GetDataType();
    auto stDt = context_->GetInputDesc(ST_IDX)->GetDataType();
    if (mqDt != ge::DT_BF16 || aDt != ge::DT_BF16 || bDt != ge::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "mq/a/b must be bf16");
        return ge::GRAPH_FAILED;
    }
    if (stDt != ge::DT_FLOAT && stDt != ge::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "state must be float32 or bf16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    if (RuleCheckShape() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (RuleFillShapeData() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (RuleUpdateBlockDim() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::RuleCheckShape()
{
    auto &mqS  = context_->GetInputShape(MQ_IDX)->GetOriginShape();
    auto &aS   = context_->GetInputShape(A_IDX)->GetOriginShape();
    auto &bS   = context_->GetInputShape(B_IDX)->GetOriginShape();
    auto &alS  = context_->GetInputShape(AL_IDX)->GetOriginShape();
    auto &dtS  = context_->GetInputShape(DT_IDX)->GetOriginShape();
    auto &stS  = context_->GetInputShape(ST_IDX)->GetOriginShape();
    auto &ssiS = context_->GetInputShape(SSI_IDX)->GetOriginShape();

    if (mqS.GetDimNum() != 2 || aS.GetDimNum() != 2 || bS.GetDimNum() != 2 ||
        alS.GetDimNum() != 1 || dtS.GetDimNum() != 1 ||
        stS.GetDimNum() != 4 || ssiS.GetDimNum() != 1) {
        OP_LOGE(context_->GetNodeName(), "dimension count mismatch");
        return ge::GRAPH_FAILED;
    }
    if (mqS.GetDim(D0) != aS.GetDim(D0) || aS.GetDim(D0) != bS.GetDim(D0)) {
        OP_LOGE(context_->GetNodeName(), "batch dim mismatch");
        return ge::GRAPH_FAILED;
    }
    if (alS.GetDim(D0) != aS.GetDim(D1) || dtS.GetDim(D0) != aS.GetDim(D1)) {
        OP_LOGE(context_->GetNodeName(), "HV dim mismatch");
        return ge::GRAPH_FAILED;
    }
    if (stS.GetDim(D1) != aS.GetDim(D1)) {
        OP_LOGE(context_->GetNodeName(), "state HV mismatch");
        return ge::GRAPH_FAILED;
    }
    if (ssiS.GetDim(D0) != mqS.GetDim(D0)) {
        OP_LOGE(context_->GetNodeName(), "ssi batch mismatch");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::RuleFillShapeData()
{
    auto &mqS = context_->GetInputShape(MQ_IDX)->GetOriginShape();
    auto &aS  = context_->GetInputShape(A_IDX)->GetOriginShape();
    auto &stS = context_->GetInputShape(ST_IDX)->GetOriginShape();

    tilingData_.b    = static_cast<uint32_t>(mqS.GetDim(D0));
    tilingData_.qkvDim = static_cast<uint32_t>(mqS.GetDim(D1));
    tilingData_.hv   = static_cast<uint32_t>(aS.GetDim(D1));
    tilingData_.dv   = static_cast<uint32_t>(stS.GetDim(D2));
    tilingData_.dk   = static_cast<uint32_t>(stS.GetDim(D3));
    tilingData_.sBlockNum = static_cast<uint32_t>(stS.GetDim(D0));

    uint32_t hvV = tilingData_.hv * tilingData_.dv;
    uint32_t hk2 = tilingData_.qkvDim - hvV;
    if (hk2 == 0 || (hk2 % 2) != 0) {
        OP_LOGE(context_->GetNodeName(), "bad qkvDim=%u for hv=%u dv=%u",
                tilingData_.qkvDim, tilingData_.hv, tilingData_.dv);
        return ge::GRAPH_FAILED;
    }
    uint32_t hk = hk2 / 2;
    if (hk % tilingData_.dk != 0) {
        OP_LOGE(context_->GetNodeName(), "H*K=%u not divisible by K=%u", hk, tilingData_.dk);
        return ge::GRAPH_FAILED;
    }
    tilingData_.h = hk / tilingData_.dk;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::RuleUpdateBlockDim()
{
    uint64_t tasks = static_cast<uint64_t>(tilingData_.b) * tilingData_.hv;
    if (tasks == 0) tasks = 1;
    uint64_t maxCores = compileInfo_.aivNum > 0 ? compileInfo_.aivNum : 1;
    tilingData_.vectorCoreNum = static_cast<uint32_t>(tasks < maxCores ? tasks : maxCores);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::CalUbSize()
{
    constexpr uint32_t F8A = 8;
    uint32_t aK = ((tilingData_.dk + F8A - 1) / F8A) * F8A;
    uint32_t aV = ((tilingData_.dv + F8A - 1) / F8A) * F8A;
    uint64_t constUB  = 2ULL * tilingData_.hv * 4ULL;
    uint64_t headUB   = static_cast<uint64_t>(aK) * 4ULL * 2ULL
                      + static_cast<uint64_t>(aV) * 4ULL
                      + static_cast<uint64_t>(aV) * static_cast<uint64_t>(aK) * 4ULL * 2ULL
                      + static_cast<uint64_t>(aV) * 4ULL * 2ULL;
    uint64_t totalUB = constUB + headUB + 1024ULL;
    if (totalUB > compileInfo_.ubSize) {
        OP_LOGE(context_->GetNodeName(),
                "UB overflow: need %llu, have %llu",
                static_cast<unsigned long long>(totalUB),
                static_cast<unsigned long long>(compileInfo_.ubSize));
        return ge::GRAPH_FAILED;
    }
    tilingData_.ubRestBytes = static_cast<uint32_t>(compileInfo_.ubSize - totalUB);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTiling::GetScaleAttr()
{
    auto *attrs = context_->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        tilingData_.scale = *attrs->GetAttrPointer<float>(0);
    } else {
        tilingData_.scale = 1.0f;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *ctx)
{
    if (!ctx) return ge::GRAPH_FAILED;
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(ctx);
}

IMPL_OP_OPTILING(FusedPackedRecurrentGatedDeltaRule)
    .Tiling(TilingFunc);

} // namespace optiling
