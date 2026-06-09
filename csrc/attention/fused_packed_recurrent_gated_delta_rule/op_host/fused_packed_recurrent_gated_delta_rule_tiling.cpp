/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "fused_packed_recurrent_gated_delta_rule_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "log/ops_log.h"
#include <cstring>

using namespace std;

namespace optiling {

static constexpr size_t MQ_IDX  = 0;
static constexpr size_t A_IDX   = 1;
static constexpr size_t B_IDX   = 2;
static constexpr size_t AL_IDX  = 3;
static constexpr size_t DT_IDX  = 4;
static constexpr size_t ST_IDX  = 5;
static constexpr size_t SSI_IDX = 6;
static constexpr size_t D0 = 0, D1 = 1, D2 = 2, D3 = 3;
static constexpr uint32_t F8A = 8;
static constexpr uint32_t UB_RESERVED = 1024;

static ge::graphStatus Tiling4FusedPackedRecurrentGatedDeltaRule(gert::TilingContext *context)
{
    if (!context) return ge::GRAPH_FAILED;

    auto *pi = context->GetPlatformInfo();
    if (!pi) {
        OP_LOGE(context->GetNodeName(), "platformInfo null");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ap(pi);
    uint64_t ubSize = 0;
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t aivNum = ap.GetCoreNumAiv();
    if (aivNum <= 0) {
        OP_LOGE(context->GetNodeName(), "aivNum <= 0");
        return ge::GRAPH_FAILED;
    }

    // Dtype checks
    auto mqDt = context->GetInputDesc(MQ_IDX)->GetDataType();
    auto aDt  = context->GetInputDesc(A_IDX)->GetDataType();
    auto bDt  = context->GetInputDesc(B_IDX)->GetDataType();
    auto stDt = context->GetInputDesc(ST_IDX)->GetDataType();
    if (mqDt != ge::DT_BF16 || aDt != ge::DT_BF16 || bDt != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "mq/a/b must be bf16");
        return ge::GRAPH_FAILED;
    }
    if (stDt != ge::DT_FLOAT && stDt != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "state must be float32 or bf16");
        return ge::GRAPH_FAILED;
    }

    // Shape checks
    auto &mqS  = context->GetInputShape(MQ_IDX)->GetOriginShape();
    auto &aS   = context->GetInputShape(A_IDX)->GetOriginShape();
    auto &bS   = context->GetInputShape(B_IDX)->GetOriginShape();
    auto &alS  = context->GetInputShape(AL_IDX)->GetOriginShape();
    auto &dtS  = context->GetInputShape(DT_IDX)->GetOriginShape();
    auto &stS  = context->GetInputShape(ST_IDX)->GetOriginShape();
    auto &ssiS = context->GetInputShape(SSI_IDX)->GetOriginShape();

    if (mqS.GetDimNum() != 2 || aS.GetDimNum() != 2 || bS.GetDimNum() != 2 ||
        alS.GetDimNum() != 1 || dtS.GetDimNum() != 1 ||
        stS.GetDimNum() != 4 || ssiS.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "dimension count mismatch");
        return ge::GRAPH_FAILED;
    }
    if (mqS.GetDim(D0) != aS.GetDim(D0) || aS.GetDim(D0) != bS.GetDim(D0)) {
        OP_LOGE(context->GetNodeName(), "batch dim mismatch");
        return ge::GRAPH_FAILED;
    }
    if (alS.GetDim(D0) != aS.GetDim(D1) || dtS.GetDim(D0) != aS.GetDim(D1)) {
        OP_LOGE(context->GetNodeName(), "HV dim mismatch");
        return ge::GRAPH_FAILED;
    }
    if (stS.GetDim(D1) != aS.GetDim(D1)) {
        OP_LOGE(context->GetNodeName(), "state HV mismatch");
        return ge::GRAPH_FAILED;
    }
    if (ssiS.GetDim(D0) != mqS.GetDim(D0)) {
        OP_LOGE(context->GetNodeName(), "ssi batch mismatch");
        return ge::GRAPH_FAILED;
    }

    // Fill tiling data
    FusedPackedRecurrentGatedDeltaRuleTilingData td;
    td.set_b(static_cast<uint32_t>(mqS.GetDim(D0)));
    td.set_qkvDim(static_cast<uint32_t>(mqS.GetDim(D1)));
    td.set_hv(static_cast<uint32_t>(aS.GetDim(D1)));
    td.set_dv(static_cast<uint32_t>(stS.GetDim(D2)));
    td.set_dk(static_cast<uint32_t>(stS.GetDim(D3)));
    td.set_sBlockNum(static_cast<uint32_t>(stS.GetDim(D0)));

    uint32_t hvV = td.get_hv() * td.get_dv();
    uint32_t hk2 = td.get_qkvDim() - hvV;
    if (hk2 == 0 || (hk2 % 2) != 0) {
        OP_LOGE(context->GetNodeName(), "bad qkvDim=%u for hv=%u dv=%u",
                td.get_qkvDim(), td.get_hv(), td.get_dv());
        return ge::GRAPH_FAILED;
    }
    uint32_t hk = hk2 / 2;
    if (hk % td.get_dk() != 0) {
        OP_LOGE(context->GetNodeName(), "H*K=%u not divisible by K=%u", hk, td.get_dk());
        return ge::GRAPH_FAILED;
    }
    td.set_h(hk / td.get_dk());

    // Block dim
    uint64_t tasks = static_cast<uint64_t>(td.get_b()) * td.get_hv();
    if (tasks == 0) tasks = 1;
    uint64_t maxCores = aivNum > 0 ? aivNum : 1;
    td.set_vectorCoreNum(static_cast<uint32_t>(tasks < maxCores ? tasks : maxCores));

    // Scale attr
    auto *attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        td.set_scale(*attrs->GetAttrPointer<float>(0));
    } else {
        td.set_scale(1.0f);
    }

    // UB check
    uint32_t aK = ((td.get_dk() + F8A - 1) / F8A) * F8A;
    uint32_t aV = ((td.get_dv() + F8A - 1) / F8A) * F8A;
    uint32_t hvAligned = ((td.get_hv() + F8A - 1) / F8A) * F8A;
    uint64_t constUB  = 2ULL * hvAligned * 4ULL;
    uint64_t headUB   = static_cast<uint64_t>(aK) * 4ULL * 2ULL
                      + static_cast<uint64_t>(aV) * 4ULL
                      + static_cast<uint64_t>(aV) * static_cast<uint64_t>(aK) * 4ULL * 2ULL
                      + static_cast<uint64_t>(aV) * 4ULL * 2ULL;
    uint64_t totalUB = constUB + headUB + UB_RESERVED;
    if (totalUB > ubSize) {
        OP_LOGE(context->GetNodeName(),
                "UB overflow: need %llu, have %llu",
                static_cast<unsigned long long>(totalUB),
                static_cast<unsigned long long>(ubSize));
        return ge::GRAPH_FAILED;
    }
    td.set_ubRestBytes(static_cast<uint32_t>(ubSize - totalUB));

    context->SetBlockDim(td.get_vectorCoreNum());
    auto *raw = context->GetRawTilingData();
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(), &td, sizeof(td));
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s tiling data failed");
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(sizeof(td));

    size_t *ws = context->GetWorkspaceSizes(1);
    if (!ws) return ge::GRAPH_FAILED;
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4FusedPackedRecurrentGatedDeltaRule(
    gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<FusedPackedRecurrentGatedDeltaRuleCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedPackedRecurrentGatedDeltaRule)
    .Tiling(Tiling4FusedPackedRecurrentGatedDeltaRule)
    .TilingParse<FusedPackedRecurrentGatedDeltaRuleCompileInfo>(
        TilingPrepare4FusedPackedRecurrentGatedDeltaRule);

} // namespace optiling
