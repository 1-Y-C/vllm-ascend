/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "fused_packed_recurrent_gated_delta_rule_tiling.h"

#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/fused_packed_recurrent_gated_delta_rule_tiling_data.h"

using namespace FusedPackedRecurrentGatedDeltaRule;

namespace optiling {

namespace {

constexpr size_t MIXED_QKV_INDEX = 0;
constexpr size_t A_INDEX = 1;
constexpr size_t STATE_INDEX = 5;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

constexpr int64_t FP32_SIZE = 4;
constexpr int64_t BF16_SIZE = 2;
constexpr int64_t FP32_PER_BLOCK = 8;

struct UbCalcContext {
    int64_t ubSize = 0;
    int64_t hk = 0;
    int64_t hv = 0;
    int64_t dv = 0;
    int64_t dk = 0;
    int64_t fixedUbBytes = 0;
    int64_t workingUbBytes = 0;
    int64_t coeff = 0;
};

struct BufferProfile {
    BufferProfile() = default;
    BufferProfile(uint32_t s, uint32_t a, uint32_t v, uint32_t r, bool val)
        : stateOutBufferNum(s), attnOutBufferNum(a), vStep(v), repeatTime(r), valid(val) {}
    uint32_t stateOutBufferNum = 1;
    uint32_t attnOutBufferNum = 1;
    uint32_t vStep = 0;
    uint32_t repeatTime = 0;
    bool valid = false;
};

int64_t CalcFixedUbBytes(int64_t hv, int64_t dv, int64_t dk, int64_t hk)
{
    int64_t queueBytes = hk * dk * BF16_SIZE + hk * dk * BF16_SIZE + dv * BF16_SIZE;
    int64_t qkUbBytes = hk * dk * FP32_SIZE * 2 + dv * FP32_SIZE;
    int64_t hvAligned = ((hv + FP32_PER_BLOCK - 1) / FP32_PER_BLOCK) * FP32_PER_BLOCK;
    int64_t gatingBytes = hvAligned * FP32_SIZE * 2;
    int64_t tmpBytes = (hk * dk + 2 * hvAligned) * FP32_SIZE;
    return queueBytes + qkUbBytes + gatingBytes + tmpBytes;
}

int64_t CalcWorkingUbBytes(int64_t hv, int64_t dv, int64_t dk, int64_t hk)
{
    (void)hv; (void)dv; (void)hk;
    int64_t vStepBytes = dk * FP32_SIZE
                       + dk * FP32_SIZE
                       + dk * FP32_SIZE
                       + FP32_SIZE
                       + FP32_SIZE;
    return vStepBytes;
}

int64_t CalcVStepCoeff(int64_t dk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum)
{
    int64_t coeff = dk * FP32_SIZE
                   + dk * FP32_SIZE * stateOutBufferNum
                   + static_cast<int64_t>(attnOutBufferNum) * 2
                   + dk * FP32_SIZE
                   + dk * FP32_SIZE
                   + FP32_SIZE
                   + FP32_SIZE;
    return coeff;
}

bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes,
    int64_t dk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum,
    BufferProfile &profile, int64_t dv)
{
    int64_t remainBytes = ubSize - usedUbBytes;
    if (remainBytes <= 0) { profile.valid = false; return false; }
    int64_t coeff = CalcVStepCoeff(dk, stateOutBufferNum, attnOutBufferNum);
    if (coeff <= 0) { profile.valid = false; return false; }
    int64_t maxVStep = remainBytes / coeff;
    if (maxVStep <= 0) { profile.valid = false; return false; }
    maxVStep = (maxVStep / FP32_PER_BLOCK) * FP32_PER_BLOCK;
    if (maxVStep <= 0) { profile.valid = false; return false; }
    profile.vStep = static_cast<uint32_t>(maxVStep < dv ? maxVStep : dv);
    profile.stateOutBufferNum = stateOutBufferNum;
    profile.attnOutBufferNum = attnOutBufferNum;
    profile.repeatTime = static_cast<uint32_t>(dk / maxVStep);
    if (profile.repeatTime < 1) profile.repeatTime = 1;
    profile.valid = true;
    return true;
}

bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current)
{
    if (!current.valid) return true;
    if (candidate.repeatTime < current.repeatTime) return true;
    if (candidate.vStep > current.vStep) return true;
    return false;
}

} // namespace

ge::graphStatus FusedPackedRecurrentGatedDeltaRuleTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }

    auto mixedQkvShapePtr = context->GetInputShape(MIXED_QKV_INDEX);
    auto aShapePtr = context->GetInputShape(A_INDEX);
    auto stateShapePtr = context->GetInputShape(STATE_INDEX);
    if (mixedQkvShapePtr == nullptr || aShapePtr == nullptr || stateShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto mixedQkvShape = mixedQkvShapePtr->GetOriginShape();
    auto aShape = aShapePtr->GetOriginShape();
    auto stateShape = stateShapePtr->GetOriginShape();

    FusedPackedRecurrentGatedDeltaRuleTilingData td{};
    td.b = static_cast<uint32_t>(mixedQkvShape.GetDim(DIM_0));
    auto mixedQkvLastDim = mixedQkvShape.GetDim(DIM_1);
    td.hv = static_cast<uint32_t>(aShape.GetDim(DIM_1));
    td.dv = static_cast<uint32_t>(stateShape.GetDim(DIM_2));
    td.dk = static_cast<uint32_t>(stateShape.GetDim(DIM_3));
    td.sBlockNum = static_cast<uint32_t>(stateShape.GetDim(DIM_0));

    if (td.dk == 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t qkDim = 2 * static_cast<int64_t>(td.dk);
    int64_t expectedHeadDim = static_cast<int64_t>(td.hv) * static_cast<int64_t>(td.dv);
    if (mixedQkvLastDim < expectedHeadDim) {
        return ge::GRAPH_FAILED;
    }
    td.hk = static_cast<uint32_t>((mixedQkvLastDim - expectedHeadDim) / qkDim);

    // Scale attribute
    float userScale = 0.0f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *scaleAttr = attrs->GetAttrPointer<float>(0);
        if (scaleAttr != nullptr) {
            userScale = *scaleAttr;
        }
    }
    if (userScale > 0.0f) {
        td.scale = userScale;
    } else {
        td.scale = 1.0f / sqrtf(static_cast<float>(td.dk));
    }

    // UB calculation for vStep optimization
    UbCalcContext ubCtx;
    ubCtx.ubSize = static_cast<int64_t>(ubSize);
    ubCtx.hk = static_cast<int64_t>(td.hk);
    ubCtx.hv = static_cast<int64_t>(td.hv);
    ubCtx.dv = static_cast<int64_t>(td.dv);
    ubCtx.dk = static_cast<int64_t>(td.dk);
    ubCtx.fixedUbBytes = CalcFixedUbBytes(ubCtx.hv, ubCtx.dv, ubCtx.dk, ubCtx.hk);
    ubCtx.workingUbBytes = CalcWorkingUbBytes(ubCtx.hv, ubCtx.dv, ubCtx.dk, ubCtx.hk);
    ubCtx.coeff = CalcVStepCoeff(ubCtx.dk, 1, 1);

    // Finalize vStep from UB
    BufferProfile bestProfile;
    for (uint32_t sb = 1; sb <= 2; sb++) {
        for (uint32_t ab = 1; ab <= 2; ab++) {
            BufferProfile profile;
            int64_t stepCoeff = CalcVStepCoeff(ubCtx.dk, sb, ab);
            int64_t usedBytes = ubCtx.fixedUbBytes + stepCoeff;
            EvaluateBufferProfile(ubCtx.ubSize, usedBytes, ubCtx.dk, sb, ab, profile, ubCtx.dv);
            if (profile.valid && IsBetterProfile(profile, bestProfile)) {
                bestProfile = profile;
                bestProfile.stateOutBufferNum = sb;
                bestProfile.attnOutBufferNum = ab;
            }
        }
    }

    if (!bestProfile.valid) {
        td.vStep = 16;
        td.stateOutBufferNum = 1;
        td.attnOutBufferNum = 1;
    } else {
        td.vStep = bestProfile.vStep;
        td.stateOutBufferNum = bestProfile.stateOutBufferNum;
        td.attnOutBufferNum = bestProfile.attnOutBufferNum;
    }

    td.vectorCoreNum = aivNum;

    // Block dim: use batch count, capped at aivNum
    uint32_t blockDim = td.b;
    if (blockDim > aivNum) {
        blockDim = aivNum;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }
    context->SetBlockDim(blockDim);

    constexpr uint64_t TILING_KEY = 0;
    context->SetTilingKey(TILING_KEY);

    const size_t tilingSize = sizeof(FusedPackedRecurrentGatedDeltaRuleTilingData);
    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetCapacity() < tilingSize) {
        return ge::GRAPH_FAILED;
    }
    errno_t rc = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                          &td, tilingSize);
    if (rc != EOK) {
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingSize);

    constexpr int64_t sysWorkspaceSize = 16777216;
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = static_cast<size_t>(sysWorkspaceSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedPackedRecurrentGatedDeltaRule(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

IMPL_OP_OPTILING(FusedPackedRecurrentGatedDeltaRule)
    .Tiling(optiling::FusedPackedRecurrentGatedDeltaRuleTilingFunc)
    .TilingParse<optiling::FusedPackedRecurrentGatedDeltaRuleCompileInfo>(optiling::TilingPrepareForFusedPackedRecurrentGatedDeltaRule);
