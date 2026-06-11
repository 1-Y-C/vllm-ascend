#include "packed_recurrent_gated_delta_rule_tiling.h"

#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "error/ops_error.h"

namespace optiling {

const size_t MIXED_QKV_INDEX = 0;
const size_t A_INDEX = 1;
const size_t SSM_STATE_INDICES_INDEX = 6;
const size_t STATE_INDEX = 5;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;
const size_t DIM_4 = 4;

constexpr int64_t FP32_SIZE = 4;
constexpr int64_t BF16_SIZE = 2;
constexpr int64_t FP16_SIZE = 2;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BF16_PER_BLOCK = 16;
constexpr int64_t FP32_PER_BLOCK = 8;
constexpr int64_t REPEAT_MASK = 64;

void PackedRecurrentGatedDeltaRuleTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
    }
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null when getting platform info");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        float attrScale = *attrs->GetFloat(0);
        if (attrScale > 0.0f) {
            inputParams_.userScale = attrScale;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::DoOpTiling()
{
    auto ret = RuleInitUbCalcContext();
    if (ret != ge::GRAPH_SUCCESS) return ret;
    ret = RuleCalcFixedUbBytes();
    if (ret != ge::GRAPH_SUCCESS) return ret;
    ret = RuleCalcWorkingUbBytes();
    if (ret != ge::GRAPH_SUCCESS) return ret;
    ret = RuleCalcVStepCoeff();
    if (ret != ge::GRAPH_SUCCESS) return ret;
    ret = RuleFinalizeVStepFromUb();
    if (ret != ge::GRAPH_SUCCESS) return ret;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t PackedRecurrentGatedDeltaRuleTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::GetWorkspaceSize()
{
    constexpr int64_t sysWorkspaceSize = 16777216;
    workspaceSize_ = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(PackedRecurrentGatedDeltaRule::PackedRecurrentGatedDeltaRuleTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void PackedRecurrentGatedDeltaRuleTiling::PrintTilingData()
{
    OP_LOGI(inputParams_.opName,
            "B:%u HK:%u DK:%u HV:%u DV:%u sBlockNum:%u vStep:%u bufNum[%u/%u] scale:%f",
            tilingData_.b, tilingData_.hk, tilingData_.dk, tilingData_.hv, tilingData_.dv,
            tilingData_.sBlockNum, tilingData_.vStep, tilingData_.stateOutBufferNum,
            tilingData_.attnOutBufferNum, tilingData_.scale);
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::CheckContext()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::AnalyzeDtype()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::AnalyzeShapes()
{
    return FillTilingShapeData();
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::FillTilingShapeData()
{
    auto mixedQkvShapePtr = context_->GetInputShape(MIXED_QKV_INDEX);
    auto aShapePtr = context_->GetInputShape(A_INDEX);
    auto stateShapePtr = context_->GetInputShape(STATE_INDEX);
    if (mixedQkvShapePtr == nullptr || aShapePtr == nullptr || stateShapePtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "GetInputShape failed");
        return ge::GRAPH_FAILED;
    }

    auto mixedQkvShape = mixedQkvShapePtr->GetOriginShape();
    auto aShape = aShapePtr->GetOriginShape();
    auto stateShape = stateShapePtr->GetOriginShape();

    tilingData_.b = static_cast<uint32_t>(mixedQkvShape.GetDim(DIM_0));
    auto mixedQkvLastDim = mixedQkvShape.GetDim(DIM_1);
    tilingData_.hv = static_cast<uint32_t>(aShape.GetDim(DIM_1));
    tilingData_.dv = static_cast<uint32_t>(stateShape.GetDim(DIM_2));
    tilingData_.dk = static_cast<uint32_t>(stateShape.GetDim(DIM_3));
    tilingData_.sBlockNum = static_cast<uint32_t>(stateShape.GetDim(DIM_0));

    if (tilingData_.dk == 0) {
        OP_LOGE(context_->GetNodeName(), "dk is zero");
        return ge::GRAPH_FAILED;
    }

    int64_t qkDim = 2 * static_cast<int64_t>(tilingData_.dk);
    int64_t expectedHeadDim = static_cast<int64_t>(tilingData_.hv) * static_cast<int64_t>(tilingData_.dv);
    if (mixedQkvLastDim < expectedHeadDim) {
        OP_LOGE(context_->GetNodeName(), "mixedQkv shape mismatch: lastDim=%ld < hv*dv=%ld", mixedQkvLastDim, expectedHeadDim);
        return ge::GRAPH_FAILED;
    }
    tilingData_.hk = static_cast<uint32_t>((mixedQkvLastDim - expectedHeadDim) / qkDim);

    if (inputParams_.userScale > 0.0f) {
        tilingData_.scale = inputParams_.userScale;
    } else {
        tilingData_.scale = 1.0f / sqrtf(static_cast<float>(tilingData_.dk));
    }

    tilingData_.stateOutBufferNum = 1;
    tilingData_.attnOutBufferNum = 1;
    tilingData_.vectorCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::CalUbSize()
{
    return ge::GRAPH_SUCCESS;
}

int64_t PackedRecurrentGatedDeltaRuleTiling::CalcFixedUbBytes(int64_t hv, int64_t dv, int64_t dk, int64_t hk) const
{
    // qInQueue + kInQueue (HK heads) + vInQueue (single head)
    int64_t queueBytes = hk * dk * BF16_SIZE + hk * dk * BF16_SIZE + dv * BF16_SIZE;
    // qInUb + kInUb + vInUb (fp32, HK heads for q/k)
    int64_t qkUbBytes = hk * dk * FP32_SIZE * 2 + dv * FP32_SIZE;
    // Gating buffers: g + beta (fp32)
    int64_t gatingBytes = hv * FP32_SIZE * 2;
    // tmpUb for L2Norm (HK heads)
    int64_t tmpBytes = hk * dk * FP32_SIZE;
    return queueBytes + qkUbBytes + gatingBytes + tmpBytes;
}

int64_t PackedRecurrentGatedDeltaRuleTiling::CalcWorkingUbBytes(int64_t hv, int64_t dv, int64_t dk, int64_t hk) const
{
    (void)hv;
    (void)dv;
    (void)hk;
    // stateInQueue + stateInUb + broadTmpUb + deltaUb + attnUb (per vStep)
    int64_t vStepBytes = dk * BF16_SIZE    // stateInQueue
                       + dk * FP32_SIZE    // stateInUb
                       + dk * FP32_SIZE    // broadTmpUb
                       + FP32_SIZE         // deltaUb
                       + FP32_SIZE;        // attnUb
    return vStepBytes;
}

int64_t PackedRecurrentGatedDeltaRuleTiling::CalcVStepCoeff(int64_t dk,
    uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) const
{
    int64_t coeff = dk * BF16_SIZE                       // stateInQueue
                   + dk * BF16_SIZE * stateOutBufferNum  // stateOutQueue per vStep
                   + static_cast<int64_t>(attnOutBufferNum) * 2 // attnOutQueue per vStep (bf16/fp16)
                   + dk * FP32_SIZE                      // stateInUb
                   + dk * FP32_SIZE                      // broadTmpUb
                   + FP32_SIZE                           // deltaUb
                   + FP32_SIZE;                          // attnUb
    return coeff;
}

bool PackedRecurrentGatedDeltaRuleTiling::EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes,
    int64_t dk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum, BufferProfile &profile) const
{
    (void)attnOutBufferNum;
    int64_t remainBytes = ubSize - usedUbBytes;
    if (remainBytes <= 0) {
        profile.valid = false;
        return false;
    }
    int64_t coeff = CalcVStepCoeff(dk, stateOutBufferNum, attnOutBufferNum);
    if (coeff <= 0) {
        profile.valid = false;
        return false;
    }
    int64_t maxVStep = remainBytes / coeff;
    if (maxVStep <= 0) {
        profile.valid = false;
        return false;
    }
    // Align to 32B boundary
    maxVStep = (maxVStep / FP32_PER_BLOCK) * FP32_PER_BLOCK;
    if (maxVStep <= 0) {
        profile.valid = false;
        return false;
    }
    // Use max possible vStep (at most dv)
    int64_t dv = static_cast<int64_t>(tilingData_.dv);
    profile.vStep = static_cast<uint32_t>(maxVStep < dv ? maxVStep : dv);
    profile.stateOutBufferNum = stateOutBufferNum;
    profile.attnOutBufferNum = attnOutBufferNum;
    profile.repeatTime = static_cast<uint32_t>(128 / maxVStep);
    if (profile.repeatTime < 1) profile.repeatTime = 1;
    profile.valid = true;
    return true;
}

bool PackedRecurrentGatedDeltaRuleTiling::IsBetterProfile(const BufferProfile &candidate,
                                                           const BufferProfile &current) const
{
    if (!current.valid) return true;
    if (candidate.repeatTime < current.repeatTime) return true;
    if (candidate.vStep > current.vStep) return true;
    return false;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::FinalizeVStepFromUb()
{
    auto fixedBytes = ubCalcCtx_.fixedUbBytes;
    auto workingBytes = ubCalcCtx_.workingUbBytes;
    auto totalUb = ubCalcCtx_.ubSize;

    // Evaluate buffer profiles for different buffer configurations
    BufferProfile bestProfile;
    BufferProfile profile;
    for (uint32_t sb = 1; sb <= 2; sb++) {
        for (uint32_t ab = 1; ab <= 2; ab++) {
            int64_t stepCoeff = CalcVStepCoeff(ubCalcCtx_.dk, sb, ab);
            int64_t usedBytes = fixedBytes + workingBytes * (1); // single vStep estimate
            EvaluateBufferProfile(totalUb, usedBytes, ubCalcCtx_.dk, sb, ab, profile);
            if (profile.valid && IsBetterProfile(profile, bestProfile)) {
                bestProfile = profile;
                bestProfile.stateOutBufferNum = sb;
                bestProfile.attnOutBufferNum = ab;
            }
        }
    }

    if (!bestProfile.valid) {
        // Fallback: minimal vStep
        tilingData_.vStep = 1;
        tilingData_.stateOutBufferNum = 1;
        tilingData_.attnOutBufferNum = 1;
    } else {
        tilingData_.vStep = bestProfile.vStep;
        tilingData_.stateOutBufferNum = bestProfile.stateOutBufferNum;
        tilingData_.attnOutBufferNum = bestProfile.attnOutBufferNum;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::RuleInitUbCalcContext()
{
    AnalyzeShapes();
    ubCalcCtx_.ubSize = static_cast<int64_t>(compileInfo_.ubSize);
    ubCalcCtx_.hk = static_cast<int64_t>(tilingData_.hk);
    ubCalcCtx_.hv = static_cast<int64_t>(tilingData_.hv);
    ubCalcCtx_.dv = static_cast<int64_t>(tilingData_.dv);
    ubCalcCtx_.dk = static_cast<int64_t>(tilingData_.dk);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::RuleCalcFixedUbBytes()
{
    ubCalcCtx_.fixedUbBytes = CalcFixedUbBytes(ubCalcCtx_.hv, ubCalcCtx_.dv, ubCalcCtx_.dk, ubCalcCtx_.hk);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::RuleCalcWorkingUbBytes()
{
    ubCalcCtx_.workingUbBytes = CalcWorkingUbBytes(ubCalcCtx_.hv, ubCalcCtx_.dv, ubCalcCtx_.dk, ubCalcCtx_.hk);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::RuleCalcVStepCoeff()
{
    ubCalcCtx_.coeff = CalcVStepCoeff(ubCalcCtx_.dk, 1, 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PackedRecurrentGatedDeltaRuleTiling::RuleFinalizeVStepFromUb()
{
    return FinalizeVStepFromUb();
}

ge::graphStatus TilingForPackedRecurrentGatedDeltaRule(gert::TilingContext *context)
{
    optiling::PackedRecurrentGatedDeltaRuleTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingParseForPackedRecurrentGatedDeltaRule(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PackedRecurrentGatedDeltaRule)
    .Tiling(TilingForPackedRecurrentGatedDeltaRule)
    .TilingParse<PackedRecurrentGatedDeltaRuleCompileInfo>(TilingParseForPackedRecurrentGatedDeltaRule);

} // namespace optiling
