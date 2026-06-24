/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "fused_rgdr_packed_decode_tiling.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_rgdr_packed_decode_tiling_data.h"
#include <cmath>
#include <dlfcn.h>

namespace optiling {

constexpr size_t WORKSPACE_NUM = 1;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// Compute optimal vStep from UB budget.
// coeff_per_dkA_per_vStep = 2*4 (cSt,cStProd always float) + ST_BUF_NUM*sizeof(S)*2 (qStRd,qStWr)
// ST_BUF_NUM = 2, so: float state → 8+16=24, bf16 state → 8+8=16
static uint32_t ComputeVStep(uint32_t ubSize, uint32_t HK, uint32_t HV,
                              uint32_t DV, uint32_t dkA, uint32_t dvA,
                              uint32_t L, uint32_t stateDtypeSize)
{
    constexpr uint32_t SAFETY_MARGIN = 512U;
    constexpr uint32_t VSTEP_ALIGN = 8U;
    constexpr uint32_t VSTEP_MIN = 8U;

    // --- Fixed UB bytes (independent of vStep) ---
    // TQue (single-buffered): qSid_, qMq_, qA_, qB_, qAl_, qDb_, qOut_
    uint32_t fixedBytes = 0;
    fixedBytes += 32U;                                                     // qSid_: 1 slot × 32B
    fixedBytes += L * 2U;                                                  // qMq_: L × sizeof(bf16)=2, 1 slot
    fixedBytes += 3U * ((HV * 2U         < 32U) ? 32U : (HV * 2U));       // qA_, qB_, qOut_
    fixedBytes += 2U * ((HV * 4U         < 32U) ? 32U : (HV * 4U));       // qAl_, qDb_

    // TBuf: all compute buffers except state-dependent cSt_, cStProd_
    fixedBytes += HK * dkA * 4U;                                           // cQ_
    fixedBytes += HK * dkA * 4U;                                           // cK_
    fixedBytes += HV * dvA * 4U;                                           // cV_
    fixedBytes += 512U;                                                    // cRd_ (REDUCE_TMP_BYTES)
    fixedBytes += 256U;                                                    // cGateSg_ (SIGMOID_TMP_BYTES)
    fixedBytes += dvA * 4U;                                                // cOutFp32_
    fixedBytes += dkA * 4U;                                                // cSqBuf_
    fixedBytes += 32U;                                                     // cScBuf_
    fixedBytes += dkA * 4U;                                                // cTmp_
    fixedBytes += 5U * ((HV * 4U < 32U) ? 32U : (HV * 4U));               // cGateAl_,A_,S_,E_, cBeta_

    if (fixedBytes + SAFETY_MARGIN >= ubSize) {
        return VSTEP_MIN;  // not enough UB, use minimum
    }

    // --- vStep-dependent bytes ---
    // cSt_ + cStProd_ (always float in compute buffer)
    uint32_t coeff = 2U * 4U + 2U * 4U;
    // qStRd_ + qStWr_ (double-buffered, ST_BUF_NUM=2)
    coeff += 2U * stateDtypeSize + 2U * stateDtypeSize;

    uint32_t available = ubSize - fixedBytes - SAFETY_MARGIN;
    uint32_t vStepMax = available / (coeff * dkA);

    if (vStepMax > DV) vStepMax = DV;
    vStepMax = (vStepMax / VSTEP_ALIGN) * VSTEP_ALIGN;
    if (vStepMax < VSTEP_MIN) vStepMax = VSTEP_MIN;

    return vStepMax;
}

ge::graphStatus FusedRgdrPackedDecodeTilingFunc(gert::TilingContext* context)
{
    auto* platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto aicNum = ascendcPlatform.GetCoreNumAic();

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    if (aivNum == 0) {
        void* handle = dlopen("libruntime.so", RTLD_LAZY);
        if (handle != nullptr) {
            using RtGetSocFn = void(*)(char*, const uint32_t);
            auto rtGetSocVersion = reinterpret_cast<RtGetSocFn>(dlsym(handle, "rtGetSocVersion"));
            if (rtGetSocVersion != nullptr) {
                char chipName[64] = {0};
                rtGetSocVersion(chipName, sizeof(chipName));
                fe::PlatformInfoManager::Instance().InitRuntimePlatformInfos(chipName);
                fe::PlatFormInfos runtimePi;
                fe::PlatformInfoManager::Instance().GetRuntimePlatformInfosByDevice(0, runtimePi);
                auto runtimePlat = platform_ascendc::PlatformAscendC(&runtimePi);
                aivNum = runtimePlat.GetCoreNumAiv();
                aicNum = runtimePlat.GetCoreNumAic();
            }
            dlclose(handle);
        }
    }
    if (aivNum == 0) {
        OP_LOGE(context, "Cannot determine core count (aivNum=0), Tiling failed.");
        return ge::GRAPH_FAILED;
    }
    uint32_t totalCores = static_cast<uint32_t>(aivNum);

    auto shapeMixedQkv = context->GetInputShape(0)->GetStorageShape();
    auto shapeA = context->GetInputShape(1)->GetStorageShape();
    auto shapeState = context->GetInputShape(5)->GetStorageShape();

    auto storageMixedQkv = EnsureNotScalar(shapeMixedQkv);
    auto storageA = EnsureNotScalar(shapeA);
    auto storageState = EnsureNotScalar(shapeState);

    uint32_t B  = static_cast<uint32_t>(storageMixedQkv.GetDim(0));
    uint32_t L  = static_cast<uint32_t>(storageMixedQkv.GetDim(1));
    uint32_t HV = static_cast<uint32_t>(storageA.GetDim(1));
    uint32_t DV = static_cast<uint32_t>(storageState.GetDim(2));
    uint32_t DK = static_cast<uint32_t>(storageState.GetDim(3));
    uint32_t N  = static_cast<uint32_t>(storageState.GetDim(0));

    uint32_t HK = 0;
    if (DK > 0) {
        HK = (L - HV * DV) / (2 * DK);
    }
    uint32_t G = (HK > 0) ? (HV / HK) : 0;

    uint32_t totalUnits = B * HV;
    uint32_t batchPerCore = (B + totalCores - 1) / totalCores;
    uint32_t dkAligned = ((DK + 7) / 8) * 8;
    uint32_t dvAligned = ((DV + 7) / 8) * 8;

    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }

    auto tiling = context->GetTilingData<FusedRgdrPackedDecodeTilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context, "GetTilingData returns null");
        return ge::GRAPH_FAILED;
    }
    if (memset_s(tiling, sizeof(FusedRgdrPackedDecodeTilingData), 0,
                 sizeof(FusedRgdrPackedDecodeTilingData)) != EOK) {
        OP_LOGE(context, "memset_s tiling data error");
        return ge::GRAPH_FAILED;
    }

    tiling->B = B; tiling->HK = HK; tiling->HV = HV;
    tiling->DK = DK; tiling->DV = DV; tiling->N = N;
    tiling->G = G; tiling->L_qkv = L;
    tiling->totalUnits = totalUnits;
    tiling->batchPerCore = batchPerCore;
    tiling->dkAligned = dkAligned; tiling->dvAligned = dvAligned;

    float scaleVal = (DK > 0) ? (1.0f / sqrtf(static_cast<float>(DK))) : 1.0f;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        const float* scalePtr = attrs->GetFloat(0);
        if (scalePtr != nullptr) {
            scaleVal = *scalePtr;
        }
    }
    tiling->scaleVal = scaleVal;
    tiling->eps = 1e-6f;

    // --- vStep: dynamically computed from UB budget ---
    auto stateDesc = context->GetInputDesc(5);
    if (stateDesc == nullptr) {
        OP_LOGE(context, "GetInputDesc(5) returns null");
        return ge::GRAPH_FAILED;
    }
    ge::DataType stateDType = stateDesc->GetDataType();
    size_t stateDtypeSize = (stateDType == ge::DT_FLOAT) ? 4 : 2;

    uint32_t vStep = ComputeVStep(static_cast<uint32_t>(ubSize), HK, HV,
                                    DV, dkAligned, dvAligned, L,
                                    static_cast<uint32_t>(stateDtypeSize));
    tiling->vStep = vStep;

    if (stateDType == ge::DT_FLOAT) {
        context->SetTilingKey(1);
    } else if (stateDType == ge::DT_BF16) {
        context->SetTilingKey(2);
    } else {
        OP_LOGE(context, "Unsupported state dtype, expected float or bf16");
        return ge::GRAPH_FAILED;
    }

    // Dynamic BlockDim: don't launch more cores than work units (RGDR pattern)
    uint64_t taskUnits = static_cast<uint64_t>(totalUnits);
    uint64_t maxCoreNum = (aivNum > 0) ? static_cast<uint64_t>(aivNum) : 0;
    uint64_t selectedCoreNum = (taskUnits < maxCoreNum) ? taskUnits : maxCoreNum;
    uint32_t blockDimVal = static_cast<uint32_t>(selectedCoreNum);
    if (blockDimVal == 0) {
        blockDimVal = 1;
    }
    context->SetBlockDim(static_cast<int64_t>(blockDimVal));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedRgdrPackedDecode(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedRgdrPackedDecode)
    .Tiling(FusedRgdrPackedDecodeTilingFunc)
    .TilingParse<FusedRgdrPackedDecodeCompileInfo>(TilingPrepareForFusedRgdrPackedDecode);

} // namespace optiling
