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

ge::graphStatus FusedRgdrPackedDecodeTilingFunc(gert::TilingContext* context)
{
    auto* platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto aicNum = ascendcPlatform.GetCoreNumAic();

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

    uint32_t batchPerCore = (B + totalCores - 1) / totalCores;
    uint32_t dkAligned = ((DK + 7) / 8) * 8;
    uint32_t dvAligned = ((DV + 7) / 8) * 8;

    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }

    auto stateDesc = context->GetInputDesc(5);
    if (stateDesc == nullptr) {
        OP_LOGE(context, "GetInputDesc(5) returns null");
        return ge::GRAPH_FAILED;
    }
    ge::DataType stateDType = stateDesc->GetDataType();
    if (stateDType == ge::DT_FLOAT) {
        context->SetTilingKey(1);  // TilingKey = 1 for float state
    } else if (stateDType == ge::DT_BF16) {
        context->SetTilingKey(2);  // TilingKey = 2 for bf16 state
    } else {
        OP_LOGE(context, "Unsupported state dtype, expected float or bf16");
        return ge::GRAPH_FAILED;
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

    // vStep: tile DV dimension to stay within UB budget.
    // vStep=32 works for float state (UB ~192KB on A3).
    uint32_t vStep = (HV <= 16) ? 32U : 16U;
    if (DV < vStep) vStep = DV;
    tiling->vStep = vStep;

    context->SetBlockDim(ascendcPlatform.CalcTschNumBlocks(aivNum, aicNum, aivNum));
    auto blockDim = context->GetBlockDim();
    if (blockDim == 0) {
        blockDim = static_cast<int64_t>(aivNum);
        context->SetBlockDim(blockDim);
    }

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
