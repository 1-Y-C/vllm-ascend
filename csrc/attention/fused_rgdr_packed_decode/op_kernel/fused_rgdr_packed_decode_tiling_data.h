/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_RGDR_PACKED_DECODE_TILING_DATA_H_
#define FUSED_RGDR_PACKED_DECODE_TILING_DATA_H_

#include <cstdint>

struct FusedRgdrPackedDecodeTilingData {
    uint32_t B = 0;
    uint32_t HK = 0;
    uint32_t HV = 0;
    uint32_t DK = 0;
    uint32_t DV = 0;
    uint32_t N = 0;
    uint32_t G = 0;
    uint32_t L_qkv = 0;
    uint32_t batchPerCore = 0;
    uint32_t dkAligned = 0;
    uint32_t dvAligned = 0;
    float scaleVal = 0.0f;
    float eps = 1e-6f;
    uint32_t vStep = 16;
};

#endif
