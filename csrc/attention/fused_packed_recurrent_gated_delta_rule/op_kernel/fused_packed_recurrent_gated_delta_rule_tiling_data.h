/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_packed_recurrent_gated_delta_rule_tiling_data.h
 * \brief Tiling data shared between host tiling and device kernel.
 *
 * This kernel fuses L2Norm, sigmoid gating, and delta rule recurrence
 * for the decode path, operating directly on packed mixed_qkv layout.
 */

#ifndef FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
#define FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace FusedPackedRecurrentGatedDeltaRule {

#pragma pack(push, 8)
struct alignas(8) FusedPackedRecurrentGatedDeltaRuleTilingData {
    uint32_t vectorCoreNum;   // number of AIV cores to launch
    uint32_t b;               // batch size (num tokens in decode batch)
    uint32_t h;               // num key heads (H)
    uint32_t hv;              // num value heads (HV)
    uint32_t dk;              // head key dim (K)
    uint32_t dv;              // head value dim (V)
    uint32_t qkvDim;          // packed qkv dimension = 2*H*K + HV*V
    uint32_t sBlockNum;       // number of state slots (N)
    uint32_t ubRestBytes;     // remaining UB bytes after fixed allocations
    float    scale;           // head_k_dim ** -0.5
};
#pragma pack(pop)

} // namespace FusedPackedRecurrentGatedDeltaRule

#endif // FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
