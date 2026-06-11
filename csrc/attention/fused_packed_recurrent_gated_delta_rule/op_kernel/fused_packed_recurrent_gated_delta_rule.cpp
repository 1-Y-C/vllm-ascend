/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#include "fused_packed_recurrent_gated_delta_rule.h"
#include "fused_packed_recurrent_gated_delta_rule_tiling_data.h"

using namespace AscendC;
using namespace FusedPackedRecurrentGatedDeltaRule;

extern "C" __global__ __aicore__ void
fused_packed_recurrent_gated_delta_rule(GM_ADDR mq, GM_ADDR a, GM_ADDR b,
                                         GM_ADDR al, GM_ADDR dt,
                                         GM_ADDR st, GM_ADDR ssi,
                                         GM_ADDR out, GM_ADDR stOut,
                                         GM_ADDR ws, GM_ADDR tg)
{
    REGISTER_TILING_DEFAULT(FusedPackedRecurrentGatedDeltaRuleTilingData);
    GET_TILING_DATA(tilingData, tg);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    FPRGDR<bfloat16_t, bfloat16_t, float> op(&tilingData);
    // For in-place state: use st for both initState and finalState
    FPRGDRInitParams initParams{mq, a, b, al, dt, st, ssi, out, st};
    op.Init(initParams, &pipe);
    op.Process();
}
