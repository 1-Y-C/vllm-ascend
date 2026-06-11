#include "packed_recurrent_gated_delta_rule.h"
#include "packed_recurrent_gated_delta_rule_tiling_data.h"

using namespace AscendC;
using namespace PackedRecurrentGatedDeltaRule;

extern "C" __global__ __aicore__ void
packed_recurrent_gated_delta_rule(GM_ADDR mixed_qkv, GM_ADDR a, GM_ADDR b,
                                  GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR state,
                                  GM_ADDR ssm_state_indices, GM_ADDR attn_out,
                                  GM_ADDR state_out, GM_ADDR workspaceGM,
                                  GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(PackedRecurrentGatedDeltaRuleTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    PackedRGDR<bfloat16_t, bfloat16_t, bfloat16_t> op(&tilingData);
    PackedRGDRInitParams initParams{mixed_qkv, a, b, a_log, dt_bias, state,
                                    ssm_state_indices, attn_out, state_out};
    op.Init(initParams, &pipe);
    op.Process();
}
