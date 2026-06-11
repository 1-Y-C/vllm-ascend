#ifndef PACKED_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
#define PACKED_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace PackedRecurrentGatedDeltaRule {
#pragma pack(push, 8)
struct alignas(8) PackedRecurrentGatedDeltaRuleTilingData {
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t b;
    uint32_t hk;
    uint32_t dk;
    uint32_t hv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
};
#pragma pack(pop)
} // PackedRecurrentGatedDeltaRule

#endif // PACKED_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
