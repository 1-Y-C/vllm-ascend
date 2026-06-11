#ifndef __OP_HOST_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
#define __OP_HOST_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/error_log.h"
#include "../op_kernel/packed_recurrent_gated_delta_rule_tiling_data.h"

namespace optiling {

struct PackedRecurrentGatedDeltaRuleCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct PackedRecurrentGatedDeltaRuleInfo {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "PackedRecurrentGatedDeltaRule";
    float userScale = 0.0f;
};

class PackedRecurrentGatedDeltaRuleTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit PackedRecurrentGatedDeltaRuleTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~PackedRecurrentGatedDeltaRuleTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();

    using HostRuleFn = ge::graphStatus (PackedRecurrentGatedDeltaRuleTiling::*)();
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

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus CalUbSize();
    ge::graphStatus FillTilingShapeData();
    int64_t CalcFixedUbBytes(int64_t hv, int64_t dv, int64_t dk, int64_t hk) const;
    int64_t CalcWorkingUbBytes(int64_t hv, int64_t dv, int64_t dk, int64_t hk) const;
    int64_t CalcVStepCoeff(int64_t dk, uint32_t stateOutBufferNum, uint32_t attnOutBufferNum) const;
    bool EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes, int64_t dk,
                               uint32_t stateOutBufferNum, uint32_t attnOutBufferNum,
                               BufferProfile &profile) const;
    bool IsBetterProfile(const BufferProfile &candidate, const BufferProfile &current) const;
    ge::graphStatus FinalizeVStepFromUb();
    ge::graphStatus RuleInitUbCalcContext();
    ge::graphStatus RuleCalcFixedUbBytes();
    ge::graphStatus RuleCalcWorkingUbBytes();
    ge::graphStatus RuleCalcVStepCoeff();
    ge::graphStatus RuleFinalizeVStepFromUb();

    PackedRecurrentGatedDeltaRuleCompileInfo compileInfo_;
    PackedRecurrentGatedDeltaRule::PackedRecurrentGatedDeltaRuleTilingData tilingData_;
    PackedRecurrentGatedDeltaRuleInfo inputParams_;
    UbCalcContext ubCalcCtx_;
};

} // namespace optiling
#endif // __OP_HOST_PACKED_RECURRENT_GATED_DELTA_RULE_TILING_H__
