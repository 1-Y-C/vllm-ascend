#ifndef __PACKED_RECURRENT_GATED_DELTA_RULE_KERNEL_H_
#define __PACKED_RECURRENT_GATED_DELTA_RULE_KERNEL_H_

#include "kernel_operator.h"
#include "packed_recurrent_gated_delta_rule_tiling_data.h"

namespace PackedRecurrentGatedDeltaRule {

using namespace AscendC;

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64;
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;
constexpr bool kUseAddFoldReduce = false;

struct PackedRGDRInitParams {
    GM_ADDR mixedQkv;
    GM_ADDR a;
    GM_ADDR b;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR initState;
    GM_ADDR ssmStateIndices;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename inType, typename outType, typename stateType>
class PackedRGDR {
public:
    __aicore__ inline explicit PackedRGDR(const PackedRecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        HK_ = tilingData->hk;
        DK_ = tilingData->dk;
        HV_ = tilingData->hv;
        DV_ = tilingData->dv;
        sBlockNum_ = tilingData->sBlockNum;
        scale_ = tilingData->scale;
        vStep_ = tilingData->vStep;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        alignK_ = Ceil(DK_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(DV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        qkvDim_ = 2 * HK_ * DK_ + HV_ * DV_;
        numKVGroups_ = HV_ / HK_;
    }

    __aicore__ inline void Init(const PackedRGDRInitParams &initParams, TPipe *pipe)
    {
        blockIdx = GetBlockIdx();
        if (blockIdx >= GetBlockNum()) return;
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const PackedRGDRInitParams &initParams)
    {
        mixedQkvGm_.SetGlobalBuffer((__gm__ inType *)initParams.mixedQkv);
        aGm_.SetGlobalBuffer((__gm__ inType *)initParams.a);
        bGm_.SetGlobalBuffer((__gm__ inType *)initParams.b);
        aLogGm_.SetGlobalBuffer((__gm__ float *)initParams.aLog);
        dtBiasGm_.SetGlobalBuffer((__gm__ float *)initParams.dtBias);
        initStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.initState);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType *)initParams.finalState);
        ssmStateIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)initParams.ssmStateIndices);
        attnOutGm_.SetGlobalBuffer((__gm__ outType *)initParams.attnOut);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(qInQueue_, BUFFER_NUM, HK_ * alignK_ * sizeof(inType));
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, HK_ * alignK_ * sizeof(inType));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, vStep_ * sizeof(inType));
        pipe_->InitBuffer(stateInQueue_, BUFFER_NUM, vStep_ * alignK_ * sizeof(stateType));
        pipe_->InitBuffer(stateOutQueue_, stateOutBufferNum_, vStep_ * alignK_ * sizeof(stateType));
        pipe_->InitBuffer(attnOutQueue_, attnOutBufferNum_, vStep_ * sizeof(outType));

        pipe_->InitBuffer(qInUb_, HK_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(kInUb_, HK_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(vInUb_, vStep_ * sizeof(float));
        pipe_->InitBuffer(stateInUb_, vStep_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(broadTmpUb_, vStep_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(deltaUb_, vStep_ * sizeof(float));
        pipe_->InitBuffer(attnUb_, vStep_ * sizeof(float));
        pipe_->InitBuffer(gUb_, Ceil(HV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK * sizeof(float));
        pipe_->InitBuffer(betaUb_, Ceil(HV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK * sizeof(float));
        pipe_->InitBuffer(tmpUb_, (HK_ * alignK_ + 2 * Ceil(HV_, FP32_NUM_PER_BLOCK)) * FP32_NUM_PER_BLOCK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t batchIdx = blockIdx; batchIdx < B_; batchIdx += GetBlockNum()) {
            int32_t stateIdx = ssmStateIndicesGm_.GetValue(batchIdx);
            if (stateIdx < 0) continue;
            if (static_cast<uint32_t>(stateIdx) >= sBlockNum_) continue;

            ComputeGBeta(batchIdx);

            LoadQK(batchIdx);

            for (uint64_t hv = 0; hv < HV_; hv++) {
                Duplicate(stateInUb_.Get<float>(), 0.0f, vStep_ * alignK_);
                ProcessHead(batchIdx, hv, stateIdx);
            }
        }
    }

private:
    __aicore__ inline void LoadQK(uint64_t batchIdx)
    {
        uint64_t batchOffset = batchIdx * qkvDim_;
        uint64_t qOffset = batchOffset;
        uint64_t kOffset = batchOffset + HK_ * DK_;

        LocalTensor<inType> qLocal = qInQueue_.AllocTensor<inType>();
        LocalTensor<inType> kLocal = kInQueue_.AllocTensor<inType>();
        DataCopy(qLocal, mixedQkvGm_[qOffset], HK_ * DK_ * sizeof(inType));
        DataCopy(kLocal, mixedQkvGm_[kOffset], HK_ * DK_ * sizeof(inType));
        qInQueue_.EnQue(qLocal);
        kInQueue_.EnQue(kLocal);

        qLocal = qInQueue_.DeQue<inType>();
        kLocal = kInQueue_.DeQue<inType>();
        Cast(qInUb_.Get<float>(), qLocal, RoundMode::CAST_NONE, HK_ * DK_);
        Cast(kInUb_.Get<float>(), kLocal, RoundMode::CAST_NONE, HK_ * DK_);
        qInQueue_.FreeTensor(qLocal);
        kInQueue_.FreeTensor(kLocal);

        if (alignK_ > DK_) {
            LocalTensor<float> qUb = qInUb_.Get<float>();
            LocalTensor<float> kUb = kInUb_.Get<float>();
            for (uint32_t h = 0; h < HK_; h++) {
                Duplicate<float>(qUb[h * alignK_ + DK_], 0.0f, alignK_ - DK_);
                Duplicate<float>(kUb[h * alignK_ + DK_], 0.0f, alignK_ - DK_);
            }
        }
    }

    __aicore__ inline void ComputeGBeta(uint64_t batchIdx)
    {
        uint32_t hCnt = Ceil(HV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
        constexpr uint32_t BLOCK_SIZE = 32;
        uint32_t alignedBytes = Ceil(HV_ * sizeof(inType), BLOCK_SIZE);
        LocalTensor<float> gUb = gUb_.Get<float>();
        LocalTensor<float> betaUb = betaUb_.Get<float>();
        LocalTensor<float> tmp = tmpUb_.Get<float>();

        Duplicate(gUb, 0.0f, hCnt);
        Duplicate(betaUb, 0.0f, hCnt);
        Duplicate(tmp, 0.0f, hCnt);
        PipeBarrier<PIPE_V>();

        // Load a via qInQueue_
        {
            LocalTensor<inType> aLocal = qInQueue_.AllocTensor<inType>();
            DataCopy(aLocal, aGm_[batchIdx * HV_], alignedBytes);
            qInQueue_.EnQue(aLocal);
            aLocal = qInQueue_.DeQue<inType>();
            Cast(gUb, aLocal, RoundMode::CAST_NONE, HV_);
            PipeBarrier<PIPE_V>();
            qInQueue_.FreeTensor(aLocal);
        }
        // Load b via kInQueue_
        {
            LocalTensor<inType> bLocal = kInQueue_.AllocTensor<inType>();
            DataCopy(bLocal, bGm_[batchIdx * HV_], alignedBytes);
            kInQueue_.EnQue(bLocal);
            bLocal = kInQueue_.DeQue<inType>();
            Cast(betaUb, bLocal, RoundMode::CAST_NONE, HV_);
            PipeBarrier<PIPE_V>();
            kInQueue_.FreeTensor(bLocal);
        }

        // Load dtBias via tmp (vectorized) and add to gUb
        {
            uint32_t dtAlignedBytes = Ceil(HV_ * sizeof(float), BLOCK_SIZE);
            DataCopy(tmp, dtBiasGm_[0], dtAlignedBytes);
            PipeBarrier<PIPE_V>();
            Add(gUb, gUb, tmp, HV_);
        }

        // Load aLog
        {
            uint32_t aLogAlignedBytes = Ceil(HV_ * sizeof(float), BLOCK_SIZE);
            DataCopy(tmp, aLogGm_[batchIdx * HV_], aLogAlignedBytes);
            PipeBarrier<PIPE_V>();
            float savedALog = tmp.GetValue(0);
            Duplicate(tmp, 0.0f, hCnt);
            PipeBarrier<PIPE_V>();
            tmp.SetValue(0, savedALog);
            PipeBarrier<PIPE_V>();
        }

        Exp(gUb, gUb, hCnt);
        Adds(gUb, gUb, 1.0f, hCnt);
        Log(gUb, gUb, hCnt);

        Exp(tmp, tmp, hCnt);
        Muls(tmp, tmp, -1.0f, hCnt);

        Mul(gUb, gUb, tmp, hCnt);
        Exp(gUb, gUb, hCnt);

        Muls(betaUb, betaUb, -1.0f, hCnt);
        Exp(betaUb, betaUb, hCnt);
        Adds(betaUb, betaUb, 1.0f, hCnt);
        Reciprocal(betaUb, betaUb, hCnt);
    }

    __aicore__ inline void ProcessHead(uint64_t batchIdx, uint64_t hv, int32_t stateIdx)
    {
        uint64_t hk = hv / numKVGroups_;
        float exp_g = gUb_.Get<float>().GetValue(hv);
        float betaVal = betaUb_.Get<float>().GetValue(hv);

        LocalTensor<float> kUb = kInUb_.Get<float>();
        LocalTensor<float> qUb = qInUb_.Get<float>();
        LocalTensor<float> vUb = vInUb_.Get<float>();
        LocalTensor<float> stateUb = stateInUb_.Get<float>();
        LocalTensor<float> broadTmpUb = broadTmpUb_.Get<float>();
        LocalTensor<float> deltaUb = deltaUb_.Get<float>();
        LocalTensor<float> attnUb = attnUb_.Get<float>();

        for (uint64_t v0 = 0; v0 < DV_; v0 += vStep_) {
            uint32_t curV = Std::min(vStep_, DV_ - v0);

            LoadVChunk(batchIdx, hv, v0, curV);

            uint64_t stateOff = ((uint64_t)stateIdx * HV_ + hv) * DV_ * DK_ + v0 * DK_;
            LoadState(stateOff, curV);

            for (uint32_t vi = 0; vi < curV; vi++) {
                Muls(stateUb[vi * alignK_], stateUb[vi * alignK_], exp_g, alignK_);
            }

            MatVecMul(stateUb, kUb[hk * alignK_], broadTmpUb, curV, false);
            ReduceSumDispatch(deltaUb, broadTmpUb, curV);

            Sub(deltaUb, vUb, deltaUb, curV);

            Muls(deltaUb, deltaUb, betaVal, curV);

            for (uint32_t vi = 0; vi < curV; vi++) {
                Muls(broadTmpUb[vi * alignK_], kUb[hk * alignK_], deltaUb.GetValue(vi), alignK_);
            }
            PipeBarrier<PIPE_V>();
            Add(stateUb, stateUb, broadTmpUb, curV * alignK_);

            MatVecMul(stateUb, qUb[hk * alignK_], broadTmpUb, curV, false);
            ReduceSumDispatch(attnUb, broadTmpUb, curV);

            WriteAttn(batchIdx, hv, v0, curV);
            WriteState(stateOff, curV);
        }
    }

    __aicore__ inline void LoadVChunk(uint64_t batchIdx, uint64_t hv, uint64_t v0, uint32_t curV)
    {
        uint64_t vGmOffset = batchIdx * qkvDim_ + 2 * HK_ * DK_ + hv * DV_ + v0;
        LocalTensor<inType> vLocal = vInQueue_.AllocTensor<inType>();
        DataCopy(vLocal, mixedQkvGm_[vGmOffset], curV * sizeof(inType));
        vInQueue_.EnQue(vLocal);
        vLocal = vInQueue_.DeQue<inType>();
        Cast(vInUb_.Get<float>(), vLocal, RoundMode::CAST_NONE, curV);
        PipeBarrier<PIPE_V>();
        vInQueue_.FreeTensor(vLocal);
    }

    __aicore__ inline void LoadState(uint64_t stateOff, uint32_t curV)
    {
        LocalTensor<stateType> stateLocal = stateInQueue_.AllocTensor<stateType>();
        DataCopyParams params{static_cast<uint16_t>(curV),
                               static_cast<uint16_t>(DK_ * sizeof(stateType)),
                               0, 0};
        DataCopyPadParams padParams{};
        DataCopyPad(stateLocal, initStateGm_[stateOff], params, padParams);
        stateInQueue_.EnQue(stateLocal);
        stateLocal = stateInQueue_.DeQue<stateType>();
        LocalTensor<float> sUb = stateInUb_.Get<float>();
        if (alignK_ == DK_) {
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(sUb, stateLocal, DK_ * curV);
            } else {
                Cast(sUb, stateLocal, RoundMode::CAST_NONE, DK_ * curV);
            }
        } else {
            for (uint32_t vi = 0; vi < curV; vi++) {
                if constexpr (std::is_same<stateType, float32_t>()) {
                    DataCopy(sUb[vi * alignK_], stateLocal[vi * DK_], DK_);
                } else {
                    Cast(sUb[vi * alignK_], stateLocal[vi * DK_], RoundMode::CAST_NONE, DK_);
                }
                Duplicate(sUb[vi * alignK_ + DK_], 0.0f, alignK_ - DK_);
            }
        }
        stateInQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void WriteAttn(uint64_t batchIdx, uint64_t hv, uint64_t v0, uint32_t curV)
    {
        // Write attnUb (normal behavior)
        LocalTensor<outType> outLocal = attnOutQueue_.AllocTensor<outType>();
        uint64_t attnOffset = batchIdx * HV_ * DV_ + hv * DV_ + v0;
        Cast(outLocal, attnUb_.Get<float>(), RoundMode::CAST_ROUND, curV);
        PipeBarrier<PIPE_V>();
        attnOutQueue_.EnQue(outLocal);
        outLocal = attnOutQueue_.DeQue<outType>();
        DataCopyParams params{1, static_cast<uint16_t>(curV * sizeof(outType)), 0, 0};
        DataCopyPad(attnOutGm_[attnOffset], outLocal, params);
        attnOutQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void WriteState(uint64_t stateOff, uint32_t curV)
    {
        LocalTensor<stateType> stateLocal = stateOutQueue_.AllocTensor<stateType>();
        LocalTensor<float> sUb = stateInUb_.Get<float>();
        if (alignK_ == DK_) {
            if constexpr (std::is_same<stateType, float32_t>()) {
                DataCopy(stateLocal, sUb, DK_ * curV);
            } else {
                Cast(stateLocal, sUb, RoundMode::CAST_ROUND, DK_ * curV);
            }
        } else {
            for (uint32_t vi = 0; vi < curV; vi++) {
                if constexpr (std::is_same<stateType, float32_t>()) {
                    DataCopy(stateLocal[vi * DK_], sUb[vi * alignK_], DK_);
                } else {
                    Cast(stateLocal[vi * DK_], sUb[vi * alignK_], RoundMode::CAST_ROUND, DK_);
                }
            }
        }
        PipeBarrier<PIPE_V>();
        stateOutQueue_.EnQue(stateLocal);
        stateLocal = stateOutQueue_.DeQue<stateType>();
        DataCopyParams params{static_cast<uint16_t>(curV),
                               static_cast<uint16_t>(DK_ * sizeof(stateType)),
                               0, 0};
        DataCopyPad(finalStateGm_[stateOff], stateLocal, params);
        stateOutQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void MatVecMul(const LocalTensor<float> &cubeTensor, const LocalTensor<float> &vecTensor,
                                     LocalTensor<float> &dstTensor, uint32_t cols, bool isAdd)
    {
        uint8_t repeatStride = alignK_ / FP32_NUM_PER_BLOCK;
        for (uint32_t i = 0; i < alignK_; i += REPEAT_LENTH) {
            uint64_t mask = Std::min(REPEAT_LENTH, alignK_ - i);
            for (uint32_t j = 0; j < cols; j += MAX_REPEAT_TIME) {
                uint64_t repeatTime = Std::min(MAX_REPEAT_TIME, cols - j);
                if (isAdd) {
                    MulAddDst(dstTensor[j * alignK_ + i], cubeTensor[j * alignK_ + i], vecTensor[i], mask, repeatTime,
                              {1, 1, 1, repeatStride, repeatStride, 0});
                } else {
                    Mul(dstTensor[j * alignK_ + i], cubeTensor[j * alignK_ + i], vecTensor[i], mask, repeatTime,
                        {1, 1, 1, repeatStride, repeatStride, 0});
                }
            }
        }
    }

    __aicore__ inline void ReduceSumBaseline(LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
                                              uint32_t rows)
    {
        uint32_t rsShape[2] = {rows, alignK_};
        ReduceSum<float, Pattern::Reduce::AR, true>(dstTensor, srcTensor, rsShape, true);
    }

    __aicore__ inline bool CanUseK128AddFoldFastPath(uint32_t rows) const
    {
        if (alignK_ != ADD_FOLD_REDUCE_MIN_K) return false;
        if (rows == 0 || rows > MAX_REPEAT_TIME) return false;
        return true;
    }

    __aicore__ inline void ReduceSumAddFoldK128(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                                 uint32_t rows)
    {
        const uint8_t repeatTime = static_cast<uint8_t>(rows);
        const uint8_t rowRepStride = static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK);
        Add(srcTensor[REPEAT_LENTH], srcTensor, srcTensor[REPEAT_LENTH], REPEAT_LENTH, repeatTime,
            {1, 1, 1, rowRepStride, rowRepStride, rowRepStride});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstTensor, srcTensor[REPEAT_LENTH], REPEAT_LENTH, repeatTime, 1, 1, rowRepStride);
    }

    __aicore__ inline void ReduceSumAddFold(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                             uint32_t rows)
    {
        if (alignK_ < REPEAT_LENTH) {
            ReduceSumBaseline(dstTensor, srcTensor, rows);
            return;
        }
        if ((alignK_ & (alignK_ - 1)) != 0) {
            ReduceSumBaseline(dstTensor, srcTensor, rows);
            return;
        }
        if (CanUseK128AddFoldFastPath(rows)) {
            ReduceSumAddFoldK128(dstTensor, srcTensor, rows);
            return;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t rowOffset = row * alignK_;
            uint32_t activeLen = alignK_;
            while (activeLen > REPEAT_LENTH) {
                uint32_t half = activeLen >> 1;
                Add(srcTensor[rowOffset], srcTensor[rowOffset], srcTensor[rowOffset + half], half);
                PipeBarrier<PIPE_V>();
                activeLen = half;
            }
            WholeReduceSum(dstTensor[row], srcTensor[rowOffset], REPEAT_LENTH, 1, 1, 1, FP32_NUM_PER_BLOCK);
        }
    }

    __aicore__ inline void ReduceSumDispatch(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                              uint32_t rows)
    {
        if (kUseAddFoldReduce && alignK_ >= ADD_FOLD_REDUCE_MIN_K) {
            ReduceSumAddFold(dstTensor, srcTensor, rows);
            return;
        }
        ReduceSumBaseline(dstTensor, srcTensor, rows);
    }

private:
    uint32_t B_, HK_, DK_, HV_, DV_;
    uint32_t vStep_, stateOutBufferNum_, attnOutBufferNum_;
    uint32_t alignK_, alignV_, qkvDim_, numKVGroups_, sBlockNum_;
    float scale_;
    uint64_t blockIdx;

    TPipe *pipe_;

    TBuf<TPosition::VECCALC> qInUb_, kInUb_, vInUb_, stateInUb_, broadTmpUb_, deltaUb_, attnUb_;
    TBuf<TPosition::VECCALC> gUb_, betaUb_, tmpUb_;

    TQue<QuePosition::VECIN, BUFFER_NUM> qInQueue_, kInQueue_, vInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> stateInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> stateOutQueue_, attnOutQueue_;

    GlobalTensor<inType> mixedQkvGm_, aGm_, bGm_;
    GlobalTensor<float> aLogGm_, dtBiasGm_;
    GlobalTensor<stateType> initStateGm_, finalStateGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<outType> attnOutGm_;
};

} // namespace PackedRecurrentGatedDeltaRule
#endif
