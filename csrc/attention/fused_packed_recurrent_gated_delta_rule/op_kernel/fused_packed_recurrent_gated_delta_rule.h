/** Copyright (c) 2026 Huawei Technologies Co., Ltd. SPDX-License-Identifier: Apache-2.0 */
#ifndef FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_KERNEL_H
#define FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_KERNEL_H

#include "kernel_operator.h"
#include "fused_packed_recurrent_gated_delta_rule_tiling_data.h"

namespace FusedPackedRecurrentGatedDeltaRule {

using namespace AscendC;

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t MAX_OUT_BUFFER_NUM = 2;
constexpr uint64_t BF16_NUM_PER_BLOCK = 16;
constexpr uint64_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LENTH = 64;
constexpr uint32_t MAX_REPEAT_TIME = 255;
constexpr uint32_t ADD_FOLD_REDUCE_MIN_K = 128;
constexpr bool kUseAddFoldReduce = false;

struct FPRGDRInitParams {
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
class FPRGDR {
public:
    __aicore__ inline explicit FPRGDR(const FusedPackedRecurrentGatedDeltaRuleTilingData *tilingData)
    {
        B_ = tilingData->b;
        HK_ = tilingData->hk;
        DK_ = tilingData->dk;
        HV_ = tilingData->hv;
        DV_ = tilingData->dv;
        sBlockNum_ = tilingData->sBlockNum;
        scale_ = tilingData->scale;
        vStep_ = tilingData->vStep > 0 ? tilingData->vStep : 16;
        stateOutBufferNum_ = (tilingData->stateOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        attnOutBufferNum_ = (tilingData->attnOutBufferNum == MAX_OUT_BUFFER_NUM) ? MAX_OUT_BUFFER_NUM : BUFFER_NUM;
        if (stateOutBufferNum_ == 0) stateOutBufferNum_ = BUFFER_NUM;
        if (attnOutBufferNum_ == 0) attnOutBufferNum_ = BUFFER_NUM;
        alignK_ = Ceil(DK_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignV_ = Ceil(DV_, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        qkvDim_ = 2 * HK_ * DK_ + HV_ * DV_;
        numKVGroups_ = HV_ / HK_;
    }

    __aicore__ inline void Init(const FPRGDRInitParams &initParams, TPipe *pipe)
    {
        blockIdx = GetBlockIdx();
        if (blockIdx >= GetBlockNum()) return;
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const FPRGDRInitParams &initParams)
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
        uint32_t hvAligned = Ceil(HV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
        constexpr uint32_t BLOCK = 32;

        // Queue buffers (for async data transfer)
        // TQue InitBuffer: (queue, num_buffers, buffer_size)
        uint32_t qQueueSize = (HK_ * DK_ * sizeof(inType) > Ceil(HV_ * sizeof(inType), BLOCK))
                              ? HK_ * DK_ * sizeof(inType)
                              : Ceil(HV_ * sizeof(inType), BLOCK);
        uint32_t kQueueSize = (HK_ * DK_ * sizeof(inType) > Ceil(HV_ * sizeof(inType), BLOCK))
                              ? HK_ * DK_ * sizeof(inType)
                              : Ceil(HV_ * sizeof(inType), BLOCK);
        uint32_t vQueueSize = vStep_ * sizeof(inType);
        uint32_t stateInQueueSize = vStep_ * DK_ * sizeof(stateType);
        uint32_t stateOutQueueSize = vStep_ * DK_ * sizeof(stateType);
        uint32_t attnOutQueueSize = vStep_ * sizeof(outType);
        uint32_t tmpUbSize = (hvAligned > HK_ * alignK_) ? hvAligned : HK_ * alignK_;

        pipe_->InitBuffer(qInQueue_, BUFFER_NUM, qQueueSize);
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, kQueueSize);
        pipe_->InitBuffer(aInQueue_, BUFFER_NUM, Ceil(HV_ * sizeof(inType), BLOCK));
        pipe_->InitBuffer(bInQueue_, BUFFER_NUM, Ceil(HV_ * sizeof(inType), BLOCK));
        pipe_->InitBuffer(tmpInQueue_, BUFFER_NUM, Ceil(hvAligned * sizeof(float), BLOCK));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, vQueueSize);
        pipe_->InitBuffer(stateInQueue_, BUFFER_NUM, stateInQueueSize);
        pipe_->InitBuffer(stateOutQueue_, BUFFER_NUM, stateOutQueueSize);
        pipe_->InitBuffer(attnOutQueue_, BUFFER_NUM, attnOutQueueSize);

        // UB working buffers (TBuf InitBuffer: (buf, size))
        pipe_->InitBuffer(qInUb_, HK_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(kInUb_, HK_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(vInUb_, vStep_ * sizeof(float));
        pipe_->InitBuffer(stateInUb_, vStep_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(broadTmpUb_, vStep_ * alignK_ * sizeof(float));
        pipe_->InitBuffer(deltaUb_, vStep_ * sizeof(float));
        pipe_->InitBuffer(attnUb_, vStep_ * sizeof(float));
        pipe_->InitBuffer(gUb_, hvAligned * sizeof(float) + 64);
        pipe_->InitBuffer(betaUb_, hvAligned * sizeof(float) + 64);
        pipe_->InitBuffer(tmpUb_, tmpUbSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx >= GetBlockNum()) return;

        for (uint64_t bi = blockIdx; bi < B_; bi += GetBlockNum()) {
            int32_t stateIdx = ssmStateIndicesGm_.GetValue(bi);
            if (stateIdx < 0) continue;

            LoadQK(bi);
            ComputeGBeta(bi);
            PipeBarrier<PIPE_V>();

            for (uint32_t hvi = 0; hvi < HV_; hvi++) {
                ProcessHead(bi, hvi, stateIdx);
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
            // Cast writes contiguously at [h*DK_]; rearrange to aligned strides [h*alignK_]
            // Process in reverse to avoid overwriting source data
            for (int32_t h = static_cast<int32_t>(HK_) - 1; h >= 0; h--) {
                uint32_t srcOff = static_cast<uint32_t>(h) * DK_;
                uint32_t dstOff = static_cast<uint32_t>(h) * alignK_;
                for (uint32_t i = DK_; i > 0; i--) {
                    qUb.SetValue(dstOff + i - 1, qUb.GetValue(srcOff + i - 1));
                    kUb.SetValue(dstOff + i - 1, kUb.GetValue(srcOff + i - 1));
                }
            }
            // Zero-fill padding after valid data
            for (uint32_t h = 0; h < HK_; h++) {
                Duplicate<float>(qUb[h * alignK_ + DK_], 0.0f, alignK_ - DK_);
                Duplicate<float>(kUb[h * alignK_ + DK_], 0.0f, alignK_ - DK_);
            }
        }

        // L2Norm Q and K (per head), then scale Q.
        {
            LocalTensor<float> qUb = qInUb_.Get<float>();
            LocalTensor<float> kUb = kInUb_.Get<float>();
            const float eps = 1e-6f;
            for (uint32_t h = 0; h < HK_; h++) {
                float qSq = 0.0f, kSq = 0.0f;
                for (uint32_t i = 0; i < DK_; i++) {
                    float qv = qUb.GetValue(h * alignK_ + i);
                    float kv = kUb.GetValue(h * alignK_ + i);
                    qSq += qv * qv;
                    kSq += kv * kv;
                }
                float qInvRms = 1.0f / sqrt(qSq + eps);
                float kInvRms = 1.0f / sqrt(kSq + eps);
                for (uint32_t i = 0; i < DK_; i++) {
                    qUb.SetValue(h * alignK_ + i,
                                 qUb.GetValue(h * alignK_ + i) * qInvRms);
                    kUb.SetValue(h * alignK_ + i,
                                 kUb.GetValue(h * alignK_ + i) * kInvRms);
                }
            }
        }
        PipeBarrier<PIPE_V>();

        // Scale Q
        if (scale_ != 1.0f) {
            Muls(qInUb_.Get<float>(), qInUb_.Get<float>(), scale_, HK_ * DK_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeGBeta(uint64_t batchIdx)
    {
        uint32_t hCnt = Ceil(HV_, FP32_NUM_PER_BLOCK) * FP32_NUM_PER_BLOCK;
        constexpr uint32_t BLOCK_SIZE = 32;
        // Cast requires at least FP32_NUM_PER_BLOCK elements, ensure data copy covers that
        uint32_t castCnt = (HV_ >= FP32_NUM_PER_BLOCK) ? HV_ : FP32_NUM_PER_BLOCK;
        uint32_t abCopyBytes = castCnt * sizeof(inType);
        LocalTensor<float> gUb = gUb_.Get<float>();
        LocalTensor<float> betaUb = betaUb_.Get<float>();
        LocalTensor<float> tmp = tmpUb_.Get<float>();

        Duplicate(gUb, 0.0f, hCnt);
        Duplicate(betaUb, 0.0f, hCnt);
        Duplicate(tmp, 0.0f, hCnt);
        PipeBarrier<PIPE_V>();

        // Load a and b via queues, cast directly to gUb/betaUb with padded count
        {
            LocalTensor<inType> aLocal = aInQueue_.AllocTensor<inType>();
            DataCopy(aLocal, aGm_[batchIdx * HV_], abCopyBytes);
            aInQueue_.EnQue(aLocal);
            aLocal = aInQueue_.DeQue<inType>();
            Cast(gUb, aLocal, RoundMode::CAST_NONE, castCnt);
            PipeBarrier<PIPE_V>();
            aInQueue_.FreeTensor(aLocal);
        }
        {
            LocalTensor<inType> bLocal = bInQueue_.AllocTensor<inType>();
            DataCopy(bLocal, bGm_[batchIdx * HV_], abCopyBytes);
            bInQueue_.EnQue(bLocal);
            bLocal = bInQueue_.DeQue<inType>();
            Cast(betaUb, bLocal, RoundMode::CAST_NONE, castCnt);
            PipeBarrier<PIPE_V>();
            bInQueue_.FreeTensor(bLocal);
        }

        // Load dtBias via scalar GM reads into TBuf (GlobalTensor<float>::GetValue)
        for (uint32_t i = 0; i < HV_; i++) {
            float dv = dtBiasGm_.GetValue(i);
            gUb.SetValue(i, gUb.GetValue(i) + dv);
        }
        PipeBarrier<PIPE_V>();

        // Load A_log via scalar GM reads into TBuf, then vector Exp
        for (uint32_t i = 0; i < HV_; i++) {
            tmp.SetValue(i, aLogGm_.GetValue(i));
        }
        PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, castCnt);
        Muls(tmp, tmp, -1.0f, castCnt);
        PipeBarrier<PIPE_V>();

        // Compute exp_g: softplus(gUb) * (-exp(A_log)), then exp
        Exp(gUb, gUb, castCnt);
        Adds(gUb, gUb, 1.0f, castCnt);
        Ln(gUb, gUb, castCnt);

        // tmp = -exp(A_log), multiply with softplus result
        Mul(gUb, gUb, tmp, castCnt);
        Exp(gUb, gUb, castCnt);

        // Compute beta: sigmoid(betaUb)
        Muls(betaUb, betaUb, -1.0f, castCnt);
        Exp(betaUb, betaUb, castCnt);
        Adds(betaUb, betaUb, 1.0f, castCnt);
        Reciprocal(betaUb, betaUb, castCnt);
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessHead(uint64_t batchIdx, uint64_t hv, int32_t stateIdx)
    {
        PipeBarrier<PIPE_V>();
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

        // Zero-initialize to avoid non-deterministic uninitialized data
        Duplicate(broadTmpUb, 0.0f, vStep_ * alignK_);
        Duplicate(deltaUb, 0.0f, vStep_);
        Duplicate(attnUb, 0.0f, vStep_);
        PipeBarrier<PIPE_V>();

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

            uint32_t curVPadded = (curV >= FP32_NUM_PER_BLOCK) ? curV : FP32_NUM_PER_BLOCK;
            Sub(deltaUb, vUb, deltaUb, curVPadded);
            Muls(deltaUb, deltaUb, betaVal, curVPadded);

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
        PipeBarrier<PIPE_V>();
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
        // Pad cols to FP32_NUM_PER_BLOCK for reliable vector operation
        uint32_t paddedCols = (cols >= FP32_NUM_PER_BLOCK) ? cols : FP32_NUM_PER_BLOCK;
        for (uint32_t i = 0; i < alignK_; i += REPEAT_LENTH) {
            uint64_t mask = Std::min(REPEAT_LENTH, alignK_ - i);
            for (uint32_t j = 0; j < paddedCols; j += MAX_REPEAT_TIME) {
                uint64_t repeatTime = Std::min(MAX_REPEAT_TIME, paddedCols - j);
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

    __aicore__ inline void ReduceSumDispatch(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor,
                                              uint32_t rows)
    {
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
    TQue<QuePosition::VECIN, BUFFER_NUM> aInQueue_, bInQueue_, tmpInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> stateInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> stateOutQueue_, attnOutQueue_;

    GlobalTensor<inType> mixedQkvGm_, aGm_, bGm_;
    GlobalTensor<float> aLogGm_, dtBiasGm_;
    GlobalTensor<stateType> initStateGm_, finalStateGm_;
    GlobalTensor<int32_t> ssmStateIndicesGm_;
    GlobalTensor<outType> attnOutGm_;
};

} // namespace FusedPackedRecurrentGatedDeltaRule
#endif
