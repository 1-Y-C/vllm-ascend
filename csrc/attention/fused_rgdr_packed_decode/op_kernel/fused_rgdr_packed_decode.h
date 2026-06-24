/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 *
 * Fused RGDR Packed Decode — Decode-phase kernel for Gated Delta Rule.
 * vStep-tiled state processing with double-buffered data movement.
 */

#ifndef FUSED_RGDR_PACKED_DECODE_KERNEL_H_
#define FUSED_RGDR_PACKED_DECODE_KERNEL_H_

#include "kernel_operator.h"
#include "fused_rgdr_packed_decode_tiling_data.h"

namespace NsFusedRgd {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t ST_BUF_NUM = 2;   // double-buffered state transfer
constexpr uint32_t REDUCE_TMP_BYTES = 512U;
constexpr uint32_t SIGMOID_TMP_BYTES = 256U;
constexpr uint32_t MIN_BUF = 32U;
constexpr uint32_t DIMS_PER_BLOCK = 8U;
constexpr uint32_t FOLD_HALF = 64U;  // half of DK=128

template <typename S>
class KernelFusedRgd {
public:
    __aicore__ inline KernelFusedRgd() {}
    __aicore__ inline void Init(
        GM_ADDR mq, GM_ADDR a, GM_ADDR b, GM_ADDR al, GM_ADDR db,
        GM_ADDR st, GM_ADDR si, GM_ADDR o, GM_ADDR so,
        const FusedRgdrPackedDecodeTilingData* td, TPipe* pipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitLocalBuffers();
    __aicore__ inline void ProcessBatch(uint32_t bi, int32_t sid);
    __aicore__ inline void PrefetchState(uint64_t sOff, uint32_t rows);
    __aicore__ inline void LoadPrefetchedState(uint32_t rows);
    __aicore__ inline void WriteBackState(uint64_t sOff, uint32_t rows);

    TPipe* pipe_;
    FusedRgdrPackedDecodeTilingData td_;
    GlobalTensor<bfloat16_t> mqGm_, aGm_, bGm_, outGm_;
    GlobalTensor<float>      alGm_, dbGm_;
    GlobalTensor<S>          stGm_, stoGm_;
    GlobalTensor<int32_t>    siGm_;

    // Data-movement queues (non-state)
    TQue<QuePosition::VECIN, BUFFER_NUM>  qSid_, qMq_, qA_, qB_, qAl_, qDb_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qOut_;
    // State double-buffered queues
    TQue<QuePosition::VECIN, ST_BUF_NUM>  qStRd_;
    TQue<QuePosition::VECOUT, ST_BUF_NUM> qStWr_;

    // Compute buffers
    TBuf<TPosition::VECCALC> cQ_, cK_, cV_, cSt_, cStProd_, cRd_, cOutFp32_, cTmp_;
    TBuf<TPosition::VECCALC> cGateAl_, cGateA_, cGateS_, cGateE_, cGateSg_, cBeta_;
    TBuf<TPosition::VECCALC> cSqBuf_, cScBuf_;

    // Fixed tensor views
    LocalTensor<float>   qtB_, kB_, vB_, stB_, stP_, sq_, sr_, rdT_, oB_, tr_;
    LocalTensor<float>   alT_, aT_, sT_, eG_, beB_;
    LocalTensor<uint8_t> sgT_;
};

template<typename S>
__aicore__ inline void KernelFusedRgd<S>::Init(
    GM_ADDR mq, GM_ADDR a, GM_ADDR b, GM_ADDR al, GM_ADDR db,
    GM_ADDR st, GM_ADDR si, GM_ADDR o, GM_ADDR so,
    const FusedRgdrPackedDecodeTilingData* td, TPipe* pipe)
{
    pipe_ = pipe;
    td_ = *td;
    uint32_t L = td_.L_qkv, HV = td_.HV, DV = td_.DV, DK = td_.DK;
    mqGm_.SetGlobalBuffer((__gm__ bfloat16_t*)mq, td_.B * L);
    aGm_.SetGlobalBuffer((__gm__ bfloat16_t*)a,  td_.B * HV);
    bGm_.SetGlobalBuffer((__gm__ bfloat16_t*)b,  td_.B * HV);
    alGm_.SetGlobalBuffer((__gm__ float*)al, HV);
    dbGm_.SetGlobalBuffer((__gm__ float*)db, HV);
    stGm_.SetGlobalBuffer((__gm__ S*)st,  td_.N * HV * DV * DK);
    stoGm_.SetGlobalBuffer((__gm__ S*)so, td_.N * HV * DV * DK);
    siGm_.SetGlobalBuffer((__gm__ int32_t*)si, td_.B);
    outGm_.SetGlobalBuffer((__gm__ bfloat16_t*)o, td_.B * 1 * HV * DV);

    InitLocalBuffers();
}


template<typename S>
__aicore__ inline void KernelFusedRgd<S>::InitLocalBuffers()
{
    uint32_t HK = td_.HK, HV = td_.HV, DK = td_.DK, DV = td_.DV;
    uint32_t dkA = td_.dkAligned, dvA = td_.dvAligned;
    uint32_t L  = td_.L_qkv;
    uint32_t vStep = td_.vStep;

    // --- TQue data-movement queues (non-state) ---
    pipe_->InitBuffer(qSid_,  BUFFER_NUM, 32U);
    pipe_->InitBuffer(qMq_,   BUFFER_NUM, L * sizeof(bfloat16_t));
    pipe_->InitBuffer(qA_,    BUFFER_NUM, (HV * sizeof(bfloat16_t) < MIN_BUF) ? MIN_BUF : (HV * sizeof(bfloat16_t)));
    pipe_->InitBuffer(qB_,    BUFFER_NUM, (HV * sizeof(bfloat16_t) < MIN_BUF) ? MIN_BUF : (HV * sizeof(bfloat16_t)));
    pipe_->InitBuffer(qAl_,   BUFFER_NUM, (HV * sizeof(float)      < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(qDb_,   BUFFER_NUM, (HV * sizeof(float)      < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(qOut_,  BUFFER_NUM, (dvA * sizeof(bfloat16_t) < MIN_BUF) ? MIN_BUF : (dvA * sizeof(bfloat16_t)));

    // --- State double-buffered queues ---
    uint32_t stChunkBytes = vStep * dkA * sizeof(S);
    pipe_->InitBuffer(qStRd_, ST_BUF_NUM, stChunkBytes);
    pipe_->InitBuffer(qStWr_, ST_BUF_NUM, stChunkBytes);

    // --- Compute buffers (VECCALC) ---
    pipe_->InitBuffer(cQ_,     HK * dkA * sizeof(float));
    pipe_->InitBuffer(cK_,     HK * dkA * sizeof(float));
    pipe_->InitBuffer(cV_,     HV * dvA * sizeof(float));
    pipe_->InitBuffer(cSt_,    vStep * dkA * sizeof(float));
    pipe_->InitBuffer(cStProd_, vStep * dkA * sizeof(float));
    pipe_->InitBuffer(cRd_,    REDUCE_TMP_BYTES);
    pipe_->InitBuffer(cOutFp32_, dvA * sizeof(float));
    pipe_->InitBuffer(cSqBuf_, dkA * sizeof(float));
    pipe_->InitBuffer(cScBuf_, 32U);
    pipe_->InitBuffer(cTmp_,   dkA * sizeof(float));
    pipe_->InitBuffer(cGateAl_, (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateA_,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateS_,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateE_,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateSg_, SIGMOID_TMP_BYTES);
    pipe_->InitBuffer(cBeta_,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));

    // --- Fixed tensor views ---
    qtB_ = cQ_.Get<float>(HK * dkA);
    kB_  = cK_.Get<float>(HK * dkA);
    vB_  = cV_.Get<float>(HV * dvA);
    stB_ = cSt_.Get<float>(vStep * dkA);
    stP_ = cStProd_.Get<float>(vStep * dkA);
    sq_  = cSqBuf_.Get<float>(dkA);
    sr_  = cScBuf_.Get<float>(8);
    rdT_ = cRd_.Get<float>(REDUCE_TMP_BYTES / sizeof(float));
    oB_  = cOutFp32_.Get<float>(dvA);
    tr_  = cTmp_.Get<float>(dkA);

    alT_ = cGateAl_.Get<float>(HV);
    aT_  = cGateA_.Get<float>(HV);
    sT_  = cGateS_.Get<float>(HV);
    eG_  = cGateE_.Get<float>(HV);
    sgT_ = cGateSg_.Get<uint8_t>(SIGMOID_TMP_BYTES);
    beB_ = cBeta_.Get<float>(HV);
}


template<typename S>
__aicore__ inline void KernelFusedRgd<S>::Process() {
    if (td_.B == 0) return;

    uint32_t HK = td_.HK, HV = td_.HV, DK = td_.DK, DV = td_.DV;
    uint32_t dkA = td_.dkAligned, dvA = td_.dvAligned;
    uint32_t L  = td_.L_qkv, G = td_.G;
    float ep = td_.eps, scaleVal = td_.scaleVal;
    uint32_t vStep = td_.vStep;

    uint32_t totalUnits = td_.totalUnits;
    uint32_t blockNum = GetBlockNum();
    if (totalUnits == 0 || blockNum == 0) return;
    uint32_t unitsPerCore = (totalUnits + blockNum - 1) / blockNum;
    uint32_t startUnit = GetBlockIdx() * unitsPerCore;
    uint32_t endUnit = (startUnit + unitsPerCore > totalUnits) ? totalUnits : (startUnit + unitsPerCore);
    if (startUnit >= endUnit) return;

    int32_t lastBi = -1;
    int32_t sid = 0;
    bool skipBatch = false;

    for (uint32_t ui = startUnit; ui < endUnit; ++ui) {
        uint32_t bi = ui / HV;
        uint32_t hvi = ui % HV;

        // --- Per-batch setup (once per batch on this core) ---
        if (static_cast<int32_t>(bi) != lastBi) {
            lastBi = static_cast<int32_t>(bi);
            skipBatch = false;

            // Read ssm_state_indices[bi]
            {
                LocalTensor<int32_t> ib = qSid_.AllocTensor<int32_t>();
                DataCopyExtParams cp ={1, (uint16_t)sizeof(int32_t), 0, 0, 0};
                DataCopyPad(ib, siGm_[bi], cp, {false, 0, 0, 0});
                qSid_.EnQue(ib);
                ib = qSid_.DeQue<int32_t>();
                sid = static_cast<int32_t>(ib.GetValue(0));
                qSid_.FreeTensor(ib);
            }

            if (sid < 0) {
                skipBatch = true;
            }

            if (!skipBatch) {
                // --- Step 1 UNPACK ---
                {
                    LocalTensor<bfloat16_t> mx = qMq_.AllocTensor<bfloat16_t>();
                    DataCopyExtParams cp ={1, (uint16_t)(L * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(mx, mqGm_[bi * L], cp, {false, 0, 0, 0});
                    qMq_.EnQue(mx);
                    mx = qMq_.DeQue<bfloat16_t>();
                    uint32_t qO = 0, kO = HK * DK, vO = 2 * HK * DK;
                    Cast<float, bfloat16_t>(qtB_, mx[qO], RoundMode::CAST_NONE, HK * DK);
                    Cast<float, bfloat16_t>(kB_, mx[kO], RoundMode::CAST_NONE, HK * DK);
                    Cast<float, bfloat16_t>(vB_, mx[vO], RoundMode::CAST_NONE, HV * DV);
                    qMq_.FreeTensor(mx);
                }

                // --- Step 2 L2NORM Q ---
                PipeBarrier<PIPE_V>();
                for (uint32_t h = 0; h < HK; ++h) {
                    Mul(sq_, qtB_[h * dkA], qtB_[h * dkA], DK);
                    ReduceSum<float>(sr_, sq_, rdT_, (int32_t)DK);
                    PipeBarrier<PIPE_V>();
                    Adds(sr_, sr_, ep, 1);
                    Rsqrt(sr_, sr_, 1);
                    PipeBarrier<PIPE_V>();
                    Muls(qtB_[h * dkA], qtB_[h * dkA], sr_.GetValue(0), DK);
                }
                // --- Step 2 L2NORM K ---
                for (uint32_t h = 0; h < HK; ++h) {
                    Mul(sq_, kB_[h * dkA], kB_[h * dkA], DK);
                    ReduceSum<float>(sr_, sq_, rdT_, (int32_t)DK);
                    PipeBarrier<PIPE_V>();
                    Adds(sr_, sr_, ep, 1);
                    Rsqrt(sr_, sr_, 1);
                    PipeBarrier<PIPE_V>();
                    Muls(kB_[h * dkA], kB_[h * dkA], sr_.GetValue(0), DK);
                }

                // --- Step 3 SCALE ---
                for (uint32_t h = 0; h < HK; ++h) {
                    Muls(qtB_[h * dkA], qtB_[h * dkA], scaleVal, DK);
                }

                // --- Step 4 GATE ---
                {
                    LocalTensor<float> alL = qAl_.AllocTensor<float>();
                    DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(float)), 0, 0, 0};
                    DataCopyPad(alL, alGm_, cp, {false, 0, 0, 0});
                    qAl_.EnQue(alL);
                    alL = qAl_.DeQue<float>();
                    Exp(eG_, alL, HV);
                    Muls(eG_, eG_, -1.0f, HV);
                    qAl_.FreeTensor(alL);
                }
                PipeBarrier<PIPE_V>();
                {
                    LocalTensor<bfloat16_t> aL = qA_.AllocTensor<bfloat16_t>();
                    DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(aL, aGm_[bi * HV], cp, {false, 0, 0, 0});
                    qA_.EnQue(aL);
                    aL = qA_.DeQue<bfloat16_t>();
                    Cast<float, bfloat16_t>(aT_, aL, RoundMode::CAST_NONE, HV);
                    qA_.FreeTensor(aL);
                }
                {
                    LocalTensor<float> dL = qDb_.AllocTensor<float>();
                    DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(float)), 0, 0, 0};
                    DataCopyPad(dL, dbGm_, cp, {false, 0, 0, 0});
                    qDb_.EnQue(dL);
                    dL = qDb_.DeQue<float>();
                    Add(sT_, aT_, dL, HV);
                    qDb_.FreeTensor(dL);
                }
                PipeBarrier<PIPE_V>();
                Exp(sT_, sT_, HV);
                Adds(sT_, sT_, 1.0f, HV);
                Ln(sT_, sT_, HV);
                Mul(eG_, eG_, sT_, HV);
                Exp(eG_, eG_, HV);

                // --- Step 5 BETA ---
                {
                    LocalTensor<bfloat16_t> bL = qB_.AllocTensor<bfloat16_t>();
                    DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(bL, bGm_[bi * HV], cp, {false, 0, 0, 0});
                    qB_.EnQue(bL);
                    bL = qB_.DeQue<bfloat16_t>();
                    Cast<float, bfloat16_t>(sT_, bL, RoundMode::CAST_NONE, HV);
                    qB_.FreeTensor(bL);
                }
                PipeBarrier<PIPE_V>();
                Sigmoid<float>(beB_, sT_, sgT_, HV);
                PipeBarrier<PIPE_MTE3>();
            }
        }

        // --- Skip branch (output zeros for this head) ---
        if (skipBatch) {
            LocalTensor<bfloat16_t> ow = qOut_.AllocTensor<bfloat16_t>();
            Duplicate<bfloat16_t>(ow, 0, dvA);
            qOut_.EnQue(ow);
            ow = qOut_.DeQue<bfloat16_t>();
            DataCopyExtParams cp ={1, (uint16_t)(dvA * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(outGm_[bi * HV * DV + hvi * DV], ow, cp);
            qOut_.FreeTensor(ow);
            continue;
        }

        // --- Step 6+7 RECURRENCE + OUTPUT for head (bi, hvi) ---
        {
            uint32_t hki  = (G > 0) ? (hvi / G) : hvi;
            uint32_t sOff = ((uint32_t)sid * HV + hvi) * DV * DK;
            float    ev   = eG_.GetValue(hvi);
            float    bv   = beB_.GetValue(hvi);

            // Prefetch first state chunk
            uint32_t firstV = (vStep <= DV) ? vStep : DV;
            {
                LocalTensor<S> ch = qStRd_.AllocTensor<S>();
                DataCopyExtParams cp ={1, (uint16_t)(firstV * DK * sizeof(S)), 0, 0, 0};
                DataCopyPad(ch, stGm_[sOff], cp, {false, 0, 0, 0});
                qStRd_.EnQue(ch);
            }

            for (uint32_t dvi = 0; dvi < DV; dvi += vStep) {
                uint32_t curV = (dvi + vStep <= DV) ? vStep : (DV - dvi);
                uint32_t chunkOff = dvi * DK;

                // Load prefetched state into compute buffer
                {
                    LocalTensor<S> ch = qStRd_.DeQue<S>();
                    if constexpr (std::is_same<S, float>::value) {
                        DataCopy(stB_, ch, curV * DK);
                    } else {
                        Cast<float, bfloat16_t>(stB_, ch, RoundMode::CAST_NONE, curV * DK);
                    }
                    qStRd_.FreeTensor(ch);
                }

                // Prefetch next chunk (if any)
                if (dvi + vStep < DV) {
                    uint32_t nextV = (dvi + 2 * vStep <= DV) ? vStep : (DV - dvi - vStep);
                    LocalTensor<S> ch = qStRd_.AllocTensor<S>();
                    DataCopyExtParams cp ={1, (uint16_t)(nextV * DK * sizeof(S)), 0, 0, 0};
                    DataCopyPad(ch, stGm_[sOff + (dvi + vStep) * DK], cp, {false, 0, 0, 0});
                    qStRd_.EnQue(ch);
                }

                PipeBarrier<PIPE_V>();

                // Scale state by gate
                if (ev != 1.0f) {
                    constexpr uint32_t MULS_CHUNK = 8192U;
                    uint32_t tot = curV * DK;
                    for (uint32_t o = 0; o < tot; o += MULS_CHUNK) {
                        uint32_t c = (o + MULS_CHUNK <= tot) ? MULS_CHUNK : (tot - o);
                        Muls(stB_[o], stB_[o], ev, c);
                    }
                    PipeBarrier<PIPE_V>();
                }

                // --- S@K: repeat mul + reduce ---
                {
                    uint32_t rs = dkA / DIMS_PER_BLOCK;
                    for (uint32_t k = 0; k < DK; k += FOLD_HALF) {
                        uint64_t mask = (k + FOLD_HALF <= DK) ? FOLD_HALF : (DK - k);
                        Mul(stP_[k], stB_[k], kB_[hki * dkA + k], mask, (uint64_t)curV,
                            {1, 1, 1, static_cast<uint8_t>(rs), static_cast<uint8_t>(rs), 0});
                    }
                }
                PipeBarrier<PIPE_V>();
                if (DK >= FOLD_HALF) {
                    for (uint32_t d = 0; d < curV; ++d) {
                        uint32_t ro = d * dkA;
                        uint32_t active = DK;
                        while (active > FOLD_HALF) {
                            uint32_t half = active >> 1;
                            Add(stP_[ro], stP_[ro], stP_[ro + half], half);
                            PipeBarrier<PIPE_V>();
                            active = half;
                        }
                        WholeReduceSum(oB_[d], stP_[ro], active, 1, 1, 1, DIMS_PER_BLOCK);
                    }
                } else {
                    for (uint32_t d = 0; d < curV; ++d) {
                        ReduceSum<float>(sr_, stP_[d * dkA], rdT_, (int32_t)DK);
                        PipeBarrier<PIPE_V>();
                        oB_.SetValue(d, sr_.GetValue(0));
                    }
                }
                PipeBarrier<PIPE_V>();
                Sub(oB_, vB_[hvi * dvA + dvi], oB_, curV);
                Muls(oB_, oB_, bv, curV);
                PipeBarrier<PIPE_V>();

                // --- State update: S[d] += K * delta[d] ---
                for (uint32_t d = 0; d < curV; ++d) {
                    float dv_val = oB_.GetValue(d);
                    if (dv_val != 0.0f) {
                        Muls(tr_, kB_[hki * dkA], dv_val, DK);
                        Add(tr_, stB_[d * dkA], tr_, DK);
                        DataCopy(stB_[d * dkA], tr_, DK);
                    }
                }
                PipeBarrier<PIPE_V>();

                // --- S@Q: repeat mul + reduce ---
                {
                    uint32_t rs = dkA / DIMS_PER_BLOCK;
                    for (uint32_t k = 0; k < DK; k += FOLD_HALF) {
                        uint64_t mask = (k + FOLD_HALF <= DK) ? FOLD_HALF : (DK - k);
                        Mul(stP_[k], stB_[k], qtB_[hki * dkA + k], mask, (uint64_t)curV,
                            {1, 1, 1, static_cast<uint8_t>(rs), static_cast<uint8_t>(rs), 0});
                    }
                }
                PipeBarrier<PIPE_V>();
                if (DK >= FOLD_HALF) {
                    for (uint32_t d = 0; d < curV; ++d) {
                        uint32_t ro = d * dkA;
                        uint32_t active = DK;
                        while (active > FOLD_HALF) {
                            uint32_t half = active >> 1;
                            Add(stP_[ro], stP_[ro], stP_[ro + half], half);
                            PipeBarrier<PIPE_V>();
                            active = half;
                        }
                        WholeReduceSum(oB_[d], stP_[ro], active, 1, 1, 1, DIMS_PER_BLOCK);
                    }
                } else {
                    for (uint32_t d = 0; d < curV; ++d) {
                        ReduceSum<float>(sr_, stP_[d * dkA], rdT_, (int32_t)DK);
                        PipeBarrier<PIPE_V>();
                        oB_.SetValue(d, sr_.GetValue(0));
                    }
                }
                PipeBarrier<PIPE_V>();

                // Write output
                {
                    LocalTensor<bfloat16_t> ow = qOut_.AllocTensor<bfloat16_t>();
                    Cast<bfloat16_t, float>(ow, oB_, RoundMode::CAST_RINT, curV);
                    qOut_.EnQue(ow);
                    ow = qOut_.DeQue<bfloat16_t>();
                    DataCopyExtParams cp ={1, (uint16_t)(curV * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(outGm_[bi * HV * DV + hvi * DV + dvi], ow, cp);
                    qOut_.FreeTensor(ow);
                }

                // Write state back (double-buffered)
                {
                    LocalTensor<S> ch = qStWr_.AllocTensor<S>();
                    if constexpr (std::is_same<S, float>::value) {
                        DataCopy(ch, stB_, curV * DK);
                    } else {
                        Cast(ch, stB_, RoundMode::CAST_RINT, curV * DK);
                    }
                    qStWr_.EnQue(ch);
                    ch = qStWr_.DeQue<S>();
                    DataCopyExtParams cp ={1, (uint16_t)(curV * DK * sizeof(S)), 0, 0, 0};
                    DataCopyPad(stoGm_[sOff + chunkOff], ch, cp);
                    qStWr_.FreeTensor(ch);
                }
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }

    cQ_.FreeTensor(qtB_); cK_.FreeTensor(kB_); cV_.FreeTensor(vB_); cSt_.FreeTensor(stB_); cStProd_.FreeTensor(stP_);
    cSqBuf_.FreeTensor(sq_); cScBuf_.FreeTensor(sr_); cRd_.FreeTensor(rdT_);
    cOutFp32_.FreeTensor(oB_); cTmp_.FreeTensor(tr_);
    cGateAl_.FreeTensor(alT_); cGateA_.FreeTensor(aT_); cGateS_.FreeTensor(sT_);
    cGateE_.FreeTensor(eG_); cGateSg_.FreeTensor(sgT_); cBeta_.FreeTensor(beB_);
}

} // namespace NsFusedRgd
#endif
