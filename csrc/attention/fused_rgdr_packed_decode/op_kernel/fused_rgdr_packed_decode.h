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
}


template<typename S>
__aicore__ inline void KernelFusedRgd<S>::Process() {
    if (td_.B == 0) return;

    uint32_t ci = GetBlockIdx();
    uint32_t bs = ci * td_.batchPerCore;
    uint32_t be = bs + td_.batchPerCore;
    if (be > td_.B) be = td_.B;
    if (bs >= be) return;

    uint32_t HK = td_.HK, HV = td_.HV, DK = td_.DK, DV = td_.DV;
    uint32_t dkA = td_.dkAligned, dvA = td_.dvAligned;
    uint32_t L  = td_.L_qkv, G = td_.G;
    float ep = td_.eps, scaleVal = td_.scaleVal;
    uint32_t vStep = td_.vStep;

    // --- TQue data-movement queues (non-state) ---
    TQue<QuePosition::VECIN, BUFFER_NUM>  qSid, qMq, qA, qB, qAl, qDb;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qOut;

    pipe_->InitBuffer(qSid,  BUFFER_NUM, 32U);
    pipe_->InitBuffer(qMq,   BUFFER_NUM, L * sizeof(bfloat16_t));
    pipe_->InitBuffer(qA,    BUFFER_NUM, (HV * sizeof(bfloat16_t) < MIN_BUF) ? MIN_BUF : (HV * sizeof(bfloat16_t)));
    pipe_->InitBuffer(qB,    BUFFER_NUM, (HV * sizeof(bfloat16_t) < MIN_BUF) ? MIN_BUF : (HV * sizeof(bfloat16_t)));
    pipe_->InitBuffer(qAl,   BUFFER_NUM, (HV * sizeof(float)      < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(qDb,   BUFFER_NUM, (HV * sizeof(float)      < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(qOut,  BUFFER_NUM, (dvA * sizeof(bfloat16_t) < MIN_BUF) ? MIN_BUF : (dvA * sizeof(bfloat16_t)));

    // --- State double-buffered queues ---
    uint32_t stChunkBytes = vStep * dkA * sizeof(S);
    TQue<QuePosition::VECIN, ST_BUF_NUM>  qStRd;
    TQue<QuePosition::VECOUT, ST_BUF_NUM> qStWr;
    pipe_->InitBuffer(qStRd, ST_BUF_NUM, stChunkBytes);
    pipe_->InitBuffer(qStWr, ST_BUF_NUM, stChunkBytes);

    // --- Compute buffers (VECCALC) ---
    TBuf<TPosition::VECCALC> cQ, cK, cV, cSt, cStProd, cRd, cOutFp32, cTmp;
    TBuf<TPosition::VECCALC> cGateAl, cGateA, cGateS, cGateE, cGateSg, cBeta;
    TBuf<TPosition::VECCALC> cSqBuf, cScBuf;

    pipe_->InitBuffer(cQ,     HK * dkA * sizeof(float));
    pipe_->InitBuffer(cK,     HK * dkA * sizeof(float));
    pipe_->InitBuffer(cV,     HV * dvA * sizeof(float));
    pipe_->InitBuffer(cSt,    vStep * dkA * sizeof(float));
    pipe_->InitBuffer(cStProd, vStep * dkA * sizeof(float));  // product buffer for repeat-based MatVecMul
    pipe_->InitBuffer(cRd,    REDUCE_TMP_BYTES);
    pipe_->InitBuffer(cOutFp32, dvA * sizeof(float));
    pipe_->InitBuffer(cSqBuf, dkA * sizeof(float));
    pipe_->InitBuffer(cScBuf, 32U);
    pipe_->InitBuffer(cTmp,   dkA * sizeof(float));
    pipe_->InitBuffer(cGateAl, (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateA,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateS,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateE,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));
    pipe_->InitBuffer(cGateSg, SIGMOID_TMP_BYTES);
    pipe_->InitBuffer(cBeta,  (HV * sizeof(float) < MIN_BUF) ? MIN_BUF : (HV * sizeof(float)));

    LocalTensor<float> qtB = cQ.Get<float>(HK * dkA);
    LocalTensor<float> kB  = cK.Get<float>(HK * dkA);
    LocalTensor<float> vB  = cV.Get<float>(HV * dvA);
    LocalTensor<float> stB = cSt.Get<float>(vStep * dkA);
    LocalTensor<float> stP = cStProd.Get<float>(vStep * dkA);  // product buffer
    LocalTensor<float> sq  = cSqBuf.Get<float>(dkA);
    LocalTensor<float> sr  = cScBuf.Get<float>(8);
    LocalTensor<float> rdT = cRd.Get<float>(REDUCE_TMP_BYTES / sizeof(float));
    LocalTensor<float> oB  = cOutFp32.Get<float>(dvA);
    LocalTensor<float> tr  = cTmp.Get<float>(dkA);

    LocalTensor<float>    alT = cGateAl.Get<float>(HV);
    LocalTensor<float>    aT  = cGateA.Get<float>(HV);
    LocalTensor<float>    sT  = cGateS.Get<float>(HV);
    LocalTensor<float>    eG  = cGateE.Get<float>(HV);
    LocalTensor<uint8_t>  sgT = cGateSg.Get<uint8_t>(SIGMOID_TMP_BYTES);
    LocalTensor<float>    beB = cBeta.Get<float>(HV);

    for (uint32_t bi = bs; bi < be; ++bi) {
        // --- Read ssm_state_indices[bi] ---
        int32_t sid;
        {
            LocalTensor<int32_t> ib = qSid.AllocTensor<int32_t>();
            DataCopyExtParams cp ={1, (uint16_t)sizeof(int32_t), 0, 0, 0};
            DataCopyPad(ib, siGm_[bi], cp, {false, 0, 0, 0});
            qSid.EnQue(ib);
            ib = qSid.DeQue<int32_t>();
            sid = ib.GetValue(0);
            qSid.FreeTensor(ib);
        }

        // --- Skip branch (output zeros, no state write) ---
        if (sid < 0) {
            for (uint32_t hvi = 0; hvi < HV; ++hvi) {
                LocalTensor<bfloat16_t> ow = qOut.AllocTensor<bfloat16_t>();
                Duplicate<bfloat16_t>(ow, 0, dvA);
                qOut.EnQue(ow);
                ow = qOut.DeQue<bfloat16_t>();
                DataCopyExtParams cp ={1, (uint16_t)(dvA * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outGm_[bi * HV * DV + hvi * DV], ow, cp);
                qOut.FreeTensor(ow);
            }
            continue;
        }

        // --- Step 1 UNPACK ---
        {
            LocalTensor<bfloat16_t> mx = qMq.AllocTensor<bfloat16_t>();
            DataCopyExtParams cp ={1, (uint16_t)(L * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(mx, mqGm_[bi * L], cp, {false, 0, 0, 0});
            qMq.EnQue(mx);
            mx = qMq.DeQue<bfloat16_t>();
            uint32_t qO = 0, kO = HK * DK, vO = 2 * HK * DK;
            Cast<float, bfloat16_t>(qtB, mx[qO], RoundMode::CAST_NONE, HK * DK);
            Cast<float, bfloat16_t>(kB, mx[kO], RoundMode::CAST_NONE, HK * DK);
            Cast<float, bfloat16_t>(vB, mx[vO], RoundMode::CAST_NONE, HV * DV);
            qMq.FreeTensor(mx);
        }

        // --- Step 2 L2NORM Q ---
        PipeBarrier<PIPE_V>();
        for (uint32_t h = 0; h < HK; ++h) {
            Mul(sq, qtB[h * dkA], qtB[h * dkA], DK);
            ReduceSum<float>(sr, sq, rdT, (int32_t)DK);
            PipeBarrier<PIPE_V>();
            Adds(sr, sr, ep, 1);
            Rsqrt(sr, sr, 1);
            PipeBarrier<PIPE_V>();
            Muls(qtB[h * dkA], qtB[h * dkA], sr.GetValue(0), DK);
        }
        // --- Step 2 L2NORM K ---
        for (uint32_t h = 0; h < HK; ++h) {
            Mul(sq, kB[h * dkA], kB[h * dkA], DK);
            ReduceSum<float>(sr, sq, rdT, (int32_t)DK);
            PipeBarrier<PIPE_V>();
            Adds(sr, sr, ep, 1);
            Rsqrt(sr, sr, 1);
            PipeBarrier<PIPE_V>();
            Muls(kB[h * dkA], kB[h * dkA], sr.GetValue(0), DK);
        }

        // --- Step 3 SCALE ---
        for (uint32_t h = 0; h < HK; ++h) {
            Muls(qtB[h * dkA], qtB[h * dkA], scaleVal, DK);
        }

        // --- Step 4 GATE ---
        {
            LocalTensor<float> alL = qAl.AllocTensor<float>();
            DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(float)), 0, 0, 0};
            DataCopyPad(alL, alGm_, cp, {false, 0, 0, 0});
            qAl.EnQue(alL);
            alL = qAl.DeQue<float>();
            Exp(eG, alL, HV);
            Muls(eG, eG, -1.0f, HV);
            qAl.FreeTensor(alL);
        }
        PipeBarrier<PIPE_V>();
        {
            LocalTensor<bfloat16_t> aL = qA.AllocTensor<bfloat16_t>();
            DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(aL, aGm_[bi * HV], cp, {false, 0, 0, 0});
            qA.EnQue(aL);
            aL = qA.DeQue<bfloat16_t>();
            Cast<float, bfloat16_t>(aT, aL, RoundMode::CAST_NONE, HV);
            qA.FreeTensor(aL);
        }
        {
            LocalTensor<float> dL = qDb.AllocTensor<float>();
            DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(float)), 0, 0, 0};
            DataCopyPad(dL, dbGm_, cp, {false, 0, 0, 0});
            qDb.EnQue(dL);
            dL = qDb.DeQue<float>();
            Add(sT, aT, dL, HV);
            qDb.FreeTensor(dL);
        }
        PipeBarrier<PIPE_V>();
        Exp(sT, sT, HV);
        Adds(sT, sT, 1.0f, HV);
        Ln(sT, sT, HV);
        Mul(eG, eG, sT, HV);
        Exp(eG, eG, HV);

        // --- Step 5 BETA ---
        {
            LocalTensor<bfloat16_t> bL = qB.AllocTensor<bfloat16_t>();
            DataCopyExtParams cp ={1, (uint16_t)(HV * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(bL, bGm_[bi * HV], cp, {false, 0, 0, 0});
            qB.EnQue(bL);
            bL = qB.DeQue<bfloat16_t>();
            Cast<float, bfloat16_t>(sT, bL, RoundMode::CAST_NONE, HV);
            qB.FreeTensor(bL);
        }
        PipeBarrier<PIPE_V>();
        Sigmoid<float>(beB, sT, sgT, HV);
        PipeBarrier<PIPE_MTE3>();

        // --- Step 6+7 RECURRENCE + OUTPUT (vStep-tiled, double-buffered) ---
        for (uint32_t hvi = 0; hvi < HV; ++hvi) {
            uint32_t hki  = (G > 0) ? (hvi / G) : hvi;
            uint32_t sOff = ((uint32_t)sid * HV + hvi) * DV * DK;
            float    ev   = eG.GetValue(hvi);
            float    bv   = beB.GetValue(hvi);

            // Prefetch first state chunk
            uint32_t firstV = (vStep <= DV) ? vStep : DV;
            {
                LocalTensor<S> ch = qStRd.AllocTensor<S>();
                DataCopyExtParams cp ={1, (uint16_t)(firstV * DK * sizeof(S)), 0, 0, 0};
                DataCopyPad(ch, stGm_[sOff], cp, {false, 0, 0, 0});
                qStRd.EnQue(ch);
            }

            for (uint32_t dvi = 0; dvi < DV; dvi += vStep) {
                uint32_t curV = (dvi + vStep <= DV) ? vStep : (DV - dvi);
                uint32_t chunkOff = dvi * DK;

                // Load prefetched state into compute buffer
                {
                    LocalTensor<S> ch = qStRd.DeQue<S>();
                    if constexpr (std::is_same<S, float>::value) {
                        DataCopy(stB, ch, curV * DK);
                    } else {
                        Cast<float, bfloat16_t>(stB, ch, RoundMode::CAST_NONE, curV * DK);
                    }
                    qStRd.FreeTensor(ch);
                }

                // Prefetch next chunk (if any)
                if (dvi + vStep < DV) {
                    uint32_t nextV = (dvi + 2 * vStep <= DV) ? vStep : (DV - dvi - vStep);
                    LocalTensor<S> ch = qStRd.AllocTensor<S>();
                    DataCopyExtParams cp ={1, (uint16_t)(nextV * DK * sizeof(S)), 0, 0, 0};
                    DataCopyPad(ch, stGm_[sOff + (dvi + vStep) * DK], cp, {false, 0, 0, 0});
                    qStRd.EnQue(ch);
                }

                PipeBarrier<PIPE_V>();

                // Scale state by gate
                if (ev != 1.0f) {
                    constexpr uint32_t MULS_CHUNK = 8192U;
                    uint32_t tot = curV * DK;
                    for (uint32_t o = 0; o < tot; o += MULS_CHUNK) {
                        uint32_t c = (o + MULS_CHUNK <= tot) ? MULS_CHUNK : (tot - o);
                        Muls(stB[o], stB[o], ev, c);
                    }
                    PipeBarrier<PIPE_V>();
                }

                // --- S@K: repeat mul + reduce ---
                {
                    uint32_t rs = dkA / DIMS_PER_BLOCK;
                    for (uint32_t k = 0; k < DK; k += FOLD_HALF) {
                        uint64_t mask = (k + FOLD_HALF <= DK) ? FOLD_HALF : (DK - k);
                        Mul(stP[k], stB[k], kB[hki * dkA + k], mask, (uint64_t)curV,
                            {1, 1, 1, rs, rs, 0});
                    }
                }
                PipeBarrier<PIPE_V>();
                if (DK >= FOLD_HALF) {
                    for (uint32_t d = 0; d < curV; ++d) {
                        uint32_t ro = d * dkA;
                        uint32_t active = DK;
                        while (active > FOLD_HALF) {
                            uint32_t half = active >> 1;
                            Add(stP[ro], stP[ro], stP[ro + half], half);
                            PipeBarrier<PIPE_V>();
                            active = half;
                        }
                        WholeReduceSum(oB[d], stP[ro], active, 1, 1, 1, DIMS_PER_BLOCK);
                    }
                } else {
                    for (uint32_t d = 0; d < curV; ++d) {
                        ReduceSum<float>(sr, stP[d * dkA], rdT, (int32_t)DK);
                        PipeBarrier<PIPE_V>();
                        oB.SetValue(d, sr.GetValue(0));
                    }
                }
                PipeBarrier<PIPE_V>();
                Sub(oB, vB[hvi * dvA + dvi], oB, curV);
                Muls(oB, oB, bv, curV);
                PipeBarrier<PIPE_V>();

                // --- State update: S[d] += K * delta[d] ---
                for (uint32_t d = 0; d < curV; ++d) {
                    float dv_val = oB.GetValue(d);
                    if (dv_val != 0.0f) {
                        Muls(tr, kB[hki * dkA], dv_val, DK);
                        Add(tr, stB[d * dkA], tr, DK);
                        DataCopy(stB[d * dkA], tr, DK);
                    }
                }
                PipeBarrier<PIPE_V>();

                // --- S@Q: repeat mul + reduce ---
                {
                    uint32_t rs = dkA / DIMS_PER_BLOCK;
                    for (uint32_t k = 0; k < DK; k += FOLD_HALF) {
                        uint64_t mask = (k + FOLD_HALF <= DK) ? FOLD_HALF : (DK - k);
                        Mul(stP[k], stB[k], qtB[hki * dkA + k], mask, (uint64_t)curV,
                            {1, 1, 1, rs, rs, 0});
                    }
                }
                PipeBarrier<PIPE_V>();
                if (DK >= FOLD_HALF) {
                    for (uint32_t d = 0; d < curV; ++d) {
                        uint32_t ro = d * dkA;
                        uint32_t active = DK;
                        while (active > FOLD_HALF) {
                            uint32_t half = active >> 1;
                            Add(stP[ro], stP[ro], stP[ro + half], half);
                            PipeBarrier<PIPE_V>();
                            active = half;
                        }
                        WholeReduceSum(oB[d], stP[ro], active, 1, 1, 1, DIMS_PER_BLOCK);
                    }
                } else {
                    for (uint32_t d = 0; d < curV; ++d) {
                        ReduceSum<float>(sr, stP[d * dkA], rdT, (int32_t)DK);
                        PipeBarrier<PIPE_V>();
                        oB.SetValue(d, sr.GetValue(0));
                    }
                }
                PipeBarrier<PIPE_V>();

                // Write output
                {
                    LocalTensor<bfloat16_t> ow = qOut.AllocTensor<bfloat16_t>();
                    Cast<bfloat16_t, float>(ow, oB, RoundMode::CAST_RINT, curV);
                    qOut.EnQue(ow);
                    ow = qOut.DeQue<bfloat16_t>();
                    DataCopyExtParams cp ={1, (uint16_t)(curV * sizeof(bfloat16_t)), 0, 0, 0};
                    DataCopyPad(outGm_[bi * HV * DV + hvi * DV + dvi], ow, cp);
                    qOut.FreeTensor(ow);
                }

                // Write state back (double-buffered)
                {
                    LocalTensor<S> ch = qStWr.AllocTensor<S>();
                    if constexpr (std::is_same<S, float>::value) {
                        DataCopy(ch, stB, curV * DK);
                    } else {
                        Cast(ch, stB, RoundMode::CAST_RINT, curV * DK);
                    }
                    qStWr.EnQue(ch);
                    ch = qStWr.DeQue<S>();
                    DataCopyExtParams cp ={1, (uint16_t)(curV * DK * sizeof(S)), 0, 0, 0};
                    DataCopyPad(stoGm_[sOff + chunkOff], ch, cp);
                    qStWr.FreeTensor(ch);
                }
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }

    cQ.FreeTensor(qtB); cK.FreeTensor(kB); cV.FreeTensor(vB); cSt.FreeTensor(stB); cStProd.FreeTensor(stP);
    cSqBuf.FreeTensor(sq); cScBuf.FreeTensor(sr); cRd.FreeTensor(rdT);
    cOutFp32.FreeTensor(oB); cTmp.FreeTensor(tr);
    cGateAl.FreeTensor(alT); cGateA.FreeTensor(aT); cGateS.FreeTensor(sT);
    cGateE.FreeTensor(eG); cGateSg.FreeTensor(sgT); cBeta.FreeTensor(beB);
}

} // namespace NsFusedRgd
#endif
