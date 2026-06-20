/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 *
 * Fused RGDR Packed Decode — Decode-phase kernel for Gated Delta Rule.
 * Reads packed mixed_qkv, L2-norms Q/K, applies scale/gate/beta,
 * performs recurrent state update, writes output and state.
 *
 * Data movement: TQue<QuePosition::VECIN/VECOUT, 1> for GM<->UB
 * Compute: TBuf<TPosition::VECCALC> for UB scratch
 * Type conversion: Cast (bf16<->fp32), vector operations
 */

#ifndef FUSED_RGDR_PACKED_DECODE_KERNEL_H_
#define FUSED_RGDR_PACKED_DECODE_KERNEL_H_

#include "kernel_operator.h"
#include "fused_rgdr_packed_decode_tiling_data.h"

namespace NsFusedRgd {
using namespace AscendC;

constexpr uint32_t ST_CHUNK_ELEMS = 128U;
constexpr uint32_t ST_CHUNK_BYTES = ST_CHUNK_ELEMS * sizeof(float);
constexpr uint32_t REDUCE_TMP_BYTES = 512U;
constexpr uint32_t SIGMOID_TMP_BYTES = 256U;
constexpr uint32_t BUFFER_NUM = 1;

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

    // --- TQue data-movement queues ---
    TQue<QuePosition::VECIN, BUFFER_NUM>  qSid, qMq, qA, qB, qAl, qDb, qStRd;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qStWr, qOut;

    pipe_->InitBuffer(qSid,  BUFFER_NUM, 32U);
    pipe_->InitBuffer(qMq,   BUFFER_NUM, L * sizeof(bfloat16_t));
    pipe_->InitBuffer(qA,    BUFFER_NUM, HV * sizeof(bfloat16_t));
    pipe_->InitBuffer(qB,    BUFFER_NUM, HV * sizeof(bfloat16_t));
    pipe_->InitBuffer(qAl,   BUFFER_NUM, HV * sizeof(float));
    pipe_->InitBuffer(qDb,   BUFFER_NUM, HV * sizeof(float));
    pipe_->InitBuffer(qStRd, BUFFER_NUM, ST_CHUNK_BYTES);
    pipe_->InitBuffer(qStWr, BUFFER_NUM, ST_CHUNK_BYTES);
    pipe_->InitBuffer(qOut,  BUFFER_NUM, dvA * sizeof(bfloat16_t));

    // --- Compute buffers (VECCALC) ---
    TBuf<TPosition::VECCALC> cQ, cK, cV, cSt, cRd, cOutFp32, cTmp, cGate, cBeta;
    TBuf<TPosition::VECCALC> cSqBuf, cScBuf;

    pipe_->InitBuffer(cQ,     HK * dkA * sizeof(float));
    pipe_->InitBuffer(cK,     HK * dkA * sizeof(float));
    pipe_->InitBuffer(cV,     HV * dvA * sizeof(float));
    pipe_->InitBuffer(cSt,    dvA * dkA * sizeof(float));
    pipe_->InitBuffer(cRd,    REDUCE_TMP_BYTES);
    pipe_->InitBuffer(cOutFp32, dvA * sizeof(float));
    pipe_->InitBuffer(cSqBuf, dkA * sizeof(float));
    pipe_->InitBuffer(cScBuf, 8 * sizeof(float));
    pipe_->InitBuffer(cTmp,   dkA * sizeof(float));
    pipe_->InitBuffer(cGate,  4 * HV * sizeof(float) + SIGMOID_TMP_BYTES);
    pipe_->InitBuffer(cBeta,  HV * sizeof(float));

    LocalTensor<float> qtB = cQ.Get<float>(HK * dkA);
    LocalTensor<float> kB  = cK.Get<float>(HK * dkA);
    LocalTensor<float> vB  = cV.Get<float>(HV * dvA);
    LocalTensor<float> stB = cSt.Get<float>(dvA * dkA);
    LocalTensor<float> sq  = cSqBuf.Get<float>(dkA);
    LocalTensor<float> sr  = cScBuf.Get<float>(8);
    LocalTensor<float> rdT = cRd.Get<float>(REDUCE_TMP_BYTES / sizeof(float));
    LocalTensor<float> oB  = cOutFp32.Get<float>(dvA);
    LocalTensor<float> tr  = cTmp.Get<float>(dkA);

    LocalTensor<float>    alT = cGate.Get<float>(HV);
    LocalTensor<float>    aT  = cGate.GetWithOffset<float>(1 * HV * sizeof(float), HV);
    LocalTensor<float>    sT  = cGate.GetWithOffset<float>(2 * HV * sizeof(float), HV);
    LocalTensor<float>    eG  = cGate.GetWithOffset<float>(3 * HV * sizeof(float), HV);
    LocalTensor<uint8_t>  sgT = cGate.GetWithOffset<uint8_t>(4 * HV * sizeof(float), SIGMOID_TMP_BYTES);
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

        // --- Skip branch ---
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
            Duplicate<float>(stB, 0.0f, DV * DK);
            PipeBarrier<PIPE_V>();
            for (uint32_t hvi = 0; hvi < HV; ++hvi) {
                uint32_t sOff = ((uint32_t)0 * HV + hvi) * DV * DK;
                for (uint32_t o = 0; o < DV * DK; o += ST_CHUNK_ELEMS) {
                    uint32_t c = (o + ST_CHUNK_ELEMS <= DV * DK) ? ST_CHUNK_ELEMS : (DV * DK - o);
                    LocalTensor<S> ch = qStWr.AllocTensor<S>();
                    if constexpr (std::is_same<S, float>::value) {
                        DataCopy(ch, stB[o], c);
                    } else {
                        Cast(ch, stB[o], RoundMode::CAST_RINT, c);
                    }
                    qStWr.EnQue(ch);
                    ch = qStWr.DeQue<S>();
                    DataCopyExtParams cp ={1, (uint16_t)(c * sizeof(S)), 0, 0, 0};
                    DataCopyPad(stoGm_[sOff + o], ch, cp);
                    qStWr.FreeTensor(ch);
                }
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
            DataCopy(alT, alL, HV);
            qAl.FreeTensor(alL);
        }
        PipeBarrier<PIPE_V>();
        Exp(eG, alT, HV);
        Muls(eG, eG, -1.0f, HV);
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

        // --- Step 6+7 RECURRENCE + OUTPUT ---
        PipeBarrier<PIPE_ALL>();
        for (uint32_t hvi = 0; hvi < HV; ++hvi) {
            uint32_t hki  = (G > 0) ? (hvi / G) : hvi;
            uint32_t sOff = ((uint32_t)sid * HV + hvi) * DV * DK;
            uint32_t tot  = DV * DK;

            for (uint32_t o = 0; o < tot; o += ST_CHUNK_ELEMS) {
                uint32_t c = (o + ST_CHUNK_ELEMS <= tot) ? ST_CHUNK_ELEMS : (tot - o);
                LocalTensor<S> ch = qStRd.AllocTensor<S>();
                DataCopyExtParams cp ={1, (uint16_t)(c * sizeof(S)), 0, 0, 0};
                DataCopyPad(ch, stGm_[sOff + o], cp, {false, 0, 0, 0});
                qStRd.EnQue(ch);
                ch = qStRd.DeQue<S>();
                if constexpr (std::is_same<S, float>::value) {
                    DataCopy(stB[o], ch, c);
                } else {
                    Cast<float, bfloat16_t>(stB[o], ch, RoundMode::CAST_NONE, c);
                }
                qStRd.FreeTensor(ch);
            }
            PipeBarrier<PIPE_V>();

            {
                float ev = eG.GetValue(hvi);
                if (ev != 1.0f) {
                    constexpr uint32_t MULS_CHUNK = 8192U;
                    for (uint32_t o = 0; o < tot; o += MULS_CHUNK) {
                        uint32_t c = (o + MULS_CHUNK <= tot) ? MULS_CHUNK : (tot - o);
                        Muls(stB[o], stB[o], ev, c);
                    }
                }
                PipeBarrier<PIPE_V>();
            }

            for (uint32_t dvi = 0; dvi < DV; ++dvi) {
                Mul(tr, stB[dvi * dkA], kB[hki * dkA], DK);
                ReduceSum<float>(sr, tr, rdT, (int32_t)DK);
                PipeBarrier<PIPE_V>();
                oB.SetValue(dvi, sr.GetValue(0));
            }
            Sub(oB, vB[hvi * dvA], oB, DV);
            {
                float bv = beB.GetValue(hvi);
                Muls(oB, oB, bv, DV);
            }
            PipeBarrier<PIPE_V>();

            for (uint32_t dvi = 0; dvi < DV; ++dvi) {
                float dv = oB.GetValue(dvi);
                if (dv != 0.0f) {
                    Muls(tr, kB[hki * dkA], dv, DK);
                    Add(stB[dvi * dkA], stB[dvi * dkA], tr, DK);
                }
            }
            PipeBarrier<PIPE_V>();

            for (uint32_t dvi = 0; dvi < DV; ++dvi) {
                Mul(tr, stB[dvi * dkA], qtB[hki * dkA], DK);
                ReduceSum<float>(sr, tr, rdT, (int32_t)DK);
                PipeBarrier<PIPE_V>();
                oB.SetValue(dvi, sr.GetValue(0));
            }
            PipeBarrier<PIPE_V>();

            {
                LocalTensor<bfloat16_t> ow = qOut.AllocTensor<bfloat16_t>();
                Cast<bfloat16_t, float>(ow, oB, RoundMode::CAST_RINT, DV);
                qOut.EnQue(ow);
                ow = qOut.DeQue<bfloat16_t>();
                DataCopyExtParams cp ={1, (uint16_t)(DV * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outGm_[bi * HV * DV + hvi * DV], ow, cp);
                qOut.FreeTensor(ow);
            }

            for (uint32_t o = 0; o < tot; o += ST_CHUNK_ELEMS) {
                uint32_t c = (o + ST_CHUNK_ELEMS <= tot) ? ST_CHUNK_ELEMS : (tot - o);
                LocalTensor<S> ch = qStWr.AllocTensor<S>();
                if constexpr (std::is_same<S, float>::value) {
                    DataCopy(ch, stB[o], c);
                } else {
                    Cast(ch, stB[o], RoundMode::CAST_RINT, c);
                }
                qStWr.EnQue(ch);
                ch = qStWr.DeQue<S>();
                DataCopyExtParams cp ={1, (uint16_t)(c * sizeof(S)), 0, 0, 0};
                DataCopyPad(stoGm_[sOff + o], ch, cp);
                qStWr.FreeTensor(ch);
            }
            PipeBarrier<PIPE_MTE3>();
        }
    }

    cQ.FreeTensor(qtB); cK.FreeTensor(kB); cV.FreeTensor(vB); cSt.FreeTensor(stB);
    cSqBuf.FreeTensor(sq); cScBuf.FreeTensor(sr); cRd.FreeTensor(rdT);
    cOutFp32.FreeTensor(oB); cTmp.FreeTensor(tr);
    cGate.FreeTensor(alT); cGate.FreeTensor(aT); cGate.FreeTensor(sT);
    cGate.FreeTensor(eG); cGate.FreeTensor(sgT); cBeta.FreeTensor(beB);
}

} // namespace NsFusedRgd
#endif
