/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_KERNEL_H
#define FUSED_PACKED_RECURRENT_GATED_DELTA_RULE_KERNEL_H

#include "kernel_operator.h"
#include "fused_packed_recurrent_gated_delta_rule_tiling_data.h"

namespace FusedPackedRecurrentGatedDeltaRule {

using namespace AscendC;

// 32-byte alignment: 16 bf16 elements, 8 fp32 elements, or 1 DMA block on NPU.
constexpr uint32_t BF16_PER_BLK = 16;
constexpr uint32_t FP32_PER_BLK = 8;
constexpr uint32_t RLEN = 64;
constexpr uint32_t MXR  = 255;
constexpr float    EPS  = 1e-6f;
constexpr float    SPTH = 20.0f;

template <typename T> __aicore__ inline T CD(T a, T b) { return (a + b - 1) / b; }
template <typename T> __aicore__ inline T AU(T a, T b) { return CD(a, b) * b; }

template <typename inType, typename stateType>
class FPRGDR {
public:
    __aicore__ inline FPRGDR(const FusedPackedRecurrentGatedDeltaRuleTilingData *td)
    {
        B_=td->b; H_=td->h; HV_=td->hv; DK_=td->dk; DV_=td->dv;
        qD_=td->qkvDim; sN_=td->sBlockNum; sc_=td->scale;

        // Compute strides: aligned to 32-byte for fp32 vector ops.
        CK_ = AU(DK_, FP32_PER_BLK);
        CV_ = AU(DV_, FP32_PER_BLK);

        // DMA padding deltas (in elements).
        // bf16 DMA needs 16-element (32-byte) alignment on UB.
        // fp32 DMA needs 8-element (32-byte) alignment on UB.
        padK_bf16_ = static_cast<uint8_t>(AU(DK_, BF16_PER_BLK) - DK_);
        padV_bf16_ = static_cast<uint8_t>(AU(DV_, BF16_PER_BLK) - DV_);
        padK_fp32_ = static_cast<uint8_t>(AU(DK_, FP32_PER_BLK) - DK_);

        // Aligned sizes for DMA.
        bufK_bf16_ = AU(DK_, BF16_PER_BLK);  // Q/K DMA buffer (bf16)
        bufV_bf16_ = AU(DV_, BF16_PER_BLK);  // V/O DMA buffer (bf16)
        bufK_fp32_ = AU(DK_, FP32_PER_BLK);  // state per-row stride (fp32)
    }

    __aicore__ inline void Init(GM_ADDR mq, GM_ADDR ag, GM_ADDR bg, GM_ADDR alG, GM_ADDR dtG,
                                 GM_ADDR sG, GM_ADDR ssi, GM_ADDR oG,
                                 const FusedPackedRecurrentGatedDeltaRuleTilingData *td,
                                 TPipe *pipe)
    {
        pipe_ = pipe;
        mqGm_.SetGlobalBuffer((__gm__ inType*)mq, (uint64_t)B_*qD_);
        aGm_.SetGlobalBuffer((__gm__ inType*)ag, (uint64_t)B_*HV_);
        bGm_.SetGlobalBuffer((__gm__ inType*)bg, (uint64_t)B_*HV_);
        alGm_.SetGlobalBuffer((__gm__ float*)alG, HV_);
        dtGm_.SetGlobalBuffer((__gm__ float*)dtG, HV_);
        sGm_.SetGlobalBuffer((__gm__ stateType*)sG, (uint64_t)sN_*HV_*DV_*DK_);
        ssiGm_.SetGlobalBuffer((__gm__ int32_t*)ssi, B_);
        oGm_.SetGlobalBuffer((__gm__ inType*)oG, (uint64_t)B_*HV_*DV_);

        // Compute buffers (TBuf for fp32 compute). Sized with compute strides.
        pipe_->InitBuffer(qBuf_, CK_ * sizeof(float));
        qUb_ = qBuf_.Get<float>();

        pipe_->InitBuffer(kBuf_, CK_ * sizeof(float));
        kUb_ = kBuf_.Get<float>();

        pipe_->InitBuffer(vBuf_, CV_ * sizeof(float));
        vUb_ = vBuf_.Get<float>();

        pipe_->InitBuffer(sBuf_, CV_ * CK_ * sizeof(float));
        sUb_ = sBuf_.Get<float>();

        pipe_->InitBuffer(dBuf_, CV_ * sizeof(float));
        dUb_ = dBuf_.Get<float>();

        pipe_->InitBuffer(oBuf_, CV_ * sizeof(float));
        oUb_ = oBuf_.Get<float>();

        pipe_->InitBuffer(wBuf_, CV_ * CK_ * sizeof(float));
        wUb_ = wBuf_.Get<float>();

        // DMA Queues (TQue, depth=1) for GM↔UB transfers.
        // Sized with 32-byte alignment: bf16 use BF16_PER_BLK, state uses compute stride.
        pipe_->InitBuffer(qInQ_,  1, bufK_bf16_ * sizeof(inType));
        pipe_->InitBuffer(kInQ_,  1, bufK_bf16_ * sizeof(inType));
        pipe_->InitBuffer(vInQ_,  1, bufV_bf16_ * sizeof(inType));
        pipe_->InitBuffer(sInQ_,  1, DV_ * bufK_fp32_ * sizeof(stateType));
        pipe_->InitBuffer(sOutQ_, 1, DV_ * bufK_fp32_ * sizeof(stateType));
        pipe_->InitBuffer(oOutQ_, 1, bufV_bf16_ * sizeof(inType));

        // Constants: preload A_log, dt_bias (HV fp32 elements).
        // Use a separate TBuf with aligned size.
        uint32_t hvAligned = AU(HV_, FP32_PER_BLK);
        pipe_->InitBuffer(alBuf_,  hvAligned * sizeof(float));
        alUb_ = alBuf_.Get<float>();
        pipe_->InitBuffer(dtBuf_,  hvAligned * sizeof(float));
        dtUb_ = dtBuf_.Get<float>();

        DataCopyExtParams cp{1, (uint32_t)(HV_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
        DataCopyPad(alUb_, alGm_, cp, pp);
        DataCopyPad(dtUb_, dtGm_, cp, pp);
        PipeBarrier<PIPE_V>();
        Exp(alUb_, alUb_, HV_);
        PipeBarrier<PIPE_V>();
        Muls(alUb_, alUb_, -1.0f, HV_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        uint32_t bid = GetBlockIdx(), bn = GetBlockNum();
        if (bn == 0) bn = 1;
        uint64_t tot = (uint64_t)B_ * HV_, pb = CD<uint64_t>(tot, bn);
        uint64_t st = bid * pb, ed = st + pb;
        if (ed > tot) ed = tot;
        for (uint64_t t = st; t < ed; ++t) {
            uint32_t bi = (uint32_t)(t / HV_), hvi = (uint32_t)(t % HV_);
            int32_t sid = ssiGm_.GetValue(bi);
            if (sid <= 0) continue;
            DoOne(bi, hvi, (uint32_t)sid);
        }
    }

private:
    __aicore__ inline void DoOne(uint32_t bi, uint32_t hvi, uint32_t sid)
    {
        // 1. DMA Q/K/V in via TQue (GM→UB, bf16 with 32B alignment).
        ReadQkv(bi, hvi);

        // 2. L2Norm + scale Q/K (fp32 compute).
        L2Norm(qUb_, CK_, DK_);
        L2Norm(kUb_, CK_, DK_);
        Muls(qUb_, qUb_, sc_, DK_);

        // 3. Gating scalars. Use DMA+Cast for bf16→fp32;
        // bisheng backend does NOT support (float) C-style scalar cast on bf16.
        float al = alUb_.GetValue(hvi);
        float dv = dtUb_.GetValue(hvi);
        float av, bv, gv, btv;
        Gate(bi, hvi, al, dv, av, bv, gv, btv);

        // 4. DMA state in via TQue.
        LoadState(sid, hvi);

        // 5. Recurrence.
        Recur(gv, btv);

        // 6. DMA state + output out via TQue.
        StoreState(sid, hvi);
        StoreOut(bi, hvi);
    }

    // ============ DMA via TQue (all GM↔UB use DataCopyPad) ============

    __aicore__ inline void ReadQkv(uint32_t bi, uint32_t hvi)
    {
        uint64_t rb = (uint64_t)bi * qD_;
        uint64_t qo = rb + (uint64_t)hvi / (HV_ / H_) * DK_;
        uint64_t ko = rb + (uint64_t)H_ * DK_ + (uint64_t)hvi / (HV_ / H_) * DK_;
        uint64_t vo = rb + (uint64_t)(2 * H_) * DK_ + (uint64_t)hvi * DV_;

        // Q.
        {
            LocalTensor<inType> ql = qInQ_.template AllocTensor<inType>();
            DataCopyExtParams qp{1, (uint32_t)(DK_ * sizeof(inType)), 0, 0, 0};
            DataCopyPadExtParams<inType> qpp{true, 0, padK_bf16_, 0};
            DataCopyPad(ql, mqGm_[qo], qp, qpp);
            qInQ_.template EnQue<inType>(ql);
            ql = qInQ_.template DeQue<inType>();
            Cast(qUb_, ql, RoundMode::CAST_NONE, DK_);
            qInQ_.template FreeTensor(ql);
        }
        PipeBarrier<PIPE_V>();

        // K.
        {
            LocalTensor<inType> kl = kInQ_.template AllocTensor<inType>();
            DataCopyExtParams kp{1, (uint32_t)(DK_ * sizeof(inType)), 0, 0, 0};
            DataCopyPadExtParams<inType> kpp{true, 0, padK_bf16_, 0};
            DataCopyPad(kl, mqGm_[ko], kp, kpp);
            kInQ_.template EnQue<inType>(kl);
            kl = kInQ_.template DeQue<inType>();
            Cast(kUb_, kl, RoundMode::CAST_NONE, DK_);
            kInQ_.template FreeTensor(kl);
        }
        PipeBarrier<PIPE_V>();

        // V.
        {
            LocalTensor<inType> vl = vInQ_.template AllocTensor<inType>();
            DataCopyExtParams vp{1, (uint32_t)(DV_ * sizeof(inType)), 0, 0, 0};
            DataCopyPadExtParams<inType> vpp{true, 0, padV_bf16_, 0};
            DataCopyPad(vl, mqGm_[vo], vp, vpp);
            vInQ_.template EnQue<inType>(vl);
            vl = vInQ_.template DeQue<inType>();
            Cast(vUb_, vl, RoundMode::CAST_NONE, DV_);
            vInQ_.template FreeTensor(vl);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadState(uint32_t sid, uint32_t hvi)
    {
        uint64_t bs = ((uint64_t)sid * HV_ + hvi) * DV_ * DK_;
        LocalTensor<stateType> sl = sInQ_.template AllocTensor<stateType>();
        DataCopyExtParams sp{static_cast<uint16_t>(DV_),
                             static_cast<uint32_t>(DK_ * sizeof(stateType)), 0, 0, 0};
        DataCopyPadExtParams<stateType> spp{true, 0, padK_fp32_, 0};
        DataCopyPad(sl, sGm_[bs], sp, spp);
        sInQ_.template EnQue<stateType>(sl);
        sl = sInQ_.template DeQue<stateType>();
        if constexpr (std::is_same_v<stateType, float32_t>) {
            DataCopy(sUb_, sl, DV_ * CK_);
        } else {
            Cast(sUb_, sl, RoundMode::CAST_NONE, DV_ * CK_);
        }
        sInQ_.template FreeTensor(sl);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreState(uint32_t sid, uint32_t hvi)
    {
        uint64_t bs = ((uint64_t)sid * HV_ + hvi) * DV_ * DK_;
        LocalTensor<stateType> sl = sOutQ_.template AllocTensor<stateType>();
        if constexpr (std::is_same_v<stateType, float32_t>) {
            DataCopy(sl, sUb_, DV_ * CK_);
        } else {
            Cast(sl, sUb_, RoundMode::CAST_RINT, DV_ * CK_);
        }
        sOutQ_.template EnQue<stateType>(sl);
        sl = sOutQ_.template DeQue<stateType>();
        DataCopyParams sp{static_cast<uint16_t>(DV_),
                          static_cast<uint16_t>(DK_ * sizeof(stateType)), 0, 0};
        DataCopyPad(sGm_[bs], sl, sp);
        sOutQ_.template FreeTensor(sl);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOut(uint32_t bi, uint32_t hvi)
    {
        uint64_t oo = ((uint64_t)bi * HV_ + hvi) * DV_;
        LocalTensor<inType> ol = oOutQ_.template AllocTensor<inType>();
        Cast(ol, oUb_, RoundMode::CAST_RINT, DV_);
        oOutQ_.template EnQue<inType>(ol);
        ol = oOutQ_.template DeQue<inType>();
        DataCopyParams op{1, static_cast<uint16_t>(DV_ * sizeof(inType)), 0, 0};
        DataCopyPad(oGm_[oo], ol, op);
        oOutQ_.template FreeTensor(ol);
        PipeBarrier<PIPE_V>();
    }

    // ============ Compute ============

    /*!
     * \brief L2Norm: x = x / sqrt(sum(x^2) + eps).
     * \param stride  fp32 compute stride (CK_ or CV_).
     * \param n       actual element count (DK_ or DV_).
     *
     * Uses scalar GetValue accumulation instead of WholeReduceSum
     * because WholeReduceSum is not available in CANN 9.0.0.
     */
    __aicore__ inline void L2Norm(LocalTensor<float> &x, uint32_t stride, uint32_t n)
    {
        // Scalar accumulation of sum of squares.
        float sumSq = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            float xi = x.GetValue(i);
            sumSq += xi * xi;
        }
        // Compute 1/sqrt(sumSq + eps) via single-element vector ops.
        LocalTensor<float> et = wUb_;
        et.SetValue(0, sumSq + EPS);
        Sqrt(et, et, 1);
        PipeBarrier<PIPE_V>();
        float invNorm = 1.0f / et.GetValue(0);
        Muls(x, x, invNorm, n);
        PipeBarrier<PIPE_V>();
    }

    /*!
     * \brief GDN sigmoid gating: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
     * Reads a and b (bf16) from GM via DMA+Cast to avoid unsupported (float) scalar cast.
     */
    __aicore__ inline void Gate(uint32_t bi, uint32_t hvi, float nea, float dt,
                                 float &av, float &bv, float &g, float &bt)
    {
        uint64_t off = (uint64_t)bi * HV_ + hvi;
        LocalTensor<inType> tmpBf16 = wBuf_.Get<inType>();
        LocalTensor<float>   tmpFp32 = wBuf_.Get<float>();
        DataCopyParams cp{1, static_cast<uint16_t>(sizeof(inType)), 0, 0};

        // Read a (bf16) via DMA, Cast to fp32.
        DataCopy(tmpBf16, aGm_[off], cp);
        PipeBarrier<PIPE_V>();
        Cast(tmpFp32, tmpBf16, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        av = tmpFp32.GetValue(0);
        float x = av + dt;

        if (x <= SPTH) {
            tmpFp32.SetValue(0, x);
            Exp(tmpFp32, tmpFp32, 1);
            PipeBarrier<PIPE_V>();
            float ev = tmpFp32.GetValue(0);
            tmpFp32.SetValue(0, 1.0f + ev);
            Ln(tmpFp32, tmpFp32, 1);
            PipeBarrier<PIPE_V>();
            g = nea * tmpFp32.GetValue(0);
        } else {
            g = nea * x;
        }

        // Read b (bf16) via DMA, Cast to fp32.
        DataCopy(tmpBf16, bGm_[off], cp);
        PipeBarrier<PIPE_V>();
        Cast(tmpFp32, tmpBf16, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_V>();
        bv = tmpFp32.GetValue(0);

        tmpFp32.SetValue(0, -bv);
        Exp(tmpFp32, tmpFp32, 1);
        PipeBarrier<PIPE_V>();
        bt = 1.0f / (1.0f + tmpFp32.GetValue(0));
    }

    __aicore__ inline void Recur(float g, float bt)
    {
        LocalTensor<float> et = wUb_;
        et.SetValue(0, g);
        Exp(et, et, 1);
        PipeBarrier<PIPE_V>();
        float eg = et.GetValue(0);

        Muls(sUb_, sUb_, eg, DV_ * CK_);
        PipeBarrier<PIPE_V>();
        MatVec(sUb_, kUb_, dUb_, DV_, DK_);
        Sub(dUb_, vUb_, dUb_, DV_);
        PipeBarrier<PIPE_V>();
        Muls(dUb_, dUb_, bt, DV_);
        PipeBarrier<PIPE_V>();
        OuterAdd(sUb_, dUb_, kUb_, DV_, DK_);
        MatVec(sUb_, qUb_, oUb_, DV_, DK_);
    }

    __aicore__ inline void MatVec(const LocalTensor<float> &mat, const LocalTensor<float> &vec,
                                   LocalTensor<float> &out, uint32_t rows, uint32_t cols)
    {
        uint8_t rst = static_cast<uint8_t>(CK_ / FP32_PER_BLK);
        for (uint32_t j = 0; j < cols; j += RLEN) {
            uint32_t m = (cols - j > RLEN) ? RLEN : (cols - j);
            for (uint32_t rb = 0; rb < rows; rb += MXR) {
                uint8_t rc = static_cast<uint8_t>((rows - rb > MXR) ? MXR : (rows - rb));
                Mul(wUb_[rb * CK_ + j], mat[rb * CK_ + j], vec[j], m, rc,
                    {1, 1, 1, rst, rst, 0});
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) {
            float s = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) s += wUb_.GetValue(r * CK_ + c);
            out.SetValue(r, s);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void OuterAdd(const LocalTensor<float> &mat, const LocalTensor<float> &delta,
                                     const LocalTensor<float> &vec, uint32_t rows, uint32_t cols)
    {
        for (uint32_t r = 0; r < rows; ++r) {
            float dv = delta.GetValue(r);
            Duplicate<float>(wUb_[r * CK_], dv, cols);
            PipeBarrier<PIPE_V>();
            Mul(wUb_[r * CK_], wUb_[r * CK_], vec, cols);
            PipeBarrier<PIPE_V>();
            Add(mat[r * CK_], mat[r * CK_], wUb_[r * CK_], cols);
            PipeBarrier<PIPE_V>();
        }
    }

    // ---- Members ----
    TPipe *pipe_{nullptr};
    GlobalTensor<inType>   mqGm_, aGm_, bGm_, oGm_;
    GlobalTensor<float>    alGm_, dtGm_;
    GlobalTensor<stateType> sGm_;
    GlobalTensor<int32_t>  ssiGm_;

    // DMA queues (TQue).
    TQue<QuePosition::VECIN,  1> qInQ_, kInQ_, vInQ_, sInQ_;
    TQue<QuePosition::VECOUT, 1> sOutQ_, oOutQ_;

    // TBuf for compute and constant buffers.
    TBuf<TPosition::VECCALC> qBuf_, kBuf_, vBuf_, sBuf_, dBuf_, oBuf_, wBuf_, alBuf_, dtBuf_;

    // Compute buffers (TBuf → LocalTensor<float>).
    LocalTensor<float> alUb_, dtUb_, qUb_, kUb_, vUb_, sUb_, dUb_, oUb_, wUb_;

    // Dimensions.
    uint32_t B_, H_, HV_, DK_, DV_, qD_, sN_;
    float    sc_;

    // Compute strides (fp32 aligned to 32B).
    uint32_t CK_, CV_;

    // DMA padding and aligned sizes.
    uint8_t  padK_bf16_, padV_bf16_, padK_fp32_;
    uint32_t bufK_bf16_, bufV_bf16_, bufK_fp32_;
};

} // namespace FusedPackedRecurrentGatedDeltaRule
#endif
