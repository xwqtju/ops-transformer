/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file moe_distribute_combine_a2_layered.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
#define MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_tiling.h"
#include "../moe_distribute_dispatch/moe_distribute_base.h"

namespace MoeDistributeCombineA2Impl {

constexpr int UB_ALIGN_SIZE = 32;
constexpr uint64_t CACHELINE_SIZE = 64;

#define TemplateMC2TypeA2layeredClass typename ExpandXType, typename ExpandIdxType, typename ExpandXTransType
#define TemplateMC2TypeA2layeredFunc ExpandXType, ExpandIdxType, ExpandXTransType

template<typename T>
struct OutputType {
    using type = T;
};
// 针对float16_t的特化
template<>
struct OutputType<half> {
    using type = half;
};
// 针对bfloat16_t的特化
template<>
struct OutputType<bfloat16_t> {
    using type = float;
};
// 辅助类型别名（C++11起支持）
template<typename T>
using OutputType_t = typename OutputType<T>::type;

using namespace AscendC;
template<TemplateMC2TypeA2layeredClass>
class MoeDistributeCombineA2Layered {
public:
    constexpr static uint32_t BUFFER_NUM = 2U;                   // 多buf
    constexpr static uint32_t STATE_OFFSET = 512U;              // 状态空间偏移地址
    constexpr static uint32_t STATE_SPACE_SIZE = 1024U * 1024U;  // 1M
    constexpr static uint32_t UB_ALIGN = 32U;                   // UB按32字节对齐
    constexpr static uint32_t SELF_STATE_OFFSET = 512U * 1024U;  // 本卡状态空间偏移地址

    constexpr static uint32_t BLOCK_SIZE = 32U;
    constexpr static uint32_t B16_PER_BLOCK = 16U;
    constexpr static uint32_t B32_PER_BLOCK = 8U;
    constexpr static uint32_t B64_PER_BLOCK = 4U;
    constexpr static uint32_t SERVER_RANK_SIZE = 8U;
    constexpr static uint32_t IPC_DATA_OFFSET = 4U * 1024U * 1024U;
    constexpr static uint32_t RDMA_DATA_SIZE = 100U * 1024U * 1024U;
    constexpr static uint32_t VEC_LEN = 256U;
    constexpr static uint32_t MAGIC_OFFSET = 2U * 1024U * 1024U - 32U * 32U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U; // 专家信息 权重信息 量化Scale 到达标志位
    constexpr static uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr static bool DynamicQuant = std::is_same<ExpandXTransType, int8_t>::value;
    constexpr static uint32_t TBUF_SIZE = 185U * 1024U;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 0U;
    constexpr static uint32_t IPC_REDUCE_USED_CORE_NUM = 32U; // 拉起远端IPC和机内reduce需要的核数
    constexpr static uint32_t WEIGHT_VALUE_NUM = 16U; // token(h * sizeof(bf/fp16)) + scale(32B) = (h + 16) * 2B
    constexpr static uint32_t TIME_CYCLE = 50;
    template <AscendC::HardEvent event>
    __aicore__ inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }
    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }
    template <typename T>
    inline __aicore__ T CeilDiv(const T dividend, const T divisor)
    {
        return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
    }

    __aicore__ inline MoeDistributeCombineA2Layered(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR waitCost, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineA2TilingData *tilingData, GM_ADDR contextGM);
    __aicore__ inline void Process();
    __aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo);

private:
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void GM2IPC();
    __aicore__ inline void WaitIPC();
    __aicore__ inline void SumToWindow();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline void SumToServer();
    __aicore__ inline void Preload();
    __aicore__ inline void ToWindowPreload();

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<ExpandIdxType> bkCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;

    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXTransType> localInWindow_;
    GlobalTensor<uint32_t> bufferIdGlobal_;     // 用于存对端状态window的变量
    GlobalTensor<int32_t> statusSpaceGlobal_;   // win区状态位置拷入相关参数
    GlobalTensor<int32_t> readStateGlobal_;
    GlobalTensor<uint64_t> waitCostU64GMTensor_;
    GlobalTensor<uint32_t> waitCostU32GMTensor_;

    uint64_t shareAddreRank[8];

    // 低精度需要用到的变量
    GlobalTensor<ExpandXType> scaleOutWindow_; // 第一层输出的scale值和offset，都是fp16格式
    GlobalTensor<ExpandXType> localInScaleWindow_;
    OutputType_t<ExpandXType> scaleMulVal;
    uint32_t mask;

    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR expandXGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR sendCountGM_;
    GM_ADDR scalesGM_;
    GM_ADDR waitCostGM_;
    GM_ADDR XOutGM_;
    __gm__ HcclAiRMAInfo* qp_info_;

    // 分层所需的参数
    GM_ADDR shareAddrGM_;
    GM_ADDR offsetInnerGM_;
    GM_ADDR countInnerGM_;
    GM_ADDR offsetOuterGM_;
    GM_ADDR countOuterGM_;

    GlobalTensor<int32_t> shareAddrGlobal_;
    GlobalTensor<int64_t> shareFlagGlobal_;
    GlobalTensor<ExpandXType> shareMemGlobal_;
    GlobalTensor<ExpandXType> dstshareMemGlobal_;
    GlobalTensor<int32_t> magicGlobal_;
    GlobalTensor<int32_t> offsetInnerGlobal_;
    GlobalTensor<int16_t> countInnerGlobal_;
    GlobalTensor<int32_t> offsetOuterGlobal_;
    GlobalTensor<int32_t> countOuterGlobal_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t countReL{0};
    uint32_t axisBS_{0};
    uint32_t globalBs{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};  // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    __gm__ HcclA2CombineOpParam *winContext_{nullptr};
    uint32_t moeExpertNum_{0};       // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0};  // 每张卡的专家数
    uint32_t expandXRows_;
    uint64_t rankSizeOnWin_{0};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint64_t dataOffsetOnWin_{0};
    uint64_t stateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t tokenNumPerCore_{0};
    uint32_t tokenIndex_{0};
    uint32_t serverNum{0};
    uint32_t ipcSliceSize{0};
    uint32_t ipcSliceNodeSize{0};
    uint64_t send_counts_inner_offset{0};
    uint64_t offset_inner_offset{0};
    uint64_t send_counts_outer_offset{0};
    uint64_t offset_outer_offset{0};
    uint64_t share_offset{0};
    uint32_t IPC_DATA_SIZE{0};
    uint32_t waitCostSize_{0};
    bool isWaitCost_=false;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;
    TBuf<> statusBuf_;
    TBuf<> waitCostBuf_; 

    int32_t sumTarget_{0};
    int32_t stateValue_{0};
    uint32_t startBs{0};
    uint32_t endBs{0};
    uint32_t processNum{0};
    uint32_t resNum{0};
    uint32_t resLen{0};
    uint32_t offsetIndex{0};
    uint32_t maxLocalBs{0};
    uint32_t stepCoreNum{0};
    int32_t magicValue{0};
    LocalTensor<int32_t> offsetReduceLocal_;
    LocalTensor<int32_t> countReduceLocal_;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;
    LocalTensor<uint64_t> waitCostU64Tensor_;
    LocalTensor<uint32_t> waitCostU32Tensor_;
    // 低精度相关
    uint32_t repeatNum{0};
    uint32_t scaleNum;
    uint32_t scaleNumAlign;
    uint32_t SCALE_GRANU;
    uint32_t lastRepeatNum{0};
};

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo*)QpInfo)->qpNum;
    auto qp_ctx_entry = (__gm__ HcclAiRMAWQ*)(((__gm__ HcclAiRMAInfo*)QpInfo)->sqPtr +
        destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo*)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAMem;
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo*)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t*)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t*)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    uint64_t shift = 15U;
    auto QP_DEPTH = qp_ctx_entry->depth;

    PipeBarrier<PIPE_ALL>();

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while(1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                       // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);   // [7] owner_bit
    byte_4 |= 1U << 8U;                         // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t*)(wqeAddr) = byte_4;          // Control set by local parameter see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen;  // message size
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0;           // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U;  // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0;          // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(wqeAddr + 20) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))->key;
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr; // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t)) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))->key; // L_Key
    *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr; // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn; // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL; // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL; // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL; // [48:50] DB_SL = qp.sl

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t* )(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR waitCost, GM_ADDR XOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData, GM_ADDR contextGM)
{
    tpipe_ = pipe;
    expandXGM_ = expandX;
    expertIdsGM_ = expertIds;
    expandIdxGM_ = expandIdx;
    sendCountGM_ = sendCount;
    scalesGM_ = scales;
    XOutGM_ = XOut;
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;
    waitCostSize_ = worldSize_;
    isWaitCost_ = (waitCost != nullptr);
    if (isWaitCost_) {
        waitCostGM_ = waitCost;
    }

    globalBs = tilingData->moeDistributeCombineInfo.globalBs;
    if (globalBs >= 256U) {
        maxLocalBs = 256U;
    } else {
        maxLocalBs = globalBs;
    }

    if constexpr (std::is_same<ExpandXType, half>::value) { // fp16
        SCALE_GRANU = 16U;
        scaleNum = axisH_ / SCALE_GRANU;
        scaleNumAlign = RoundUp(scaleNum, (uint32_t)(UB_ALIGN / sizeof(ExpandXType)));
        repeatNum = CeilDiv(axisH_, (VEC_LEN / static_cast<uint32_t>(sizeof(ExpandXType))));
        uint32_t vecNum = VEC_LEN / static_cast<uint32_t>(sizeof(ExpandXType));
        if (axisH_ >= vecNum) {
            mask = vecNum;
        } else {
            mask = axisH_;
        }

    } else { // bf16
        SCALE_GRANU = 8U;
        scaleNum = axisH_ / SCALE_GRANU;
        scaleNumAlign = RoundUp(scaleNum, (uint32_t)(UB_ALIGN / sizeof(ExpandXType)));
        repeatNum = CeilDiv(axisH_, (VEC_LEN / static_cast<uint32_t>(sizeof(float))));
        uint32_t vecNum = VEC_LEN / static_cast<uint32_t>(sizeof(float));//Brcb 8个datablock(32Bytes)
        if (axisH_ >= vecNum) {
            mask = vecNum;
        } else {
            mask = axisH_;
        }
    }
    scaleMulVal = 1 / 127.;

    winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM;
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineA2TilingData, mc2CcTiling));
    qp_info_ = (__gm__ HcclAiRMAInfo*)(((__gm__ HcclA2CombineOpParam*)contextGM)->aiRMAInfo);

    halfWinSize_ = RDMA_DATA_SIZE / 2U;
    IPC_DATA_SIZE = winContext_->winSize - RDMA_DATA_SIZE - IPC_DATA_OFFSET;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_ + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferIdGlobal_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;
    coreIdx_ = GetBlockIdx();

    serverNum = worldSize_ / SERVER_RANK_SIZE;
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    bkCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)(sendCount + worldSize_ * localMoeExpertNum_ * 4));
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    if (isWaitCost_) {
        waitCostU64GMTensor_.SetGlobalBuffer((__gm__ uint64_t*)waitCost);
        waitCostU32GMTensor_.SetGlobalBuffer((__gm__ uint32_t*)waitCost);
    }
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    readStateGlobal_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + dataSpaceSize_));
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    expandXRows_ = localMoeExpertNum_ * axisBS_ * worldSize_;
    rankSizeOnWin_ = static_cast<uint64_t>(dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE);
    statusSpaceGm_ = windowInGM_ + dataSpaceSize_;
    statusSpaceGlobal_.SetGlobalBuffer((__gm__ int32_t *)statusSpaceGm_);
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = static_cast<uint64_t>(dataSpaceSize_ + rankId_ * STATE_OFFSET);
    axisHFloatSize_ = axisH_ * static_cast<uint32_t>(sizeof(float));
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));

    uint64_t winSizeMin = moeExpertNum_ * axisBS_ * (axisHExpandXTypeSize_ + EXTRA_TOKEN_INFO_NUM * axisK_ * sizeof(uint32_t)) +
        IPC_DATA_OFFSET + RDMA_DATA_SIZE; // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小

    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    // coreIdx_ < serverNum
    int32_t state = selfStatusTensor(coreIdx_ * UB_ALIGN);

    if (state == 0) {
        sumTarget_ = static_cast<int32_t>(1);
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 1;
        stateValue_ = 1;
    } else {
        sumTarget_ = 0;
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 0;
        stateValue_ = 0;
    }

    BuffInit();

    SplitCoreCal();

    if (coreIdx_ == 0U) {
        readStateGlobal_.SetValue(0, stateValue_);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            readStateGlobal_);
    }
    send_counts_inner_offset = static_cast<uint64_t>(worldSize_ * localMoeExpertNum_);
    offset_inner_offset = send_counts_inner_offset + static_cast<uint64_t>(globalBs * serverNum / 2);
    send_counts_outer_offset = offset_inner_offset + static_cast<uint64_t>(globalBs * axisK_ * serverNum);
    offset_outer_offset = send_counts_outer_offset + static_cast<uint64_t>(axisBS_);
    share_offset = offset_outer_offset + static_cast<uint64_t>(axisBS_ * serverNum);

    shareAddrGM_ = sendCount + share_offset;
    offsetInnerGM_ = sendCount + offset_inner_offset;
    countInnerGM_ = sendCount + send_counts_inner_offset;
    offsetOuterGM_ = sendCount + offset_outer_offset;
    countOuterGM_ = sendCount + send_counts_outer_offset;

    shareAddrGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + share_offset);
    countInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(sendCount) + send_counts_inner_offset * 2);
    offsetInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_inner_offset);
    countOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + send_counts_outer_offset);
    offsetOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_outer_offset);

    PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < 8; i++) {
        shareAddreRank[i] = reinterpret_cast<uint64_t>(
            RDMA_DATA_SIZE + hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i));
    }
    magicGlobal_.SetGlobalBuffer((__gm__ int32_t*)(shareAddreRank[rankId_ % SERVER_RANK_SIZE]));
    magicValue = magicGlobal_.GetValue(MAGIC_OFFSET / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::BuffInit()
{
    // 状态tBuf
    tpipe_->InitBuffer(statusBuf_, worldSize_ * UB_ALIGN);

    // AIVRDMAPostSend函数需要的tBuf
    tpipe_->InitBuffer(rdmaInBuf_, UB_ALIGN_SIZE);
    ubLocal = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_ALIGN_SIZE);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();

    if (isWaitCost_) {
        tpipe_->InitBuffer(waitCostBuf_, waitCostSize_ * sizeof(uint64_t));
        waitCostU64Tensor_ = waitCostBuf_.Get<uint64_t>();
        waitCostU32Tensor_ = waitCostU64Tensor_.template ReinterpretCast<uint32_t>();
        Duplicate<uint32_t>(waitCostU32Tensor_, 0, waitCostSize_ * sizeof(uint64_t) / sizeof(uint32_t));
    }

    // 总tBuf
    tpipe_->InitBuffer(tBuf, TBUF_SIZE);
}
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    sendRankNum_ = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::GM2IPC()
{
    ipcSliceSize = IPC_DATA_SIZE / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    ipcSliceNodeSize = ipcSliceSize * SERVER_RANK_SIZE;

    // 初始化baseBuffOffset
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    // 申请LocalTensor : sendCount 以及计算偏移
    LocalTensor<ExpandIdxType> sendCountLocal = tBuf.GetWithOffset<int32_t>(RoundUp(moeExpertNum_, B32_PER_BLOCK),
        baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, B32_PER_BLOCK);

    // 申请LocalTensor : expandScales 以及计算偏移
    LocalTensor<float> expandScalesLocal = tBuf.GetWithOffset<float>(
        (maxLocalBs + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN, baseBuffOffset);
    baseBuffOffset += sizeof(float) * ((maxLocalBs + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN);

    // 申请LocalTensor : InUb。 token：【data】(H * fp16/bf16) + expandScales(32B)
    LocalTensor<ExpandXType> inUb = tBuf.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    LocalTensor<float> inUbTemp = inUb[axisH_].template ReinterpretCast<float>();

    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK)); // mte2
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t localShareAddr = shareAddreRank[rankId_ % SERVER_RANK_SIZE];
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {

        uint64_t targetRankAddr = localShareAddr +
                                  static_cast<uint64_t>(dstRankId * ipcSliceSize + IPC_DATA_OFFSET);

        dstshareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)(targetRankAddr));

        uint32_t rankTokenNum = 0U;

        for (uint32_t expertId = 0U; expertId < localMoeExpertNum_; ++expertId) {
            uint32_t preCount = 0U;

            if (expertId != 0U || dstRankId != 0U) {
                preCount = static_cast<uint32_t>(sendCountLocal.GetValue(expertId * worldSize_ + dstRankId - 1));
            }
            uint32_t tokenNum = sendCountLocal.GetValue(expertId * worldSize_ + dstRankId) - preCount;
            uint32_t startTokenAddr = preCount * axisH_;
            PipeBarrier<PIPE_ALL>();
            DataCopy(expandScalesLocal, expandScalesGlobal_[preCount], (tokenNum + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            for (uint32_t tokenId = 0U; tokenId < tokenNum; ++tokenId) {
                float scaleVal = expandScalesLocal.GetValue(tokenId);
                inUbTemp(0) = scaleVal;
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                DataCopy(inUb, expandXGlobal_[startTokenAddr], axisH_);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopy(dstshareMemGlobal_[rankTokenNum * (axisH_ + WEIGHT_VALUE_NUM)],
                    inUb, axisH_ + WEIGHT_VALUE_NUM);
                startTokenAddr += axisH_;
                rankTokenNum++;
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    SyncAll<true>();
    if (coreIdx_ < SERVER_RANK_SIZE) {
        uint64_t targetAddr = shareAddreRank[coreIdx_ % SERVER_RANK_SIZE];
        shareFlagGlobal_.SetGlobalBuffer((__gm__ int64_t *)targetAddr);
        LocalTensor<int64_t> inUb = statusBuf_.Get<int64_t>();
        inUb(0) = 12345 + magicValue;
        uint32_t flagOffset = rankId_ % SERVER_RANK_SIZE;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_[flagOffset * 4], inUb, 4);  // *4是因为单次拷贝256byte = 4*int64

        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WaitIPC()
{
    ///***
    uint32_t stepCoreNum_ = SERVER_RANK_SIZE;
    shareFlagGlobal_.SetGlobalBuffer((__gm__ int64_t *)shareAddreRank[rankId_ % SERVER_RANK_SIZE]);
    // 只要8个core分别wait 来自8卡的flag，然后sync一下 再进行流水

    if (coreIdx_ < stepCoreNum_){
	    uint32_t start = GetSystemCycle() / TIME_CYCLE;
	    LocalTensor<int64_t> inUb = statusBuf_.Get<int64_t>();
        uint32_t waitFlagAddr = coreIdx_ % stepCoreNum_;
        while (true) {
            DataCopy(inUb, shareFlagGlobal_[waitFlagAddr * 4], 4);
            PipeBarrier<PIPE_ALL>();
            if (inUb(0) >= (12345 + magicValue)) {
                break;
            }
        }
        inUb(0) = 0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_[waitFlagAddr * 4], inUb, 4);  // *4是因为单次拷贝256byte = 4*int64
        PipeBarrier<PIPE_ALL>();
	    uint32_t end = GetSystemCycle() / TIME_CYCLE;
        uint32_t duration = end - start;
	    auto id = (rankId_ / SERVER_RANK_SIZE) * SERVER_RANK_SIZE + coreIdx_;
    	if (isWaitCost_) {
	        waitCostU32Tensor_.SetValue(id * sizeof(uint64_t) / sizeof(uint32_t), duration);
	        AscendC::SetAtomicAdd<int32_t>();
            AscendC::DataCopy(waitCostU32GMTensor_, waitCostU32Tensor_, waitCostSize_ * sizeof(uint64_t) / sizeof(uint32_t));
            AscendC::SetAtomicNone();
        }
    }
    SyncAll<true>();
}


template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumToWindow()
{
    // 32core流水并行
    uint32_t coreNumPerServer = stepCoreNum / serverNum;
    uint32_t serverId_ = coreIdx_ / coreNumPerServer;
    uint32_t targetRankId_ = rankId_ % SERVER_RANK_SIZE + serverId_ * SERVER_RANK_SIZE;

    // 初始baseBuffOffset
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    LocalTensor<int16_t> countReduceLocal  = tBuf.GetWithOffset<int16_t>(RoundUp(maxLocalBs,
        B16_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int16_t) * RoundUp(maxLocalBs, B16_PER_BLOCK); // 需要32字节对齐

    LocalTensor<int32_t> offsetReduceLocal = tBuf.GetWithOffset<int32_t>(RoundUp(maxLocalBs * axisK_,
        B32_PER_BLOCK), baseBuffOffset);

    // 量化和不量化都要用
    baseBuffOffset += sizeof(int32_t) * RoundUp(maxLocalBs * axisK_, B32_PER_BLOCK);
    LocalTensor<ExpandXType> dataInLocal = tBuf.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    baseBuffOffset += sizeof(ExpandXType) * (axisH_ + WEIGHT_VALUE_NUM);

    // 初始化 fp16所需tBuf偏移的base Offset
    uint32_t fp16baseBuffOffset = baseBuffOffset;

    // 量化和不量化都要用 同时也为bf16的Brcb函数扩充复用，扩充到H个，至少要256B对齐
    LocalTensor<float> castInFloatLocal = tBuf.GetWithOffset<float>(
        RoundUp(axisHFloatSize_, VEC_LEN) / sizeof(float), baseBuffOffset);
    baseBuffOffset += RoundUp(axisHFloatSize_, VEC_LEN);

    // 量化和不量化都要用
    LocalTensor<float> sumFloatLocal = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);

    // token格式: data(H*sizeof(ExpandXType)) + weight值(32B)
    LocalTensor<float> inUbTemp = dataInLocal[axisH_].template ReinterpretCast<float>();

    // 量化 dataInLocal复用 存放 int8的data fp/bf16的scale
    LocalTensor<ExpandXTransType> castDataInt8 = dataInLocal.template ReinterpretCast<ExpandXTransType>();
    LocalTensor<ExpandXType> scaleData = dataInLocal[axisH_/2].template ReinterpretCast<ExpandXType>();

    // 量化fp16
    LocalTensor<ExpandXType> sumHalfLocal = tBuf.GetWithOffset<ExpandXType>(
        axisH_, fp16baseBuffOffset); // 复用castInFloatLocal
    fp16baseBuffOffset += axisH_ * sizeof(ExpandXType);

    // 16个数取最大值
    LocalTensor<ExpandXType> reduceMaxOutTensor= tBuf.GetWithOffset<ExpandXType>(scaleNum, fp16baseBuffOffset);

    // 将scale利用Brcb函数扩充到H个，至少要256B对齐   复用reduceMaxOutTensor
    LocalTensor<ExpandXType> absScaleTensor = tBuf.GetWithOffset<ExpandXType>(
        RoundUp(axisHExpandXTypeSize_, VEC_LEN) / sizeof(ExpandXType), fp16baseBuffOffset);

    // 量化 bf16 复用sumFloatLocal
    LocalTensor<half> halfLocal = tBuf.GetWithOffset<half>(axisH_, baseBuffOffset);

    baseBuffOffset += sizeof(float) * (axisH_); // 复用sumFloatLocal，但是offset要加上sumFloatLocal大小
    LocalTensor<float> reduceMaxTensorFloat = tBuf.GetWithOffset<float>(scaleNum, baseBuffOffset);

    DataCopy(countReduceLocal,
             countInnerGlobal_[globalBs * serverId_], RoundUp(maxLocalBs, B16_PER_BLOCK));
    DataCopy(offsetReduceLocal,
             offsetInnerGlobal_[globalBs * axisK_ * serverId_], RoundUp(maxLocalBs * axisK_, B32_PER_BLOCK));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t rdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                   serverId_ * rankSizeOnWin_ * SERVER_RANK_SIZE);
    scaleOutWindow_.SetGlobalBuffer((__gm__ ExpandXType*)rdmaAddr); // 16bit
    localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType*)rdmaAddr);
    LocalTensor<int64_t> rdmaFlagLocal = statusBuf_.Get<int64_t>();
    rdmaFlagLocal(0) = 123 + magicValue;
    PipeBarrier<PIPE_ALL>();
    int offsetPre = 0;
    offsetIndex = 0U;

    // 计算offsetIndex,copyNum,dataOffset,scaleOffset
    uint32_t listLen = 64 ; // maxLocalBs / coreNumPerServer;
    uint32_t offsetIndexs[65];
    uint32_t copyNums[65];
    uint32_t dataOffsets[65];
    uint32_t scaleOffsets[65];
    uint32_t totalCopyLen = 0;
    uint32_t processNum_ = 0;
    // 每个核使用的链路要岔开，不能有冲突
    for (uint32_t i = 0U; i < maxLocalBs; i++) {
        if ((i % coreNumPerServer) == (coreIdx_ % coreNumPerServer)){
            int offsetCur = static_cast<int32_t>(countReduceLocal.GetValue(i));
            uint32_t dataOffset = i * (axisH_ / 2U + scaleNumAlign); // uint8的数据
            if (i != 0U) {
                offsetPre = static_cast<int32_t>(countReduceLocal.GetValue(i - 1));
            }
            int copyNum = offsetCur - offsetPre;
            if (copyNum <= 0) {
                break;
            }
            offsetIndex = static_cast<uint32_t>(offsetPre);

            offsetIndexs[processNum_] = offsetIndex;
            copyNums[processNum_] = static_cast<uint32_t>(copyNum);
            dataOffsets[processNum_] = dataOffset;
            totalCopyLen += static_cast<uint32_t>(copyNum);
            processNum_++;
        }
    }

    uint32_t processTokenNum = 0;
    uint32_t offsetIndexStart = offsetIndexs[processTokenNum];
    offsetIndex = offsetIndexs[processTokenNum];
    uint32_t copyNum = copyNums[processTokenNum];
    uint32_t dataOffset = dataOffsets[processTokenNum];

    uint32_t tokenOffset = 0;
    for (uint32_t i = 0U; i < totalCopyLen; i++) {
        uint32_t targetIpcRank = offsetReduceLocal.GetValue(offsetIndex) / (globalBs * axisK_);
        uint32_t targetIpcOffset = offsetReduceLocal.GetValue(offsetIndex) % (globalBs * axisK_) *
                                   (axisH_ + WEIGHT_VALUE_NUM);

        uint64_t copyAddr = shareAddreRank[targetIpcRank % SERVER_RANK_SIZE] +
                            static_cast<uint64_t>(targetRankId_ * ipcSliceSize) +
                            static_cast<uint64_t>(IPC_DATA_OFFSET);
        shareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)copyAddr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(dataInLocal, shareMemGlobal_[targetIpcOffset], axisH_ + WEIGHT_VALUE_NUM); // mte2
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        float scaleVal = inUbTemp(0);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(castInFloatLocal, dataInLocal, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        if ((offsetIndex - offsetIndexStart) == 0U) {
            Muls(sumFloatLocal, castInFloatLocal, scaleVal, axisH_);
        } else {
            Axpy(sumFloatLocal, castInFloatLocal, scaleVal, axisH_);
        }

        offsetIndex += 1U;

        PipeBarrier<PIPE_V>();
        if ((offsetIndex - offsetIndexStart) == copyNum){
            tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
            if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
                if constexpr (std::is_same<ExpandXType, half>::value) {

                    Cast(sumHalfLocal, sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
                    PipeBarrier<PIPE_V>();
                    Abs(absScaleTensor, sumHalfLocal, axisH_);
                    PipeBarrier<PIPE_V>();
                    BlockReduceMax(reduceMaxOutTensor, absScaleTensor, repeatNum, mask, 1, 1, 8); // g16
                    PipeBarrier<PIPE_V>();
                    SyncFunc<AscendC::HardEvent::MTE3_V>();
                    Muls(scaleData, reduceMaxOutTensor, scaleMulVal, scaleNum); // 1/scale = dmax / 127
                    PipeBarrier<PIPE_V>();
                    Brcb(absScaleTensor, scaleData, repeatNum, {1, 8}); // 填充scale值
                    PipeBarrier<PIPE_V>();

                    Div(sumHalfLocal, sumHalfLocal, absScaleTensor, axisH_); // data_fp16/(1/scale)
                    PipeBarrier<PIPE_V>();
                    Cast(castDataInt8, sumHalfLocal, RoundMode::CAST_RINT, axisH_); // fp16->int8 四舍六入五成双
                    PipeBarrier<PIPE_V>();

                    SyncFunc<AscendC::HardEvent::V_MTE3>();
                    DataCopy(localOutWindow_[dataOffset], dataInLocal, axisH_ / 2 + scaleNumAlign); // int8数据+scale值
                    PipeBarrier<PIPE_MTE3>();
                    DataCopy(shareFlagGlobal_[(serverId_ + 1) * 1024 + tokenOffset * 4], rdmaFlagLocal, 4);
                } else {

                    PipeBarrier<PIPE_V>();
                    Abs(castInFloatLocal, sumFloatLocal, axisH_); // 求fp32张量的绝对值
                    PipeBarrier<PIPE_V>();
                    BlockReduceMax(reduceMaxTensorFloat, castInFloatLocal, repeatNum, mask, 1, 1, 8); // fp32的g16
                    PipeBarrier<PIPE_V>();
                    Muls(reduceMaxTensorFloat, reduceMaxTensorFloat, scaleMulVal, scaleNum); // scale = dmax * 1/127
                    PipeBarrier<PIPE_V>();
                    Brcb(castInFloatLocal, reduceMaxTensorFloat, repeatNum, {1, 8}); // 填充fp32的scale值
                    PipeBarrier<PIPE_V>();
                    Div(sumFloatLocal, sumFloatLocal, castInFloatLocal, axisH_); // data_fp32/(1/scale)
                    PipeBarrier<PIPE_V>();
                    SyncFunc<AscendC::HardEvent::MTE3_V>();
                    Cast(scaleData, reduceMaxTensorFloat, RoundMode::CAST_RINT, scaleNum); // 1/scale从fp32量化成bf16
                    PipeBarrier<PIPE_V>();
                    Cast(halfLocal, sumFloatLocal, RoundMode::CAST_RINT, axisH_); // token数据fp32->bf16 四舍六入五成双
                    PipeBarrier<PIPE_V>();
                    Cast(castDataInt8, halfLocal, RoundMode::CAST_RINT, axisH_); // token数据bf16->int8 四舍六入五成双
                    PipeBarrier<PIPE_V>();
                    SyncFunc<AscendC::HardEvent::V_MTE3>();
                    DataCopy(localOutWindow_[dataOffset], dataInLocal, axisH_ / 2 + scaleNumAlign); // int8数据+scale值
                    PipeBarrier<PIPE_MTE3>();
                    DataCopy(shareFlagGlobal_[(serverId_ + 1) * 1024 + tokenOffset * 4], rdmaFlagLocal, 4);
                }
            } else {
                PipeBarrier<PIPE_V>();
                Cast(dataInLocal, sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopy(localOutWindow_[tokenOffset  * axisH_], dataInLocal, axisH_); // int8数据+scale值
                PipeBarrier<PIPE_MTE3>();
                DataCopy(shareFlagGlobal_[(serverId_ + 1) * 1024 + tokenOffset * 4], rdmaFlagLocal, 4);
            }
            processTokenNum++;
            offsetIndex = offsetIndexs[processTokenNum];
            copyNum = copyNums[processTokenNum];
            dataOffset = dataOffsets[processTokenNum];
            offsetIndexStart = offsetIndex;
        }
    }
    PipeBarrier<PIPE_ALL>();
    rdmaFlagLocal(0) = 321 + magicValue;
    tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
    DataCopy(shareFlagGlobal_[(serverId_ + 1) * 1024 + tokenOffset * 4], rdmaFlagLocal, 4);
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AlltoAllServerDispatch()
{
    LocalTensor<int64_t> checkRdmaLocal = statusBuf_.Get<int64_t>();
    LocalTensor<ExpandXTransType> tmpLowUb_ = tBuf.Get<ExpandXTransType>();
    uint32_t checkServer = coreIdx_ - stepCoreNum;
    GlobalTensor<ExpandXTransType> aivSrcGlobal;
    GlobalTensor<ExpandXTransType> aivDstGlobal;
    uint32_t tragRankId = rankId_ % SERVER_RANK_SIZE + SERVER_RANK_SIZE * checkServer;
    uint32_t copySum = 0;
    uint32_t copyOnceNum = 1;
    uint32_t copyLen_;
    uint32_t copyLenAlign_;
    uint32_t selfServerID = rankId_ / SERVER_RANK_SIZE;
    bool stopFlag = false;
    uint32_t cpNum = 0;

    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
        copyLen_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXTransType)) +
                   scaleNum * static_cast<uint32_t>(sizeof(ExpandXType));
        copyLenAlign_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXTransType)) +
                        scaleNumAlign * static_cast<uint32_t>(sizeof(ExpandXType));
    } else {
        copyLen_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
        copyLenAlign_ = copyLen_;
    }
    uint64_t srcrdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                      checkServer * rankSizeOnWin_ * SERVER_RANK_SIZE);
    uint64_t dstrdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(tragRankId) + halfWinSize_ * bufferId_ +
                                      (rankId_ / SERVER_RANK_SIZE) * rankSizeOnWin_ * SERVER_RANK_SIZE);
    while (!stopFlag) {
        for(uint32_t i = 0U; i < copyOnceNum; i++){
            while (true) {
                DataCopy(checkRdmaLocal[64], shareFlagGlobal_[(checkServer + 1) * 1024 + copySum * 4], 4);
                PipeBarrier<PIPE_ALL>();
                if (checkRdmaLocal.GetValue(64) == (123 + magicValue)) {
                    copySum++;
                    break;
                } else if(checkRdmaLocal.GetValue(64) == (321 + magicValue) || copySum == maxLocalBs){
                    stopFlag = true;
                    break;
                }
            }
            PipeBarrier<PIPE_ALL>();
            if(stopFlag){
                break;
            }
        }
        if(copySum > 0U){

            if(rankId_ != tragRankId) {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(dstrdmaAddr));
                AIVRDMAPostSend((GM_ADDR)(srcrdmaAddr + copyLenAlign_ * (copySum - copyOnceNum)),
                                (GM_ADDR)(dstrdmaAddr + copyLenAlign_ * (copySum - copyOnceNum)),
                                tragRankId, copyLen_ * copyOnceNum, qp_info_);
            } else {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(dstrdmaAddr));
                if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
                    cpNum = axisH_ + scaleNumAlign * static_cast<uint32_t>(sizeof(ExpandXType)) /
                            static_cast<uint32_t>(sizeof(ExpandXTransType));
                } else {
                    cpNum = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType)) /
                            static_cast<uint32_t>(sizeof(ExpandXTransType));
                }
                for (uint32_t k=0U ; k<copyOnceNum; k++) {
                    DataCopy(tmpLowUb_,
                             aivSrcGlobal[copyLenAlign_ * (copySum - copyOnceNum + k) / sizeof(ExpandXTransType)], cpNum);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(aivDstGlobal[copyLenAlign_ * (copySum - copyOnceNum + k) / sizeof(ExpandXTransType)],
                             tmpLowUb_, cpNum);
                }
            }
        }
    }
    if(rankId_ != tragRankId) {
        AIVRDMAPostSend((GM_ADDR)((uint64_t)(readStateGlobal_.GetPhyAddr())),
                        (GM_ADDR)((uint64_t)(hccl_.GetWindowsInAddr(tragRankId) +
                            halfWinSize_ * bufferId_ + dataSpaceSize_ + selfServerID * STATE_OFFSET)),
                        tragRankId, 32, qp_info_);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WaitDispatch()
{
    if ((coreIdx_ < serverNum) && (coreIdx_ != (rankId_ / SERVER_RANK_SIZE))) {
        uint32_t targetRank = rankId_ % SERVER_RANK_SIZE + (coreIdx_)*SERVER_RANK_SIZE;
        LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
        uint32_t readNum = 1U;
        DataCopyParams intriParams{static_cast<uint16_t>(readNum), 1, 15, 0};  // srcStride为15个block
        uint32_t start = GetSystemCycle() / TIME_CYCLE;
        while (true) {
            DataCopy(statusTensor, statusSpaceGlobal_[(coreIdx_)*STATE_OFFSET / sizeof(int32_t)], intriParams);
            PipeBarrier<PIPE_ALL>();
            int32_t sumOfFlag = statusTensor.GetValue(0);
            if (sumOfFlag == sumTarget_) {
                break;
            }
        }
        uint32_t end = GetSystemCycle() / TIME_CYCLE;
        uint32_t duration = end - start;
	    if (isWaitCost_) {
	        waitCostU32Tensor_.SetValue(targetRank * sizeof(uint64_t) / sizeof(uint32_t), duration);
	        AscendC::SetAtomicAdd<int32_t>();
            AscendC::DataCopy(waitCostU32GMTensor_, waitCostU32Tensor_, waitCostSize_ * sizeof(uint64_t) / sizeof(uint32_t));
            AscendC::SetAtomicNone();
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Preload()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        return;
    }
    processNum = axisBS_ / reduceCore;
    resNum = axisBS_ - processNum * reduceCore;
    resLen = (resNum == 0U) ? 0U : 1U;
    startBs = 0U;
    endBs = 0U;
    if (coreIdx_ < resNum) {
        processNum += 1U;
        startBs = coreIdx_ * processNum;
        endBs = startBs + processNum;
    } else {
        startBs = coreIdx_ * processNum + resNum;
        endBs = startBs + processNum;
    }
    uint64_t selfRankAddr = (uint64_t)(hccl_.GetWindowsInAddr(rankId_) + halfWinSize_ * bufferId_);
    localInWindow_.SetGlobalBuffer((__gm__ ExpandXTransType *)(selfRankAddr));

    // 低精度需要用到的变量
    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
        localInScaleWindow_.SetGlobalBuffer((__gm__ ExpandXType*)(selfRankAddr));
    }

    // 初始化offset
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    offsetReduceLocal_ = tBuf.GetWithOffset<int32_t>(
        RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);
    baseBuffOffset += sizeof(uint32_t) * RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t)));

    countReduceLocal_ = tBuf.GetWithOffset<int32_t>(
        RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);

    DataCopy(
        offsetReduceLocal_, offsetOuterGlobal_, RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t))));
    DataCopy(countReduceLocal_, countOuterGlobal_, RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)))); //256 * 4
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    offsetIndex = 0U;
    if (startBs != 0U) {
        offsetIndex = countReduceLocal_.GetValue(startBs - 1U);
    }
}
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumToServer()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        SyncAll<true>();
        return;
    }
    // 初始化 fp16  bf16的offset
    uint32_t baseBuffOffset = sizeof(uint32_t) * RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t)))
        + sizeof(int32_t) * RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)));
    uint32_t fpBaseBuffOffset = baseBuffOffset;
    uint32_t bfBaseBuffOffset = baseBuffOffset;

    // 不量化
    LocalTensor<float> sumFloatLocal = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);
    LocalTensor<ExpandXType> sumFpAndBfLocal = tBuf.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);

    LocalTensor<ExpandXType> dataIn = tBuf.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(ExpandXType);
    LocalTensor<float> castFp32 = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);

    // 量化 fp16
    LocalTensor<ExpandXType> sumFp16Local = tBuf.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXType);

    LocalTensor<ExpandXTransType> dataInt8 = tBuf.GetWithOffset<ExpandXTransType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXTransType);

    LocalTensor<ExpandXType> scaleData = tBuf.GetWithOffset<ExpandXType>(scaleNumAlign, fpBaseBuffOffset);
    fpBaseBuffOffset += scaleNumAlign * sizeof(ExpandXType);

    LocalTensor<ExpandXType> castFp16 = tBuf.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXType);

    LocalTensor<ExpandXType> scaleDup = tBuf.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);

    // 量化 bf16
    LocalTensor<float> sumFloatLocal1 = tBuf.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
    LocalTensor<ExpandXType> sumBf16Local = tBuf.GetWithOffset<ExpandXType>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(float);

    LocalTensor<ExpandXTransType> dataInUbInt8 = tBuf.GetWithOffset<ExpandXTransType>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(ExpandXTransType);

    LocalTensor<ExpandXType> scaleDataBf16 = tBuf.GetWithOffset<ExpandXType>(scaleNumAlign, bfBaseBuffOffset);
    bfBaseBuffOffset += scaleNumAlign * sizeof(ExpandXType);

    LocalTensor<half> castDataHalf = tBuf.GetWithOffset<half>(axisH_, bfBaseBuffOffset); // Bf16 用half代替
    bfBaseBuffOffset += axisH_ * sizeof(half);

    LocalTensor<float> castDataFp32 = tBuf.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(float);

    LocalTensor<float> castFp32scale = tBuf.GetWithOffset<float>(scaleNum, bfBaseBuffOffset);
    bfBaseBuffOffset += scaleNumAlign * sizeof(float);

    LocalTensor<float> castFp32ScaleBrcb = tBuf.GetWithOffset<float>(axisH_, bfBaseBuffOffset);

    for (uint32_t i = startBs; i < endBs; i++) {
        int offsetPre = 0;
        int offsetCur = countReduceLocal_.GetValue(i);
        if (i != 0U) {
            offsetPre = countReduceLocal_.GetValue(i - 1);
        }
        PipeBarrier<PIPE_ALL>(); // 高精度为了同步加入的 PIPE_ALL
        int copyNum = offsetCur - offsetPre;
        if (!copyNum) {
            break;
        }
        if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
            if constexpr (std::is_same<ExpandXType, half>::value) { // fp16
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate(sumFp16Local, static_cast<ExpandXType>(0.0), axisH_);
                for (int j = 0; j < copyNum; j++) {

                    int offsetOnIpc = (offsetReduceLocal_.GetValue(offsetIndex) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                                       offsetReduceLocal_.GetValue(offsetIndex) % axisBS_ * (axisH_ * sizeof(ExpandXTransType) +
                                                                                             scaleNumAlign * sizeof(ExpandXType))) / sizeof(ExpandXTransType);
                    SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
                    DataCopy(dataInt8, localInWindow_[offsetOnIpc], axisH_);
                    DataCopy(scaleData, localInScaleWindow_[((offsetOnIpc + axisH_) * sizeof(ExpandXTransType)) / sizeof(ExpandXType)], scaleNumAlign);

                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    Cast(castFp16, dataInt8, AscendC::RoundMode::CAST_NONE, axisH_);
                    PipeBarrier<PIPE_V>();
                    Brcb(scaleDup, scaleData, repeatNum, {1, 8}); // 填充scale值
                    PipeBarrier<PIPE_V>();
                    MulAddDst(sumFp16Local, castFp16, scaleDup, axisH_); // fp16乘加scale值
                    PipeBarrier<PIPE_V>();

                    offsetIndex ++;
                }
                PipeBarrier<PIPE_V>();
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopy(expandOutGlobal_[i * axisH_], sumFp16Local, axisH_);
                PipeBarrier<PIPE_V>();
            } else { // bf16
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate(sumFloatLocal1, 0.0f, axisH_);

                for (int j = 0; j < copyNum; j++) {

                    int offsetOnIpc = (offsetReduceLocal_.GetValue(offsetIndex) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                                       offsetReduceLocal_.GetValue(offsetIndex) % axisBS_ * (axisH_ * sizeof(ExpandXTransType) +
                                                                                             scaleNumAlign * sizeof(ExpandXType))) / sizeof(ExpandXTransType);
                    SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
                    DataCopy(dataInUbInt8, localInWindow_[offsetOnIpc], axisH_);
                    DataCopy(scaleDataBf16, localInScaleWindow_[((offsetOnIpc + axisH_) * sizeof(ExpandXTransType)) / sizeof(ExpandXType)], scaleNumAlign);

                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    // cast before muls
                    Cast(castDataHalf, dataInUbInt8, AscendC::RoundMode::CAST_NONE, axisH_); // data:int8->fp16
                    PipeBarrier<PIPE_V>();
                    Cast(castDataFp32, castDataHalf, AscendC::RoundMode::CAST_NONE, axisH_); // data:fp16->fp32
                    PipeBarrier<PIPE_V>();
                    Cast(castFp32scale, scaleDataBf16, AscendC::RoundMode::CAST_NONE, scaleNum); // scale:bf16->fp32
                    PipeBarrier<PIPE_V>();
                    Brcb(castFp32ScaleBrcb, castFp32scale, repeatNum, {1, 8}); // 填充fp32的scale值
                    PipeBarrier<PIPE_V>();
                    MulAddDst(sumFloatLocal1, castDataFp32, castFp32ScaleBrcb, axisH_); // fp16乘加scale值
                    PipeBarrier<PIPE_V>();
                    offsetIndex++;
                }
                PipeBarrier<PIPE_V>();
                Cast(sumBf16Local, sumFloatLocal1, AscendC::RoundMode::CAST_RINT, axisH_);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopy(expandOutGlobal_[i * axisH_], sumBf16Local, axisH_);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Duplicate(sumFloatLocal, 0.0f, axisH_);
            for (int j = 0; j < copyNum; j++) {
                int offsetOnIpc =
                    (offsetReduceLocal_.GetValue(offsetIndex) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                     offsetReduceLocal_.GetValue(offsetIndex) % axisBS_ * axisH_ * sizeof(ExpandXType)) /
                    sizeof(ExpandXType);
                SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
                DataCopy(dataIn, localInWindow_[offsetOnIpc], axisH_);
                SyncFunc<AscendC::HardEvent::MTE2_V>();
                // cast before muls
                Cast(castFp32, dataIn, AscendC::RoundMode::CAST_NONE, axisH_);
                PipeBarrier<PIPE_V>();
                // add mulBufLocal to sumFloatBufLocal
                AscendC::Add(sumFloatLocal, sumFloatLocal, castFp32, axisH_);
                offsetIndex++;
            }
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::MTE3_V>();
            Cast(sumFpAndBfLocal, sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(expandOutGlobal_[i * axisH_], sumFpAndBfLocal, axisH_);
            PipeBarrier<PIPE_V>();
        }
    }

    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Process()
{
    if ASCEND_IS_AIV {
        GM2IPC();
        WaitIPC();
        stepCoreNum = IPC_REDUCE_USED_CORE_NUM;
        if (coreIdx_ < stepCoreNum){
            SumToWindow();
        }
        else if (coreIdx_ < (stepCoreNum + serverNum)) {
            AlltoAllServerDispatch();
        } else {
            SyncAll<true>();
        }
        if (coreIdx_ == 0U) {
            magicGlobal_.SetValue(MAGIC_OFFSET / sizeof(int32_t), magicValue + 1);
            PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(magicGlobal_[MAGIC_OFFSET / sizeof(int32_t)]);
            bufferIdGlobal_(0) = bufferId_ ^ 1;
            PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(bufferIdGlobal_[0]);
        }
        Preload();
        WaitDispatch();
        SumToServer();
        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeCombineA2Impl
#endif  // MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
