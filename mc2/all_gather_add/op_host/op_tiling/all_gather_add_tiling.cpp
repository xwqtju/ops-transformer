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
 * \file all_gather_add_tiling.cc
 * \brief
 */
#include "vector"
#include "mc2_hcom_topo_info.h"
#include "tiling/add_formulaic_tiling.h"
#include "all_gather_formulaic_tiling.h"
#include "mc2_log.h"
#include "ops_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/all_gather_add_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK = {
    {0, {2, 4, 8}},
    {1, {2, 4, 8, 16, 32}}
    };
constexpr uint32_t TILINGKEY_BIAS = 1U;
constexpr uint32_t TILINGKEY_ND2NZ = 10U;
constexpr uint32_t TILINGKEY_FULL_MESH = 100U;

static void PrintTilingData(::TCubeTiling& tiling)
{
    OP_LOGD("AllGatherAdd", " tiling.usedCoreNum %d", tiling.usedCoreNum);
    OP_LOGD("AllGatherAdd", " tiling.M %d", tiling.M);
    OP_LOGD("AllGatherAdd", " tiling.N %d", tiling.N);
    OP_LOGD("AllGatherAdd", " tiling.Ka %d", tiling.Ka);
    OP_LOGD("AllGatherAdd", " tiling.Kb %d", tiling.Kb);
    OP_LOGD("AllGatherAdd", " tiling.singleCoreM %d", tiling.singleCoreM);
    OP_LOGD("AllGatherAdd", " tiling.singleCoreN %d", tiling.singleCoreN);
    OP_LOGD("AllGatherAdd", " tiling.singleCoreK %d", tiling.singleCoreK);
    OP_LOGD("AllGatherAdd", " tiling.baseM %d", tiling.baseM);
    OP_LOGD("AllGatherAdd", " tiling.baseN %d", tiling.baseN);
    OP_LOGD("AllGatherAdd", " tiling.baseK %d", tiling.baseK);
    OP_LOGD("AllGatherAdd", " tiling.depthA1 %d", tiling.depthA1);
    OP_LOGD("AllGatherAdd", " tiling.depthB1 %d", tiling.depthB1);
    OP_LOGD("AllGatherAdd", " tiling.stepM %d", tiling.stepM);
    OP_LOGD("AllGatherAdd", " tiling.stepN %d", tiling.stepN);
    OP_LOGD("AllGatherAdd", " tiling.stepka %d", tiling.stepKa);
    OP_LOGD("AllGatherAdd", " tiling.stepkb %d", tiling.stepKb);
    OP_LOGD("AllGatherAdd", " tiling.isBias %d", tiling.isBias);
    OP_LOGD("AllGatherAdd", " tiling.transLength %d", tiling.transLength);
    OP_LOGD("AllGatherAdd", " tiling.iterateOrder %s", ((tiling.iterateOrder == 1) ? "orderM" : "orderN"));
    OP_LOGD("AllGatherAdd", " tiling.usedL1Size %d", tiling.shareL1Size);
    OP_LOGD("AllGatherAdd", " tiling.usedL0CSize %d", tiling.shareL0CSize);
    OP_LOGD("AllGatherAdd", " tiling.dbL0C %d", tiling.dbL0C);
    OP_LOGD("AllGatherAdd", " tiling.usedUBSize %d", tiling.shareUbSize);
    OP_LOGD("AllGatherAdd", " tiling.batchM %d", tiling.batchM);
    OP_LOGD("AllGatherAdd", " tiling.batchN %d", tiling.batchN);
    OP_LOGD("AllGatherAdd", " tiling.singleBatchM %d", tiling.singleBatchM);
    OP_LOGD("AllGatherAdd", " tiling.singleBatchN %d", tiling.singleBatchN);
}

static void PrintTilingData(::RCSTiling& rcsTiling)
{
    OP_LOGD("AllGatherAdd", " rcsTiling.commtype %d", rcsTiling.commtype);
    OP_LOGD("AllGatherAdd", " rcsTiling.subtype %d", rcsTiling.subtype);
    OP_LOGD("AllGatherAdd", " rcsTiling.rankDim %d", rcsTiling.rankDim);
    OP_LOGD("AllGatherAdd", " rcsTiling.rankID %d", rcsTiling.rankID);
    OP_LOGD("AllGatherAdd", " rcsTiling.tileCnt %d", rcsTiling.tileCnt);
    OP_LOGD("AllGatherAdd", " rcsTiling.tailM %d", rcsTiling.tailM);
    OP_LOGD("AllGatherAdd", " rcsTiling.tailCnt %d", rcsTiling.tailCnt);
    OP_LOGD("AllGatherAdd", " rcsTiling.isTransA %d", rcsTiling.isTransposeA);
    OP_LOGD("AllGatherAdd", " rcsTiling.isTransB %d", rcsTiling.isTransposeB);
    OP_LOGD("AllGatherAdd", " rcsTiling.rankM %d", rcsTiling.rankM);
    OP_LOGD("AllGatherAdd", " rcsTiling.rankN %d", rcsTiling.rankN);
    OP_LOGD("AllGatherAdd", " rcsTiling.rankK %d", rcsTiling.rankK);
    OP_LOGD("AllGatherAdd", " rcsTiling.gatherIndex %d", rcsTiling.gatherIndex);
    OP_LOGD("AllGatherAdd", " rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OP_LOGD("AllGatherAdd", " rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OP_LOGD("AllGatherAdd", " rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
}

static void PrintTilingData(::TileL2Tiling& tileL2Tiling)
{
    OP_LOGD("AllGatherAdd", " tileL2Tiling.mL2TileCnt %d", tileL2Tiling.mL2TileCnt);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.nL2TileCnt %d", tileL2Tiling.nL2TileCnt);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.mTileBlocks %d", tileL2Tiling.mTileBlocks);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.nTileBlocks %d", tileL2Tiling.nTileBlocks);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.mTailBlocks %d", tileL2Tiling.mTailBlocks);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.nTailBlocks %d", tileL2Tiling.nTailBlocks);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.rankTileNum %d", tileL2Tiling.rankTileNum);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.calcOrder %d", tileL2Tiling.calcOrder);
    OP_LOGD("AllGatherAdd", " tileL2Tiling.enableL2Tile %d", tileL2Tiling.enableL2Tile);
}
}

namespace optiling {

static ge::graphStatus CalcAddTiling(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling, ::TileL2Tiling &l2Tiling);

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, AllGatherAddTilingData& tilingData, mc2tiling::TilingArgs& args);

static uint32_t MC2_Splite(mc2tiling::TilingArgs& args, uint32_t maxTileCnt = 64)
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue/ args.commTurn;
    }

    if (args.inputDtypeSize == 2) { // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64); // align size
    } else if (args.inputDtypeSize == 4) { // 4 is float32 type size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // align size
    }
    if (args.mValue > tileLen) {
        return tileLen;
    }
    return args.mValue;
}

static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context)
{
    OP_TILING_CHECK(mc2tiling::Mc2TilingUtils::CommonParamCheck(context) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "common check failed"), return ge::GRAPH_FAILED);

    const gert::StorageShape* aShape = context->GetInputShape(0);
    uint64_t valueOne = aShape->GetStorageShape().GetDim(0);
    uint64_t valueTwo = aShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(valueOne == 0 || valueTwo == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "the value is invalid"), return ge::GRAPH_FAILED);
    
    if (context->GetAttrs() == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get attrs failed");
    } else {
        auto gather_index = context->GetAttrs()->GetAttrPointer<int>(3);
        OP_TILING_CHECK(*gather_index != 0,
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
        "the gather_index should be 0, but real value is %d", *gather_index), return ge::GRAPH_FAILED);

        auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(1);
        OP_TILING_CHECK(*isTransA != false,
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "the isTransA should be false, but real value is 1"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((valueTwo < KVALUE_MIN || valueTwo >= KVALUE_MAX),
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "The k-axis should be in range[256, 65535), but it is: %lu.", valueTwo), return ge::GRAPH_FAILED);
    }
    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OP_TILING_CHECK(group == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group is nullptr. "),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommAlg(AllGatherAddTilingData &tilingData)
{
    tilingData.socParam.commAlg = COMM_ALG_FULL_MESH;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAllGatherFormulateTileCnt(const gert::TilingContext* ctx,
    AllGatherAddTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    if (ctx->GetAttrs() == nullptr) {
        OP_LOGW(ctx->GetNodeName(), " ctx->GetAttrs is nullptr.");
        return ge::GRAPH_FAILED;
    }

    SocVersion inputSocVersion = (tilingData.socParam.isA3 == 0) ? SocVersion::SOC910_B : SocVersion::SOC910_93;
    
    AllGatherPlusMM tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, inputSocVersion);
    tileFormulate.GetTiling();
    CutResult mCutGather = tileFormulate.tilingM_.cutRes;
    tilingData.param.tileCnt = mCutGather.numLongTile;
    args.mValue = mCutGather.longTileLen;
    CalcAddTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    args.baseMLimit = mCutGather.longTileLen;
    args.mValue = mCutGather.longTileLen * args.rankTileNum;
    tilingData.param.tailM = mCutGather.shortTileLen;
    tilingData.param.tailCnt = 0;
    if (mCutGather.numShortTile > 0) {
        args.mValue = mCutGather.shortTileLen;
        tilingData.param.tailM = args.mValue;
        tilingData.param.tailCnt = mCutGather.numShortTile;
        CalcAddTiling(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        args.baseMLimit = mCutGather.shortTileLen;
        args.mValue = mCutGather.shortTileLen * args.rankTileNum;
    }
    args.mValue = mCutGather.longTileLen;
    return ge::GRAPH_SUCCESS;
}

// 第一个参数m
static ge::graphStatus MCSpliteM(gert::TilingContext* ctx, AllGatherAddTilingData& tilingData,
                                 mc2tiling::TilingArgs& args)
{
    args.rankTileNum = args.rankDim - 1;
    // cmdType = HCCL_CMD_ALLGATHER, 是允许切K
    if (args.enableSplitK) { // 只有1份
        tilingData.param.tileCnt = 1;
        tilingData.param.tailCnt = 0;
        tilingData.param.tailM = 0;

        CalcAddTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    } else if (args.commTurn != 0) {
        uint64_t splite = MC2_Splite(args);

        // 现在找到1个合适的切分
        auto tileCnt = args.mValue / splite; // 切的份数
        auto tileTail = args.mValue % splite; // 尾巴

        tilingData.param.tileCnt = tileCnt;
        args.mValue = splite;
        tilingData.param.tailCnt = 0;
        CalcAddTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
        tilingData.param.tailM = tileTail;
        if (tileTail != 0) {
            args.mValue = tileTail;
            tilingData.param.tailCnt = 1;
            CalcAddTiling(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        }
        args.mValue = splite;
    } else {
        GetAllGatherFormulateTileCnt(ctx, tilingData, args);
    }
    MC2SetWorkspace(ctx, tilingData, args);

    return ge::GRAPH_SUCCESS;
}

static void UpdateTilingKey(uint32_t& tilingKey, AllGatherAddTilingData& tilingData, bool isBias)
{
    tilingKey += isBias ? TILINGKEY_BIAS : 0;
    tilingKey += (tilingData.socParam.isND2NZ == 1) ? TILINGKEY_ND2NZ : 0;
    tilingKey += (tilingData.socParam.commAlg == COMM_ALG_FULL_MESH) ? TILINGKEY_FULL_MESH : 0;
}

static ge::graphStatus SetAddTilingAllGatherAdd(gert::TilingContext* context,
                                                      AllGatherAddTilingData& tilingData,
                                                      mc2tiling::TilingArgs& args)
{
    ge::DataType  biasType;
    bool isBias = true;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAic();
    auto aType = context->GetInputTensor(0)->GetDataType();
    auto bType = context->GetInputTensor(1)->GetDataType();
    auto cType = aType;
    const gert::StorageShape* matrix_bias = context->GetOptionalInputShape(2);
    if (matrix_bias == nullptr) {
        isBias = false;
        biasType = cType;
    }
    else {
        biasType = context->GetInputTensor(2)->GetDataType(); // 2 is index
    }

    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    uint64_t mValue = aShape->GetStorageShape().GetDim(0);
    uint64_t kValue = aShape->GetStorageShape().GetDim(1);
    uint64_t nValue = bShape->GetStorageShape().GetDim(1);

    if (aShape->GetStorageShape().GetDim(1) != bShape->GetStorageShape().GetDim(0)) {
        OP_LOGD(context->GetNodeName(), "A.shape(1) %lu B.shape(0) %lu, istransB = %d",
                aShape->GetStorageShape().GetDim(1), bShape->GetStorageShape().GetDim(0), args.isBTrans);
        nValue = bShape->GetStorageShape().GetDim(0);
    }

    uint64_t inputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(aType);
    uint64_t outputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(cType);

    tilingData.param.rankM = mValue; // 存放用户原始输入的mValue
    tilingData.param.rankN = nValue; // 存放用户原始输入的nValue
    tilingData.param.rankK = kValue; // 存放用户原始输入的kValue
    tilingData.param.aicCoreNum = coreNum;

    args.orgMValue = mValue;
    args.orgNValue = nValue;
    args.orgKValue = kValue;
    args.mValue = mValue;
    args.nValue = nValue;
    args.kValue = kValue;
    args.baseMLimit = -1;
    args.inputDtypeSize = inputDtypeSize;
    args.outputDtypeSize = outputDtypeSize;
    args.aicCoreNum = coreNum;
    args.enablePad = false;
    args.enableSplitK = false;
    args.isBias = isBias;
    args.geAType = aType;
    args.geBType = bType;
    args.geCType = cType;
    args.geBiasType = biasType;
    args.aType = mc2tiling::D_TYPE_MAP.at(aType);
    args.bType = mc2tiling::D_TYPE_MAP.at(bType);
    args.cType = mc2tiling::D_TYPE_MAP.at(cType);
    args.biasType = mc2tiling::D_TYPE_MAP.at(biasType); // 因为bias可能不存在，先采用biasType规避

    // 为通信而进行调整搬运
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        // 先计算出自己的Tiling
        args.rankTileNum = 1; // 1: local matrix not tile
        args.isLocal = true;
        CalcAddTiling(args, tilingData.localTiling, tilingData.localL2Tiling);
        if (tilingData.param.rankID == 0) {
            PrintTilingData(tilingData.localTiling);
            PrintTilingData(tilingData.localL2Tiling);
        }
    } else {
      OP_LOGE(context->GetNodeName(), "args.cmdType error %d", static_cast<int>(args.cmdType));
      return ge::GRAPH_FAILED;
    }

    // 本卡一次计算完,其他卡数据按照DR搬运
    if ((tilingData.socParam.commAlg == COMM_ALG_DOUBLE_RING) && (tilingData.socParam.isStep == 1)) {
        args.mValue /= DOUBLE_RING_FACTOR;
        OP_LOGI(context->GetNodeName(), " args.mValue is set to be %lu under double ring + step communication algorithm.",
            args.mValue);
    }

    args.isLocal = false;

    MCSpliteM(context, tilingData, args);

    uint32_t tilingKey = 0U;
    UpdateTilingKey(tilingKey, tilingData, isBias);     // 当前GetTilingKey函数中使用了Mc2Msg结构体，因而无法归一化，此处使用自己的tilingkey计算函数，确保计算逻辑与旧的key保持一致
    OP_LOGD(context->GetNodeName(), "tilingKey is %u, aicCoreNum is %lu.", tilingKey, args.aicCoreNum);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(args.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcAddTiling(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling, ::TileL2Tiling &l2Tiling)
{
    uint64_t mValue = args.mValue;
    uint64_t nValue = args.nValue;
    uint64_t kValue = args.kValue;

    add_tiling::MultiCoreAddTiling mm;
    mm.SetAType(add_tiling::TPosition::GM, add_tiling::CubeFormat::ND, args.aType, args.isATrans);
    mm.SetBType(add_tiling::TPosition::GM, add_tiling::CubeFormat::ND, args.bType, args.isBTrans);
    mm.SetCType(add_tiling::TPosition::GM, add_tiling::CubeFormat::ND, args.cType);
    if (args.isBias) {
        mm.SetBiasType(add_tiling::TPosition::GM, add_tiling::CubeFormat::ND, args.biasType);
        mm.SetBias(true);
    }
    else {
        mm.SetBias(false);
    }
    mm.SetDim(args.aicCoreNum);
    mm.SetShape(mValue, nValue, kValue);
    mm.SetOrgShape(mValue, nValue, kValue);
    mm.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    mm.SetSingleShape(-1, -1, -1);
    if (nValue == 0) {
        cubeTiling.M = mValue;
        cubeTiling.N = nValue;
        cubeTiling.Ka = kValue;
        cubeTiling.Kb = kValue;
    } else {
        if (mm.GetTiling(cubeTiling) == -1) {
            OP_LOGE("AllGatherAdd", "mValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu",
                    mValue, nValue, kValue, args.aicCoreNum);
            return ge::GRAPH_FAILED;
        }
    }
    mc2tiling::AddFormulaicTiling gatherTiling("AllGatherAdd");
    gatherTiling.GetCubeTiling(args, cubeTiling, l2Tiling);
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetStorage_a(AllGatherAddTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    constexpr uint64_t alignAddrLen = 512;
    auto&& cfg = tilingData.param;
    uint32_t gatherIndex = cfg.gatherIndex;
    uint64_t nd2nzLen = 0;
    uint64_t storage_a = 0;

    // step1: ND2NZ
    if (gatherIndex == 0) { // 转置B
        // 计算ND2NZ需使用空间方法保持与MMV3 tiling计算逻辑一致
        uint64_t alignByte = 256 / args.inputDtypeSize;  // 256B 对齐shape
        uint64_t kALign = OpsUtils::CeilAlign(static_cast<uint64_t>(cfg.rankK), alignByte);
        uint64_t nALign = OpsUtils::CeilAlign(static_cast<uint64_t>(cfg.rankN), alignByte);
        nd2nzLen = kALign * nALign * args.inputDtypeSize;
    }
    else {
        auto alignM = cfg.rankM + 16;
        auto alignK = cfg.rankK + 16;
        nd2nzLen = mc2tiling::AlignUp(alignM * alignK * args.inputDtypeSize, alignAddrLen);
    }

    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        uint64_t gmcFloat = 0; // allgatherMm 通信后数据只需放在gatherLen对应的workspace或者gatherout中，不需要gmcFloat
        uint64_t gatherLen = 0;
        if (args.isStorageGather == false) {
            if (gatherIndex == 0) { // A矩阵
                gatherLen = mc2tiling::AlignUp(cfg.rankM * cfg.rankK * args.inputDtypeSize, alignAddrLen);
            }
            else {
                gatherLen = mc2tiling::AlignUp(cfg.rankK * cfg.rankN * args.inputDtypeSize, alignAddrLen);
            }
            gatherLen *= cfg.rankDim;
        }

        tilingData.param.nd2NzWorkLen = nd2nzLen;
        tilingData.param.cToFloatLen = gmcFloat;
        tilingData.param.gatherLen = gatherLen;

        storage_a = nd2nzLen + gmcFloat + gatherLen; // 需要计算存放的A矩阵
    }
    return storage_a;
}

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};
struct KFCNotify {
    // 消息通信
    HcclAicpuOpParam msgSend[16]; // 填充16个
    HcclAicpuOpParam msgCnt[16];
};

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, AllGatherAddTilingData& tilingData,
                                       mc2tiling::TilingArgs& args)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    uint64_t storage_a = GetStorage_a(tilingData, args);

    int biasLen = 0;
    if (args.isBias) {
        biasLen = mc2tiling::AlignUp(args.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    tilingData.param.biasLen = biasLen;
    workspaces[0] = storage_a + 16 * 1024 * 1024 + biasLen; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("AllGatherAdd", "workspaces[0] size is %ld.", workspaces[0]);
    OP_LOGD("AllGatherAdd", "biasLen is %d.", biasLen);

    tilingData.param.dataType = static_cast<uint8_t>(mc2tiling::Mc2TilingUtils::GetDataType(args.geAType));

    if (tilingData.param.rankID == 0) {
        OP_LOGD("AllGatherAdd", "workspace size %ld.", workspaces[0]);

        PrintTilingData(tilingData.param);
        PrintTilingData(tilingData.tileTiling);
        PrintTilingData(tilingData.tileL2Tiling);
        if (tilingData.param.tailM != 0U) {
            OP_LOGD("AllGatherAdd", "have tail.");
            PrintTilingData(tilingData.tailTiling);
            PrintTilingData(tilingData.tailL2Tiling);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static bool NeedGatherOut(const gert::TilingContext* context) {
  const gert::StorageShape* gatherOut = context->GetOutputShape(1);
  int64_t mulGatherShape = 1;
  if (gatherOut != nullptr) {
    for (unsigned int i = 0;i < gatherOut->GetStorageShape().GetDimNum(); i++) {
        mulGatherShape = mulGatherShape * gatherOut->GetStorageShape().GetDim(i);
        OP_LOGD("AllGatherAdd", "gatherOut StorageShape[%d] is %ld", i, gatherOut->GetStorageShape().GetDim(i));
    }
  }

  if (gatherOut == nullptr || mulGatherShape == 0) {
    return false;
  } else {
    return true;
  }
}

static void SetSocParam(AllGatherAddTilingData* tilingData, const char* group)
{
  auto commSets = mc2tiling::Mc2TilingUtils::GetCommSets(group);
  tilingData->socParam.isA3 = (commSets == mc2tiling::COMM_MESH) ? 0 : 1; 
  tilingData->socParam.isStep = 0U;   
  tilingData->socParam.isND2NZ = 1U; 
}

static void InitHcclParam(AllGatherAddTilingData* tilingData, const char* group)
{
  std::string algConfig = (tilingData->socParam.isA3 == 0) ?
    "AllGather=level0:fullmesh" : "AllGather=level0:doublering";
  Mc2CcTilingConfig mc2CcTilingConfig(group, tilingData->param.commtype, algConfig);
  uint8_t skipBufferWindowCopy = (tilingData->param.gatherLen == 0) ? 
                                 static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                 static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
  mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
  mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
  mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
}

static ge::graphStatus AllGatherAddTilingFunc(gert::TilingContext *context) {
  // 对参数进行校验
  int index = 0;
  AllGatherAddTilingData* tilingData = context->GetTilingData<AllGatherAddTilingData>();
  mc2tiling::TilingArgs args;
  auto group = context->GetAttrs()->GetAttrPointer<char>(index++);
  OP_TILING_CHECK(AllGatherParamsCheck(context) != ge::GRAPH_SUCCESS,
                VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "param is invalid"), return ge::GRAPH_FAILED);

  auto is_trans_a = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto is_trans_b = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto gather_index = context->GetAttrs()->GetAttrPointer<int>(index++);
  auto comm_turn = *context->GetAttrs()->GetAttrPointer<int>(index++);

  auto rankSize = mc2tiling::AddFormulaicTiling::GetRankSize(group);
  OP_TILING_CHECK(comm_turn != 0, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
      "comm_turn should be 0, but the actual value is %d.", comm_turn), return ge::GRAPH_FAILED);

  OP_LOGD("AllGatherAdd"," group is %s, rankSize is %u, is_trans_a is %d, is_trans_b is %d, gather_index is %d,"
          "comm_turn is %d.", group, rankSize, *is_trans_a, *is_trans_b, *gather_index, comm_turn);
  tilingData->param.rankDim = rankSize;
  tilingData->param.isTransposeA = is_trans_a ? *is_trans_a : 0;
  tilingData->param.isTransposeB = is_trans_b ? *is_trans_b : 0;
  tilingData->param.gatherIndex = gather_index ? *gather_index : 0;
  tilingData->param.commtype = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER);
  tilingData->param.subtype = 0;
  tilingData->param.storageGather = 0;
  SetSocParam(tilingData, group);

  OP_TILING_CHECK(SetCommAlg(*tilingData) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), " Set comm algorithm failed."), return ge::GRAPH_FAILED);
  OP_LOGI(context->GetNodeName(), " Communication algorithm is %u.", tilingData->socParam.commAlg);
  
  // distinguish between 910A2 and 910A3
  auto it = std::find(VALID_RANK.at(tilingData->socParam.isA3).begin(),
  VALID_RANK.at(tilingData->socParam.isA3).end(), rankSize);
  OP_TILING_CHECK(it == VALID_RANK.at(tilingData->socParam.isA3).end(),
    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
    "world_size value is %u, which is illegal.", rankSize), return ge::GRAPH_FAILED);

  args.isATrans = is_trans_a ? *is_trans_a : 0;
  args.isBTrans = is_trans_b ? *is_trans_b : 0;
  args.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER;
  args.rankDim = rankSize;
  args.commTurn = comm_turn;
  args.commAlg = tilingData->socParam.commAlg;

  if (NeedGatherOut(context)) {
    args.isStorageGather = true;
    tilingData->param.storageGather = 1;
  } else {
    args.isStorageGather = false;
  }

  SetAddTilingAllGatherAdd(context, *tilingData, args);
  InitHcclParam(tilingData, group);
  return ge::GRAPH_SUCCESS;
}

struct AllGatherAddCompileInfo {};
static ge::graphStatus TilingParseForAllGatherAdd([[maybe_unused]] gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(AllGatherAdd)
    .Tiling(AllGatherAddTilingFunc)
    .TilingParse<AllGatherAddCompileInfo>(TilingParseForAllGatherAdd);
}  // namespace optiling

