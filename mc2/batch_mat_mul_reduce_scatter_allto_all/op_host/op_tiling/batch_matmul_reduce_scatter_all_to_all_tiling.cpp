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
 * \file batch_matmul_reduce_scatter_all_to_all_tiling.cc
 * \brief
 */

#include "batch_matmul_reduce_scatter_all_to_all_tiling.h"

#include <string>
#include <queue>
#include <vector>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

#include "mc2_tiling_common_var.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "batch_matmul_reduce_scatter_all_to_all_formulaic_tiling.h"
#include "op_mc2.h"
#include "mc2_moe_utils.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Moe;

namespace {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t OUTPUT_Y_INDEX = 0;

constexpr uint32_t ATTR_EP_GROUP_INDEX = 0;
constexpr uint32_t ATTR_TP_GROUP_INDEX = 1;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 2;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 3;
constexpr uint32_t ATTR_Y_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_IS_WEIGHT_TRANS_INDEX = 5;

constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7;    // numeric representation of ReduceScatter
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;        // numeric representation of AlltoAll

constexpr uint64_t INIT_TILINGKEY = 1000000000000000000;

constexpr int64_t LITE_THRESHOLD = 640;

constexpr uint64_t TILINGKEY_Y_SHARD = 1U;          // When y_shard = 1
constexpr uint64_t TILINGKEY_WEIGHT_TRANSPOSE = 10U;        // When weight trans is true
constexpr uint64_t TILINGKEY_USE_BIAS = 100U;         // When bias is given
constexpr uint64_t TILINGKEY_LITE = 1000U;           // When C/tp <= LITE_THRESHOLD

}

namespace optiling {

struct BmmTilingConfig {
    gert::TilingContext* context;
    BatchMatMulReduceScatterAlltoAllTilingData& tilingData;
    ReduceScatterAlltoAllBatchInfo& bmmV3BatchInfo;
    ReduceScatterAlltoAllMatmulInfo& mmV3ArgsInfo;
};

static void PrintCommonTilingVariables(BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - epGroupSize is %u.", tilingData.commonTiling.get_epGroupSize());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - tpGroupSize is %u.", tilingData.commonTiling.get_tpGroupSize());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - expert is %lu.", tilingData.commonTiling.get_expert());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - EOverEp is %lu.", tilingData.commonTiling.get_EOverEp());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - C is %lu.", tilingData.commonTiling.get_C());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - COverTp is %lu.", tilingData.commonTiling.get_COverTp());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - H is %lu.", tilingData.commonTiling.get_H());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - HOverTp is %lu.", tilingData.commonTiling.get_HOverTp());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - MOverTp is %lu.", tilingData.commonTiling.get_MOverTp());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - aivCoreNum is %u.", tilingData.commonTiling.get_aivCoreNum());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - inputDatasize is %u.", tilingData.commonTiling.get_inputDatasize());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - biasDatasize is %u.", tilingData.commonTiling.get_biasDatasize());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - ubCapacityForAdd is %lu.", tilingData.commonTiling.get_ubCapacityForAdd());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - totalUbSize is %lu.", tilingData.commonTiling.get_totalUbSize());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - isBias is %d.", tilingData.commonTiling.get_isBias());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - isWeightTrans is %d.", tilingData.commonTiling.get_isWeightTrans());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Common tiling - yShardFlag is %u.", tilingData.commonTiling.get_yShardFlag());
}

static void PrintSliceTileInfo(BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tile length is %lu.", tilingData.commonTiling.localTileE.get_tileLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tile Count is %lu.", tilingData.commonTiling.localTileE.get_tileCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tail length is %lu.", tilingData.commonTiling.localTileE.get_tailLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local E - Tail Count is %lu.", tilingData.commonTiling.localTileE.get_tailCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tile length is %lu.", tilingData.commonTiling.localTileC.get_tileLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tile Count is %lu.", tilingData.commonTiling.localTileC.get_tileCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tail length is %lu.", tilingData.commonTiling.localTileC.get_tailLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Local C - Tail Count is %lu.", tilingData.commonTiling.localTileC.get_tailCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tile length is %lu.", tilingData.commonTiling.domesticTileE.get_tileLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tile Count is %lu.", tilingData.commonTiling.domesticTileE.get_tileCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tail length is %lu.", tilingData.commonTiling.domesticTileE.get_tailLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local E - Tail Count is %lu.", tilingData.commonTiling.domesticTileE.get_tailCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tile length is %lu.", tilingData.commonTiling.domesticTileC.get_tileLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tile Count is %lu.", tilingData.commonTiling.domesticTileC.get_tileCnt());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tail length is %lu.", tilingData.commonTiling.domesticTileC.get_tailLen());
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "Non-Local C - Tail Count is %lu.", tilingData.commonTiling.domesticTileC.get_tailCnt());
}

static matmul_tiling::DataType GetMatMulTilingDataType(ge::DataType geDtype)
{
    if (geDtype == ge::DT_BF16) {
        return matmul_tiling::DataType::DT_BFLOAT16;
    } else if (geDtype == ge::DT_FLOAT16) {
        return matmul_tiling::DataType::DT_FLOAT16;
    } else {
        return matmul_tiling::DataType::DT_FLOAT;
    }
}

static uint32_t GetDataSize(ge::DataType geDtype)
{
    if (geDtype == ge::DT_BF16 || geDtype == ge::DT_FLOAT16) {
        return FP16_DATASIZE;
    } else {
        return FP32_DATASIZE;
    }
}

static void InitTileInfo(BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    if (tilingData.commonTiling.get_yShardFlag() == 1U){
        tilingData.commonTiling.localTileC.set_tileLen(tilingData.commonTiling.get_C() / tilingData.commonTiling.get_tpGroupSize());
    } else {
        tilingData.commonTiling.localTileC.set_tileLen(tilingData.commonTiling.get_C());
    }
    tilingData.commonTiling.localTileC.set_tileCnt(1U);
    tilingData.commonTiling.localTileC.set_tailLen(0U);
    tilingData.commonTiling.localTileC.set_tailCnt(0U);

    tilingData.commonTiling.localTileE.set_tileLen(tilingData.commonTiling.get_EOverEp());
    tilingData.commonTiling.localTileE.set_tileCnt(1U);
    tilingData.commonTiling.localTileE.set_tailLen(0U);
    tilingData.commonTiling.localTileE.set_tailCnt(0U);

    if (tilingData.commonTiling.get_yShardFlag() == 1U){
        tilingData.commonTiling.domesticTileC.set_tileLen(tilingData.commonTiling.get_C() / tilingData.commonTiling.get_tpGroupSize());
    } else {
        tilingData.commonTiling.domesticTileC.set_tileLen(tilingData.commonTiling.get_C());
    }
    tilingData.commonTiling.domesticTileC.set_tileCnt(1U);
    tilingData.commonTiling.domesticTileC.set_tailLen(0U);
    tilingData.commonTiling.domesticTileC.set_tailCnt(0U);

    tilingData.commonTiling.domesticTileE.set_tileLen(tilingData.commonTiling.get_EOverEp());
    tilingData.commonTiling.domesticTileE.set_tileCnt(1U);
    tilingData.commonTiling.domesticTileE.set_tailLen(0U);
    tilingData.commonTiling.domesticTileE.set_tailCnt(0U);
}

static ge::graphStatus CalculateMaxSplitUB(int64_t ubSize, bool bias_flag, bool xCastFlag, bool biasCastFlag,
    BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    uint64_t ubCapacity = 1U;
    uint64_t addLeft = 2U;
    uint64_t addLeftCast = 0U;
    uint64_t addRight = 0U;
    uint64_t addRightCast = 0U;
    uint64_t addOut = 0U;
    uint64_t coeff = 1U;

    if (bias_flag) {
        addRight = FP16_DATASIZE;
        addOut = FP16_DATASIZE;
    }
    if (xCastFlag) {
        addLeftCast = FP32_DATASIZE;
        if (bias_flag) {
            addOut = FP32_DATASIZE;
        }
    }
    if (biasCastFlag) {
        addRightCast = FP32_DATASIZE;
        addOut = FP32_DATASIZE;
    }

    coeff = addLeft + addLeftCast + addRight + addRightCast + addOut;
    ubCapacity = ubSize / coeff;
    ubCapacity = ubCapacity / ALIGN16 * ALIGN16;
    tilingData.commonTiling.set_ubCapacityForAdd(ubCapacity);
    return ge::GRAPH_SUCCESS;
}

static void UpdateTilingKey(uint64_t& tilingKey, BatchMatMulReduceScatterAlltoAllTilingData &tilingData,
    bool isLite)
{
    tilingKey += (tilingData.commonTiling.get_yShardFlag() == 1) ? TILINGKEY_Y_SHARD : 0;
    tilingKey += (tilingData.commonTiling.get_isWeightTrans() == true) ? TILINGKEY_WEIGHT_TRANSPOSE : 0;
    tilingKey += (tilingData.commonTiling.get_isBias() == true) ? TILINGKEY_USE_BIAS : 0;
    tilingKey += isLite ? TILINGKEY_LITE : 0; 

    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "The final tiling Key is: %lu!", tilingKey);
    return;
}

static void CompleteBmmStructs(ReduceScatterAlltoAllBatchInfo &BMMV3BatchInfo,
                               ReduceScatterAlltoAllMatmulInfo &MMV3ArgsInfo,
                               uint32_t m, uint32_t n, uint32_t k, uint32_t b)
{
    MMV3ArgsInfo.mValue = m;
    MMV3ArgsInfo.nValue = n;
    MMV3ArgsInfo.kValue = k;
    BMMV3BatchInfo.batchA = b;
    BMMV3BatchInfo.batchB = b;
    BMMV3BatchInfo.batchC = b;
    BMMV3BatchInfo.batchA3 = b;
    BMMV3BatchInfo.batchB3 = b;
    BMMV3BatchInfo.batchC3 = b;
    return;
}

static void GetBatchMatMulReduceScatterAlltoAllFormulateTileCnt(mc2tiling::TilingArgs& formulaicArgs,
    BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    ReduceScatterAll2AllBMMShardH formulaicTiling(formulaicArgs, tilingData.commonTiling.get_epGroupSize(),
        tilingData.commonTiling.get_tpGroupSize(), tilingData.commonTiling.get_EOverEp());
    formulaicTiling.GetTiling();

    if (formulaicTiling.tilingC.cutRes.shortTileAtBack) {
        tilingData.commonTiling.domesticTileC.set_tileLen(formulaicTiling.tilingC.cutRes.longTileLen);
        tilingData.commonTiling.domesticTileC.set_tileCnt(formulaicTiling.tilingC.cutRes.numLongTile);
        tilingData.commonTiling.domesticTileC.set_tailLen(formulaicTiling.tilingC.cutRes.shortTileLen);
        tilingData.commonTiling.domesticTileC.set_tailCnt(formulaicTiling.tilingC.cutRes.numShortTile);
    } else {
        tilingData.commonTiling.domesticTileC.set_tileLen(formulaicTiling.tilingC.cutRes.shortTileLen);
        tilingData.commonTiling.domesticTileC.set_tileCnt(formulaicTiling.tilingC.cutRes.numShortTile);
        tilingData.commonTiling.domesticTileC.set_tailLen(formulaicTiling.tilingC.cutRes.longTileLen);
        tilingData.commonTiling.domesticTileC.set_tailCnt(formulaicTiling.tilingC.cutRes.numLongTile);
    }

    if (formulaicTiling.cutE.shortTileAtBack) {
        tilingData.commonTiling.domesticTileE.set_tileLen(formulaicTiling.cutE.longTileLen);
        tilingData.commonTiling.domesticTileE.set_tileCnt(formulaicTiling.cutE.numLongTile);
        tilingData.commonTiling.domesticTileE.set_tailLen(formulaicTiling.cutE.shortTileLen);
        tilingData.commonTiling.domesticTileE.set_tailCnt(formulaicTiling.cutE.numShortTile);
    } else {
        tilingData.commonTiling.domesticTileE.set_tileLen(formulaicTiling.cutE.shortTileLen);
        tilingData.commonTiling.domesticTileE.set_tileCnt(formulaicTiling.cutE.numShortTile);
        tilingData.commonTiling.domesticTileE.set_tailLen(formulaicTiling.cutE.longTileLen);
        tilingData.commonTiling.domesticTileE.set_tailCnt(formulaicTiling.cutE.numLongTile);
    }

    if (formulaicTiling.localCutE.shortTileAtBack) {
        tilingData.commonTiling.localTileE.set_tileLen(formulaicTiling.localCutE.longTileLen);
        tilingData.commonTiling.localTileE.set_tileCnt(formulaicTiling.localCutE.numLongTile);
        tilingData.commonTiling.localTileE.set_tailLen(formulaicTiling.localCutE.shortTileLen);
        tilingData.commonTiling.localTileE.set_tailCnt(formulaicTiling.localCutE.numShortTile);
    } else {
        tilingData.commonTiling.localTileE.set_tileLen(formulaicTiling.localCutE.shortTileLen);
        tilingData.commonTiling.localTileE.set_tileCnt(formulaicTiling.localCutE.numShortTile);
        tilingData.commonTiling.localTileE.set_tailLen(formulaicTiling.localCutE.longTileLen);
        tilingData.commonTiling.localTileE.set_tailCnt(formulaicTiling.localCutE.numLongTile);
    }
    return;
}

static void GetBatchMatMulReduceScatterAlltoAllFormulateTileCntShard(mc2tiling::TilingArgs& formulaicArgs,
    BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    ReduceScatterAll2AllBMM formulaicTiling(formulaicArgs, tilingData.commonTiling.get_epGroupSize(),
                            tilingData.commonTiling.get_tpGroupSize(), tilingData.commonTiling.get_EOverEp());
    formulaicTiling.GetTiling();

    tilingData.commonTiling.domesticTileC.set_tileLen(formulaicTiling.tilingC.cutRes.longTileLen / tilingData.commonTiling.get_tpGroupSize());
    tilingData.commonTiling.domesticTileC.set_tileCnt(formulaicTiling.tilingC.cutRes.numLongTile);
    tilingData.commonTiling.domesticTileC.set_tailLen(formulaicTiling.tilingC.cutRes.shortTileLen / tilingData.commonTiling.get_tpGroupSize());
    tilingData.commonTiling.domesticTileC.set_tailCnt(formulaicTiling.tilingC.cutRes.numShortTile);

    tilingData.commonTiling.domesticTileE.set_tileLen(formulaicTiling.cutE.longTileLen);
    tilingData.commonTiling.domesticTileE.set_tileCnt(formulaicTiling.cutE.numLongTile);
    tilingData.commonTiling.domesticTileE.set_tailLen(formulaicTiling.cutE.shortTileLen);
    tilingData.commonTiling.domesticTileE.set_tailCnt(formulaicTiling.cutE.numShortTile);

    tilingData.commonTiling.localTileE.set_tileLen(formulaicTiling.localCutE.longTileLen);
    tilingData.commonTiling.localTileE.set_tileCnt(formulaicTiling.localCutE.numLongTile);
    tilingData.commonTiling.localTileE.set_tailLen(formulaicTiling.localCutE.shortTileLen);
    tilingData.commonTiling.localTileE.set_tailCnt(formulaicTiling.localCutE.numShortTile);

    return;
}

static ge::graphStatus DoBmmTiling(BmmTilingConfig& config, uint32_t tileLen, uint32_t batch, bool isLocal, bool isTail) 
{
    CompleteBmmStructs(config.bmmV3BatchInfo, config.mmV3ArgsInfo, tileLen, 
                      config.tilingData.commonTiling.get_H(),
                      config.tilingData.commonTiling.get_MOverTp(), batch);

	auto bmmTilingLambda = [isLocal, isTail, &config]() -> decltype(config.tilingData.domesticTiling.bmmTilingData)&
    {
        if (isLocal && isTail) 
        {
            return config.tilingData.localTailTiling.bmmTilingData;
        }
        if (isLocal) 
        {
            return config.tilingData.localTiling.bmmTilingData;
        }   
        if (isTail) 
        {
            return config.tilingData.domesticTailTiling.bmmTilingData;
        }  
        return config.tilingData.domesticTiling.bmmTilingData;
    };
	auto& bmmTilingData = bmmTilingLambda();

    BatchMatMulReduceScatterAlltoAllTiling bmmTiling(config.context, bmmTilingData, 
                                                   config.bmmV3BatchInfo, config.mmV3ArgsInfo);
    if (bmmTiling.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(config.context->GetNodeName(), "Do BmmV3Tiling failed under shard-1 %s %s section.",
                isLocal ? "local" : "non-local", isTail ? "tail" : "standard");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetMatmulTilingBatchMatMulReduceScatterAlltoAll(BmmTilingConfig& config, mc2tiling::TilingArgs& formulaicArgs, bool isLite)
{
    // 1. 调用公式化tiling接口，获取local和非local块的切分信息并更新commonTiling
    if (config.tilingData.commonTiling.get_yShardFlag() == 0) {
        GetBatchMatMulReduceScatterAlltoAllFormulateTileCnt(formulaicArgs, config.tilingData);
    } else {
        GetBatchMatMulReduceScatterAlltoAllFormulateTileCntShard(formulaicArgs, config.tilingData);
    }

    uint32_t factor = (isLite) ? config.tilingData.commonTiling.get_tpGroupSize() : 1U;
    uint32_t localBatch = (isLite) ? config.tilingData.commonTiling.localTileE.get_tileLen() : 1U;
    uint32_t nonLocalBatch = (isLite) ? config.tilingData.commonTiling.domesticTileE.get_tileLen() : 1U;
    // 2. 将切块信息传入BmmV3 tiling，根据后续整改方案更新Bmm所需的相应参数并调用BmmV3 Tiling接口，获取local/non-local的BMM tiling信息
	// Local standard slice BMM tiling
    if (DoBmmTiling(config, config.tilingData.commonTiling.localTileC.get_tileLen() * factor, localBatch, true, false) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Local tail slice BMM tiling
    if (config.tilingData.commonTiling.localTileC.get_tailLen() != 0) {
        if (DoBmmTiling(config, config.tilingData.commonTiling.localTileC.get_tailLen() * factor, localBatch, true, true) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    factor = (isLite) ? (config.tilingData.commonTiling.get_tpGroupSize() * (config.tilingData.commonTiling.get_epGroupSize() - 1U)) : 1U;
    // Non-local standard slice BMM tiling
    if (DoBmmTiling(config, config.tilingData.commonTiling.domesticTileC.get_tileLen() * factor, nonLocalBatch, false, false) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Non-local tail slice BMM tiling
    if (config.tilingData.commonTiling.domesticTileC.get_tailLen() != 0) {
        if (DoBmmTiling(config, config.tilingData.commonTiling.domesticTileC.get_tailLen() * factor, nonLocalBatch, false, true) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static bool CommonCheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape, const size_t wDimM)
{
    // 检查 < 0 的范围: x dim C >= 1, dim E H M 会在后面拦截
    if (xShape->GetDim(X_DIM_C) < VALUE_C_MIN) {
        OP_LOGE(nodeName, "The second dim of x should not < %ld, but got x[1] = %ld.", VALUE_C_MIN, xShape->GetDim(X_DIM_C));
        return false;
    }

    // x[2]、w[wDimM] 是 M 轴(reduce 轴)，所以需要相等
    if (xShape->GetDim(X_DIM_M) != weightShape->GetDim(wDimM)) {
        OP_LOGE(nodeName, "The last dim of x must equal the corresponding dim of weight, "
            "but got x[2] = %ld, w[%lu] = %ld.", xShape->GetDim(X_DIM_M), wDimM, weightShape->GetDim(wDimM));
        return false;
    }   

    // x[2]、w[wDimM] 是 M 轴，不支持为 0
    if (xShape->GetDim(X_DIM_M) == 0) {
        OP_LOGE(nodeName, "The last dim of x(x[2]) or the corresponding dim of weight = 0 is unsupported.");
        return false;
    }

    return true;
}

// GPT 2.2T
static bool YShardCheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
    const int64_t yShard, const int64_t epSize, const int64_t tpSize, const size_t wDimH, const size_t wDimM)
{
    // 检查 shape 维度的范围
    // value E should = [2, 512], x[DIM_E] = E / Ep
    if ((xShape->GetDim(DIM_E) * epSize < VALUE_E_MIN) || (xShape->GetDim(DIM_E) * epSize > VALUE_E_MAX)) {
        OP_LOGE(nodeName, "Value E should in [%ld, %ld], but got %ld", VALUE_E_MIN, VALUE_E_MAX,
            xShape->GetDim(DIM_E) * epSize);
        return false;
    }
    // w[wDimH] = H, value H should = [1, 65535]
    if ((weightShape->GetDim(wDimH) < VALUE_H_MIN) || (weightShape->GetDim(wDimH) > VALUE_H_MAX)) {
        OP_LOGE(nodeName, "Value H should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
            weightShape->GetDim(wDimH));
        return false;
    }
    // w[wDimM] = M / Tp, its range should same with H, so it meets M / Tp * H <= 65535 * 65535
    if ((weightShape->GetDim(wDimM) < VALUE_H_MIN) || (weightShape->GetDim(wDimM) > VALUE_H_MAX)) {
        OP_LOGE(nodeName, "Value M / tp should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
            weightShape->GetDim(wDimM));
        return false;
    }

    // x[0] = E / Ep, w[0] = E / Ep, 所以两者需要相等
    if (xShape->GetDim(DIM_E) != weightShape->GetDim(DIM_E)) {
        OP_LOGE(nodeName, "The first dim of x must equal the first dim of w, but got x[0] = %ld, w[0] = %ld.",
            xShape->GetDim(DIM_E), weightShape->GetDim(DIM_E));
        return false;
    }

    if (yShard == 0) {
        // x[1] = Ep * C, 所以 x[1] % Ep 需要等于 0
        if (xShape->GetDim(X_DIM_C) % epSize != 0) {
            OP_LOGE(nodeName, "The second dim of x mod epSize must be 0, "
                "but got x[1] = %ld, epSize = %ld.", xShape->GetDim(X_DIM_C), epSize);
            return false;
        }
	} else if (yShard == 1) {
        // x[1] = (c / Tp) * Ep * Tp, 所以 x[1] % (Ep * Tp) 需要等于 0
        if (xShape->GetDim(X_DIM_C) % (epSize * tpSize) != 0) {
            OP_LOGE(nodeName, "The second dim of x mod (epSize * tpSize) must be 0, "
                "but got x[1] = %ld, epSize * tpSize = %ld.", xShape->GetDim(X_DIM_C), epSize * tpSize);
            return false;
        }
	}

    return true;
}

static bool CheckBiasShape(const char *nodeName, const gert::Shape *weightShape, const gert::Shape *biasShape,
    const int64_t yShard, const int64_t tpSize, const size_t wDimH)
{
    // 检查 dimNum
    if ((biasShape->GetDimNum() != SUPPORT_DIM_NUM) && (biasShape->GetDimNum() != BIAS_SUPPORT_DIM_NUM)) {
        OP_LOGE(nodeName, "Dim of input bias must be 2 or 3, but got dim bias %zu.", biasShape->GetDimNum());
        return false;
    }

    // 检查 shape
    if (biasShape->GetDim(0) != weightShape->GetDim(0)) {
        OP_LOGE(nodeName, "The first dim of bias must equal the first dim of weight,"
            "but got bias[0] = %ld, w[0] = %ld.", biasShape->GetDim(0), weightShape->GetDim(0));
        return false;
    }

    size_t biasLastDimValue = 1U; // 默认 bias 是二维，所以最后一维的 index 是 1
    if (biasShape->GetDimNum() == SUPPORT_DIM_NUM) { // 三维
        if (biasShape->GetDim(1) != 1) {
            OP_LOGE(nodeName, "The second dim of bias must be 1 when 3-dim.");
            return false;
        }
        biasLastDimValue = 2; // 三维时候，bias 的最后一维是 2
    }

    if (yShard == 1) {
        if (biasShape->GetDim(biasLastDimValue) != weightShape->GetDim(wDimH)) {
            OP_LOGE(nodeName, "The last dim of bias must equal the corresponding dim of weight, "
                "but got bias[2] = %ld, w[%lu] = %ld.", biasShape->GetDim(biasLastDimValue), wDimH, weightShape->GetDim(wDimH));
            return false;
        }
    } else if (yShard == 0) {
        if (biasShape->GetDim(biasLastDimValue) * tpSize != weightShape->GetDim(wDimH)) {
            OP_LOGE(nodeName, "The last dim of bias (H / tp) must equal the corresponding dim of weight, "
                "but got bias[2] * tp = %ld, w[%lu] = %ld.", biasShape->GetDim(biasLastDimValue) * tpSize, wDimH, weightShape->GetDim(wDimH));
            return false;
        }       
    }

    return true;
}

static bool CheckTensorShape(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape,
    const gert::Shape *biasShape, const int64_t epSize, const int64_t tpSize, const size_t wDimM, const size_t wDimH,
    const int64_t yShard)
{
    if ((xShape == nullptr) || (weightShape == nullptr)) {
        OP_LOGE(nodeName, "xShape or weightShape is nullptr.");
        return false;
    }
    if (!DimNumCheck(nodeName, xShape, weightShape)) {
        OP_LOGE(nodeName, "Dim num check failed.");
        return false;
    }

    if (!CommonCheckTensorShape(nodeName, xShape, weightShape, wDimM)) {
        OP_LOGE(nodeName, "common tensor shape check failed.");
        return false;
    }

    if (yShard == 0 || yShard == 1) {
        if (!YShardCheckTensorShape(nodeName, xShape, weightShape, yShard, epSize, tpSize, wDimH, wDimM)) {
            OP_LOGE(nodeName, "yShard = [%ld] tensor shape check failed.", yShard);
            return false;
        }
    } else {
        OP_LOGE(nodeName, "y shard type [%ld] is currently unsupported.", yShard);
        return false;
    }

    if (biasShape != nullptr) {
        if (!(CheckBiasShape(nodeName, weightShape, biasShape, yShard, tpSize, wDimH))) {
            OP_LOGE(nodeName, "bias shape check failed.");
            return false;
        }
    }

    OP_LOGI(nodeName, "tensor shape check success.");
    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, int64_t &epSize, int64_t &tpSize, int64_t &yShard)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);

    const char *groupEp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_EP));
    const char *groupTp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_TP));
    const int64_t *epPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_EP_WORLD_SIZE));
    const int64_t *tpPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_TP_WORLD_SIZE));
    const int64_t *yPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_Y_SHARD_TYPE));

    if ((epPtr == nullptr) || (tpPtr == nullptr) || (yPtr == nullptr)) {
        OP_LOGE(nodeName, "attrs index in context is in valid or out of range, attrs got nullptr.");
        return false;
    }

    epSize = *epPtr;
    tpSize = *tpPtr;
    yShard = *yPtr;

    if (!GroupCheck(nodeName, groupEp, groupTp)) {
        OP_LOGE(nodeName, "group size check failed.");
        return false;
    }

    if (!EpTpSizeCheck(epSize, tpSize)) {
        OP_LOGE(nodeName, "rank size error, tpSize [%ld], epSize [%ld].", tpSize, epSize);
        return false;
    }

    if ((yShard != 1) && (yShard != 0)) { // 当前仅支持 shard = 0 or 1
        OP_LOGE(nodeName, "y shard type [%ld] is invalid.", yShard);
        return false;
    }

    OP_LOGI(nodeName, "attr info: groupEp %s, groupTp %s, tpSize %ld, epSize %ld, yShard %ld.",
        groupEp, groupTp, tpSize, epSize, yShard);
    return true;
}

static ge::graphStatus TilingCheckBatchMatMulReduceScatterAlltoAll(gert::TilingContext* context)
{
    // 检查 shape 是否为空
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter BmmReduceScatterAlltoAll tiling check impl.");
    const gert::StorageShape *xStorageShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const gert::StorageShape *weightStorageShape = context->GetInputShape(static_cast<size_t>(ops::
        MC2MoeInputIdx::K_WEIGHT));
    OP_TILING_CHECK(weightStorageShape == nullptr, OP_LOGE(nodeName, "weightShape is null."), return ge::GRAPH_FAILED);
    const gert::StorageShape *yStorageShape = context->GetOutputShape(static_cast<size_t>(ops::
        BmmReduceScatterAlltoAllOutIdx::K_Y));
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(nodeName, "yShape is null."), return ge::GRAPH_FAILED);

    // 检查属性
    int64_t epSize = -1;
    int64_t tpSize = -1;
    int64_t yShard = -1;
    if (!CheckAttrs(context, epSize, tpSize, yShard)) {
        OP_LOGE(nodeName, "Attrs check failed.");
        return ge::GRAPH_FAILED;
    }

	// 设 w = [E, M, H], dimH 指 H 轴, dimM 指 M 轴
	size_t wDimM = 1UL; 
    size_t wDimH = 2UL; // 1 2 分别代表 weight 没有转置时候的维度值, 2: H 轴, 1: M 轴
	auto attrs = context->GetAttrs();
    bool isWeightTrans = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));  
	// w_trans = [E, H, M]
    if (isWeightTrans) {
		size_t wDimMTrans = 2UL; 
        size_t wDimHTrans = 1UL;  // 2 1 分别代表 weight 转置时候的维度值, 2: M 轴, 1: H 轴
		wDimM = wDimMTrans; 
    	wDimH = wDimHTrans; 
	}  

    const gert::StorageShape *biasStorageShape = context->GetOptionalInputShape(static_cast<size_t>(ops::
        MC2MoeInputIdx::K_BIAS));
    
    const gert::Shape *xShape = &xStorageShape->GetStorageShape();
    const gert::Shape *weightShape = &weightStorageShape->GetStorageShape();
    const gert::Shape *biasShape = (biasStorageShape == nullptr) ? nullptr : &biasStorageShape->GetStorageShape(); 

    // tensor shape 检查
    if (!CheckTensorShape(nodeName, xShape, weightShape, biasShape, epSize, tpSize, wDimM, wDimH, yShard)) {
        OP_LOGE(nodeName, "Tensor shape check failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    
    // 2EC*h + ECH
    uint64_t commOut = 2UL * tilingData.commonTiling.get_expert() *
                        tilingData.commonTiling.get_C() *
                        tilingData.commonTiling.get_HOverTp() +
                        tilingData.commonTiling.get_expert() *
                        tilingData.commonTiling.get_C() *
                        tilingData.commonTiling.get_H();

    commOut = commOut * static_cast<uint64_t>(tilingData.commonTiling.get_inputDatasize());

    uint64_t maxLocalELen = std::max(tilingData.commonTiling.localTileE.get_tileLen(), tilingData.commonTiling.localTileE.get_tailLen());
    uint64_t maxLocalCLen = std::max(tilingData.commonTiling.localTileC.get_tileLen(), tilingData.commonTiling.localTileC.get_tailLen());
    uint64_t maxNonLocalELen = std::max(tilingData.commonTiling.domesticTileE.get_tileLen(), tilingData.commonTiling.domesticTileE.get_tailLen());
    uint64_t maxNonLocalCLen = std::max(tilingData.commonTiling.domesticTileC.get_tileLen(), tilingData.commonTiling.domesticTileC.get_tailLen());

    uint64_t transOut = std::max(maxLocalELen * maxLocalCLen * tilingData.commonTiling.get_H(),
                                 static_cast<uint64_t>(tilingData.commonTiling.get_epGroupSize() - 1) *
                                 maxNonLocalELen * maxNonLocalCLen * tilingData.commonTiling.get_H());

    transOut = transOut * static_cast<uint64_t>(tilingData.commonTiling.get_inputDatasize());

    workspaces[0] = commOut + transOut + 16 * 1024 * 1024; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "workspaces[0] size %ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MC2SetWorkspaceShard(gert::TilingContext* context,
                                            BatchMatMulReduceScatterAlltoAllTilingData& tilingData, bool isLite)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    
    // 2EcH + ECH
    uint64_t commOut = 2 * static_cast<uint64_t>(tilingData.commonTiling.get_expert()) * \
                        static_cast<uint64_t>(tilingData.commonTiling.get_COverTp()) * \
                        static_cast<uint64_t>(tilingData.commonTiling.get_H()) + \
                        static_cast<uint64_t>(tilingData.commonTiling.get_expert()) * \
                        static_cast<uint64_t>(tilingData.commonTiling.get_C()) * \
                        static_cast<uint64_t>(tilingData.commonTiling.get_H());

    if (isLite) {
        uint64_t localMaxC = std::max(tilingData.commonTiling.localTileC.get_tileLen(),
                                      tilingData.commonTiling.localTileC.get_tailLen());
        uint64_t nonLocalMaxC = std::max(tilingData.commonTiling.domesticTileC.get_tileLen(),
                                         tilingData.commonTiling.domesticTileC.get_tailLen());
        uint64_t localTransBefore = localMaxC * tilingData.commonTiling.localTileE.get_tileLen() *
            tilingData.commonTiling.get_tpGroupSize() * tilingData.commonTiling.get_MOverTp();
        uint64_t nonLocalTransBefore = nonLocalMaxC * tilingData.commonTiling.domesticTileE.get_tileLen() *
            tilingData.commonTiling.get_tpGroupSize() * tilingData.commonTiling.get_MOverTp() *
            (tilingData.commonTiling.get_epGroupSize() - 1U);
        uint64_t localTransAfter = localMaxC * tilingData.commonTiling.localTileE.get_tileLen() *
            tilingData.commonTiling.get_tpGroupSize() * tilingData.commonTiling.get_H();
        uint64_t nonLocalTransAfter = nonLocalMaxC * tilingData.commonTiling.domesticTileE.get_tileLen() *
            tilingData.commonTiling.get_tpGroupSize() * tilingData.commonTiling.get_H() *
            (tilingData.commonTiling.get_epGroupSize() - 1U);
        commOut += std::max(localTransBefore, nonLocalTransBefore) + std::max(localTransAfter, nonLocalTransAfter);
    }

    commOut = commOut * static_cast<uint64_t>(tilingData.commonTiling.get_inputDatasize());

    workspaces[0] = commOut + 16 * 1024 * 1024; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("BatchMatMulReduceScatterAlltoAll", "workspaces[0] size %ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext *context, BatchMatMulReduceScatterAlltoAllTilingData& tilingData)
{
    auto rawTilingData = context->GetRawTilingData();
    OP_TILING_CHECK(rawTilingData == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "GetRawTilingData returned nullptr!"),
                    return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

// 子函数1: 初始化tiling数据结构
static void InitTilingData(BatchMatMulReduceScatterAlltoAllTilingData &tilingData) {
    tilingData.set_version(TWO);
    tilingData.set_hcommCnt(TWO);
    tilingData.commonTiling.set_ubCapacityForAdd(0);
    tilingData.hcommCfgRS.set_skipLocalRankCopy(0);
    tilingData.hcommCfgRS.set_skipBufferWindowCopy(0);
    tilingData.hcommCfgRS.set_stepSize(0);
    tilingData.hcommCfgATA.set_skipLocalRankCopy(0);
    tilingData.hcommCfgATA.set_skipBufferWindowCopy(0);
    tilingData.hcommCfgATA.set_stepSize(0);
}
struct TensorInfo {
    uint64_t inputDatasize = 0; 
    uint64_t biasDatasize = 0;
    ge::DataType inputDatatype = ge::DT_FLOAT; 
    ge::DataType biasDatatype = ge::DT_FLOAT;
    bool isLite = false;  
    const char* epGroup = nullptr; 
    const char* tpGroup = nullptr;
    uint64_t aicCoreNum = 0;    // AIC 核心数
    uint64_t aivCoreNum = 0;    // AIV 核心数
    uint64_t ubSize = 0;        // UB 大小
    uint32_t blockDim=1;
    bool isWeightTrans = false;
    // 存放 tiling 公式所需的结构体参数
    ReduceScatterAlltoAllMatmulInfo mmv3ArgsInfo;    
    ReduceScatterAlltoAllBatchInfo bmmv3BatchInfo;  
};
// 子函数2: 获取平台信息
static void GetPlatformInfo(
    const gert::TilingContext *context,
    TensorInfo &tensorInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    tensorInfo.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tensorInfo.aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tensorInfo.ubSize);
    tensorInfo.blockDim = ascendcPlatform.CalcTschBlockDim(tensorInfo.aivCoreNum, tensorInfo.aicCoreNum, tensorInfo.aivCoreNum);
}
  
// 子函数3: 获取关键参数并计算，填充tilingdata   
static ge::graphStatus CalculateTensorInfo(const gert::TilingContext* context, TensorInfo &tensorInfo,
                                BatchMatMulReduceScatterAlltoAllTilingData &tilingData, const gert::StorageShape* biasInputShape) {
    constexpr int MATMUL_INPUT_M_AXIS = 2;
    const gert::StorageShape* xInputShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape* weightInputShape = context->GetInputShape(WEIGHT_INDEX);
    const gert::StorageShape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    auto attrs = context->GetAttrs();
    int64_t ep = *(attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX));
    int64_t tp = *(attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX));
    auto yShard = attrs->GetAttrPointer<int64_t>(ATTR_Y_SHARD_TYPE_INDEX);
    tensorInfo.isWeightTrans  = *(attrs->GetAttrPointer<bool>(ATTR_IS_WEIGHT_TRANS_INDEX));
    tensorInfo.epGroup = attrs->GetAttrPointer<char>(ATTR_EP_GROUP_INDEX);
    tensorInfo.tpGroup = attrs->GetAttrPointer<char>(ATTR_TP_GROUP_INDEX);
    const char* epGroupStr = tensorInfo.epGroup ? tensorInfo.epGroup : "null";
    const char* tpGroupStr = tensorInfo.tpGroup ? tensorInfo.tpGroup : "null";
    OP_LOGD("BatchMatMulReduceScatterAlltoAll",
            "EP group is %s, ep is %ld, TP group is %s, tp is %ld, weight_trans_flag is %d, y_shard_flag is %ld",
            epGroupStr, ep, tpGroupStr, tp, tensorInfo.isWeightTrans , *yShard);
    int64_t e = xInputShape->GetStorageShape().GetDim(0);
    int64_t E = yShape->GetStorageShape().GetDim(0);
    int64_t C = (*yShard == 1) ? yShape->GetStorageShape().GetDim(1) * tp : yShape->GetStorageShape().GetDim(1);
    int64_t c = (*yShard == 1) ? yShape->GetStorageShape().GetDim(1) : yShape->GetStorageShape().GetDim(1) / tp;
    size_t wDimH = 2U;
    OP_TILING_CHECK(e > MAX_HCCL_HANDLE_LIMIT,
                        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "E/ep[%li] is larger than max hccl handle limit[32]!", e),
                        return ge::GRAPH_FAILED);
    wDimH = tensorInfo.isWeightTrans  ? 1U : 2U;
    int64_t dimH = weightInputShape->GetStorageShape().GetDim(wDimH);
    tensorInfo.inputDatasize = GetDataSize(tensorInfo.inputDatatype);
    tensorInfo.biasDatasize = GetDataSize(tensorInfo.biasDatatype);
    tensorInfo.isLite = (c <= LITE_THRESHOLD) && (*yShard == 1);
    tilingData.commonTiling.set_COverTp(c);
    tilingData.commonTiling.set_C(C);
    tilingData.commonTiling.set_epGroupSize(ep);
    tilingData.commonTiling.set_tpGroupSize(tp);
    tilingData.commonTiling.set_expert(E);
    tilingData.commonTiling.set_EOverEp(e);
    tilingData.commonTiling.set_H(dimH);
    tilingData.commonTiling.set_HOverTp(static_cast<int64_t>(dimH / tp));
    tilingData.commonTiling.set_MOverTp(xInputShape->GetStorageShape().GetDim(MATMUL_INPUT_M_AXIS));//m
    tilingData.commonTiling.set_yShardFlag(*yShard);
    tilingData.commonTiling.set_isBias(biasInputShape == nullptr ? false : true);  
    tilingData.commonTiling.set_inputDatasize(tensorInfo.inputDatasize);
    tilingData.commonTiling.set_biasDatasize(tensorInfo.biasDatasize);
    tilingData.commonTiling.set_aivCoreNum(tensorInfo.aivCoreNum);
    tilingData.commonTiling.set_isWeightTrans(tensorInfo.isWeightTrans);
    tilingData.commonTiling.set_totalUbSize(tensorInfo.ubSize);
    InitTileInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}   

// 子函数4: 设置Hcom配置
static void SetHcomConfig(
    BatchMatMulReduceScatterAlltoAllTilingData &tilingData,
    const char* tpGroup,
    const char* epGroup)
{
    std::string epGroupStr = std::string(epGroup);
    std::string tpGroupStr = std::string(tpGroup);
    const std::string algConfigRSStr = "ReduceScatter=level0:ring";
    const std::string algConfigATAStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    std::vector<char> groupNameVecRS(ARR_LENGTH, '\0');
    for (auto ite = tpGroupStr.begin(); ite != tpGroupStr.end(); ite++) {
        groupNameVecRS[ite - tpGroupStr.begin()] = *ite;
    }

    std::vector<char> groupNameVecATA(ARR_LENGTH, '\0');
    for (auto ite = epGroupStr.begin(); ite != epGroupStr.end(); ite++) {
        groupNameVecATA[ite - epGroupStr.begin()] = *ite;
    }
    std::vector<char> algConfigVecRS(ARR_LENGTH, '\0');
    for (auto ite = algConfigRSStr.begin(); ite != algConfigRSStr.end(); ite++) {
        algConfigVecRS[ite - algConfigRSStr.begin()] = *ite;
    }

    std::vector<char> algConfigVecATA(ARR_LENGTH, '\0');
    for (auto ite = algConfigATAStr.begin(); ite != algConfigATAStr.end(); ite++) {
        algConfigVecATA[ite - algConfigATAStr.begin()] = *ite;
    }

    tilingData.hcommCfgRS.set_groupName(groupNameVecRS.data());
    tilingData.hcommCfgRS.set_algConfig(algConfigVecRS.data());
    tilingData.hcommCfgRS.set_opType(OP_TYPE_REDUCE_SCATTER);        // numeric representation of ReduceScatter
    tilingData.hcommCfgRS.set_reduceType(0);    // numeric representation of sum
    tilingData.hcommCfgATA.set_groupName(groupNameVecATA.data());
    tilingData.hcommCfgATA.set_algConfig(algConfigVecATA.data());
    tilingData.hcommCfgATA.set_opType(OP_TYPE_ALL_TO_ALL);       // numeric representation of AlltoAll
     }

// 子函数5: 设置公式参数
static void SetFormulaicArgs(
    mc2tiling::TilingArgs &formulaicArgs,
    BatchMatMulReduceScatterAlltoAllTilingData &tilingData,
    TensorInfo &tensorInfo)
{   
    auto &MMV3ArgsInfo =tensorInfo.mmv3ArgsInfo;
    auto &BMMV3BatchInfo=tensorInfo.bmmv3BatchInfo;
    MMV3ArgsInfo.opName = "BatchMatMulReduceScatterAlltoAll";
    MMV3ArgsInfo.isWeightTrans = tensorInfo.isWeightTrans;
    MMV3ArgsInfo.isBias = false;                // 出于性能考虑，当前bias计算在外部进行，算子内不会有bias场景
    MMV3ArgsInfo.aType = tensorInfo.inputDatatype;
    MMV3ArgsInfo.bType = tensorInfo.inputDatatype;
    MMV3ArgsInfo.cType = tensorInfo.inputDatatype;
    MMV3ArgsInfo.biasType = tensorInfo.biasDatatype;
    BMMV3BatchInfo.biasWithBatch = false;       // 出于性能考虑，当前bias计算在外部进行，算子内不会有bias场景
    formulaicArgs.mValue = tilingData.commonTiling.get_C();
    formulaicArgs.nValue = tilingData.commonTiling.get_H();
    formulaicArgs.kValue = tilingData.commonTiling.get_MOverTp();
    formulaicArgs.inputDtypeSize = tensorInfo.inputDatasize;
    formulaicArgs.outputDtypeSize = tensorInfo.inputDatasize;
    formulaicArgs.aicCoreNum = tensorInfo.aicCoreNum; 
    formulaicArgs.aType = GetMatMulTilingDataType(tensorInfo.inputDatatype);
    formulaicArgs.bType = GetMatMulTilingDataType(tensorInfo.inputDatatype);
    formulaicArgs.cType = GetMatMulTilingDataType(tensorInfo.inputDatatype);
    formulaicArgs.biasType = GetMatMulTilingDataType(tensorInfo.biasDatatype);
}

// 子函数6: 核心tiling逻辑
static ge::graphStatus ComputeCoreTiling(
    gert::TilingContext *context,
    BatchMatMulReduceScatterAlltoAllTilingData &tilingData,
    mc2tiling::TilingArgs &formulaicArgs,
    TensorInfo &tensorInfo)
{
    bool xCastFlag    = tensorInfo.inputDatatype == ge::DT_BF16;
    bool biasCastFlag = tensorInfo.biasDatatype  == ge::DT_BF16;
    BmmTilingConfig config{context, tilingData, tensorInfo.bmmv3BatchInfo, tensorInfo.mmv3ArgsInfo};	
    // 待修改，等BMM tiling提供接口和修改方案
    OP_TILING_CHECK(SetMatmulTilingBatchMatMulReduceScatterAlltoAll(config, formulaicArgs, tensorInfo.isLite) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set Matmul tiling Failed!"),
                        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CalculateMaxSplitUB(tensorInfo.ubSize, tilingData.commonTiling.get_isBias(),
                        xCastFlag, biasCastFlag, tilingData) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Calculate max split UB Failed!"),
                        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 主函数
static ge::graphStatus BatchMatMulReduceScatterAlltoAllTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(TilingCheckBatchMatMulReduceScatterAlltoAll(context) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Tiling check shape Failed!"),
                    return ge::GRAPH_FAILED);   

    BatchMatMulReduceScatterAlltoAllTilingData tilingData;
    mc2tiling::TilingArgs formulaicArgs;
    ReduceScatterAlltoAllBatchInfo BMMV3BatchInfo;
    ReduceScatterAlltoAllMatmulInfo MMV3ArgsInfo;
    //子函数1 初始化
    InitTilingData(tilingData);  
    TensorInfo tensorInfo{};
    //子函数2 获取aiv,UB
    GetPlatformInfo(context,    tensorInfo);
    uint64_t tilingKey = INIT_TILINGKEY;
    context->SetBlockDim(tensorInfo.blockDim);
    //子函数3 获取参数，计算，填充tilingData
    //存放inputDatatype，biasDatatype
    auto* biasTensor = context->GetOptionalInputTensor(BIAS_INDEX);
    const gert::StorageShape* biasInputShape = biasTensor ? context->GetOptionalInputShape(BIAS_INDEX) : nullptr;
    tensorInfo.inputDatatype = context->GetInputDesc(X_INDEX)->GetDataType();
    tensorInfo.biasDatatype = (biasInputShape == nullptr) ? tensorInfo.inputDatatype : context->GetOptionalInputDesc(BIAS_INDEX)->GetDataType();
    CalculateTensorInfo(context, tensorInfo,tilingData,biasInputShape);
    //子函数4 设置通信
    SetHcomConfig(tilingData, tensorInfo.tpGroup, tensorInfo.epGroup);   
    //子函数5  设置公式参数
    SetFormulaicArgs(formulaicArgs, tilingData, tensorInfo);   
    //子函数6 tiling分配 
    ComputeCoreTiling(context, tilingData, formulaicArgs, tensorInfo);   

    if (tilingData.commonTiling.get_yShardFlag() == 0) {
        OP_TILING_CHECK(MC2SetWorkspace(context, tilingData) != ge::GRAPH_SUCCESS,
                            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set workspace Failed!"),
                            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(MC2SetWorkspaceShard(context, tilingData, tensorInfo.isLite) != ge::GRAPH_SUCCESS,
                            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set workspace Failed!"),
                            return ge::GRAPH_FAILED);
    }
    UpdateTilingKey(tilingKey, tilingData, tensorInfo.isLite);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(tensorInfo.blockDim);
    OP_TILING_CHECK(SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Set Tiling Data Failed!"),
                        return ge::GRAPH_FAILED);
    PrintCommonTilingVariables(tilingData);
    PrintSliceTileInfo(tilingData);
    return ge::GRAPH_SUCCESS;
}
struct BatchMatMulReduceScatterAlltoAllCompileInfo {};
ge::graphStatus TilingParseForBatchMatMulReduceScatterAlltoAll(gert::TilingParseContext *context) { 
	(void)context;
	return ge::GRAPH_SUCCESS; 
}

IMPL_OP_OPTILING(BatchMatMulReduceScatterAlltoAll)
    .Tiling(BatchMatMulReduceScatterAlltoAllTilingFunc)
    .TilingParse<BatchMatMulReduceScatterAlltoAllCompileInfo>(TilingParseForBatchMatMulReduceScatterAlltoAll);
}  // namespace optiling