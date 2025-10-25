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
 * \file rope_with_sin_cos_cache.cpp
 * \brief
 */
#include <iostream>
#include <cstdio>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "rope_with_sin_cos_cache_tiling.h"

namespace {
constexpr uint32_t INPUT_POSITION_INDEX = 0;
constexpr uint32_t INPUT_QUERY_IN_INDEX = 1;
constexpr uint32_t INPUT_KEY_IN_INDEX = 2;
constexpr uint32_t INPUT_COSSINCACHE_INDEX = 3;

constexpr uint32_t INPUT_MROPE_SECTION_INDEX = 0;
constexpr uint32_t INPUT_QSTRIDE_INDEX = 1;
constexpr uint32_t INPUT_KSTRIDE_INDEX = 2;
constexpr uint32_t INPUT_IS_NEOXSTYLE_INDEX = 3;
constexpr uint32_t INPUT_NUM_QHEADS_INDEX = 4;
constexpr uint32_t INPUT_NUM_kHEADS_INDEX = 5;

constexpr uint32_t INDEX_QUERYOUT_OUTPUT_ = 0;
constexpr uint32_t INDEX_KEYOUT_OUTPUT = 1;

static constexpr uint32_t TILING_BF16 = 20;
static constexpr uint32_t TILING_FP16 = 21;
static constexpr uint32_t TILING_FP32 = 22;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;

constexpr int64_t UB_SIZE = static_cast<int64_t>(192) * 1024;
constexpr uint32_t FP32_DTYPE_SIZE = 4;

struct TilingParams {
    uint64_t core_num_use = 0;
    uint64_t num_tokens = 0;
    uint64_t num_q_heads = 0;
    uint64_t num_kv_heads = 0;
    uint64_t head_size = 0;
    uint64_t rotary_dim = 0;
    uint64_t mrope_section0 = 0;
    uint64_t mrope_section1 = 0;
    uint64_t mrope_section2 = 0;
    uint64_t q_leading_dimension = 0;
    uint64_t k_leading_dimension = 0;
    uint64_t isNeoxStyle = 0;
    uint64_t front_core = 0;
    uint64_t tail_core = 0;
    uint64_t num_tokens_front_core_each_loop = 0;
    uint64_t num_tokens_tail_core_each_loop = 0;
    uint64_t num_tokens_each_front_core = 0;
    uint64_t num_tokens_each_tail_core = 0;
    uint64_t loop_time_each_front_core = 0;
    uint64_t loop_time_each_tail_core = 0;
    uint64_t num_tokens_front_core_last_loop = 0;
    uint64_t num_tokens_tail_core_last_loop = 0;
    uint64_t tilingKey = 0;
};
} // namespace

namespace optiling {

static void SetTiling(TilingParams& params, RopeWithSinCosCacheTilingData& tiling)
{
    tiling.set_core_num_use(params.core_num_use);
    tiling.set_num_tokens(params.num_tokens);
    tiling.set_num_q_heads(params.num_q_heads);
    tiling.set_num_kv_heads(params.num_kv_heads);
    tiling.set_head_size(params.head_size);
    tiling.set_rotary_dim(params.rotary_dim);
    tiling.set_mrope_section0(params.mrope_section0);
    tiling.set_mrope_section1(params.mrope_section1);
    tiling.set_mrope_section2(params.mrope_section2);
    tiling.set_q_leading_dimension(params.q_leading_dimension);
    tiling.set_k_leading_dimension(params.k_leading_dimension);
    tiling.set_isNeoxStyle(params.isNeoxStyle);
    tiling.set_front_core(params.front_core);
    tiling.set_tail_core(params.tail_core);
    tiling.set_num_tokens_front_core_each_loop(params.num_tokens_front_core_each_loop);
    tiling.set_num_tokens_tail_core_each_loop(params.num_tokens_tail_core_each_loop);
    tiling.set_num_tokens_each_front_core(params.num_tokens_each_front_core);
    tiling.set_num_tokens_each_tail_core(params.num_tokens_each_tail_core);
    tiling.set_loop_time_each_front_core(params.loop_time_each_front_core);
    tiling.set_loop_time_each_tail_core(params.loop_time_each_tail_core);
    tiling.set_num_tokens_front_core_last_loop(params.num_tokens_front_core_last_loop);
    tiling.set_num_tokens_tail_core_last_loop(params.num_tokens_tail_core_last_loop);
}

static ge::graphStatus TilingKeyChose(const gert::TilingContext* context, TilingParams& params)
{
    auto qDtype = context->GetInputDesc(INPUT_QUERY_IN_INDEX)->GetDataType();
    if (qDtype == ge::DT_BF16) {
        params.tilingKey = TILING_BF16;
        return ge::GRAPH_SUCCESS;
    }
    if (qDtype == ge::DT_FLOAT) {
        params.tilingKey = TILING_FP32;
        return ge::GRAPH_SUCCESS;
    }
    if (qDtype == ge::DT_FLOAT16) {
        params.tilingKey = TILING_FP16;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingCompute(gert::TilingContext* context, TilingParams& params)
{
    uint64_t numTokens =
        static_cast<uint64_t>(context->GetOutputShape(INDEX_QUERYOUT_OUTPUT_)->GetStorageShape().GetDim(DIM_0));
    uint64_t numQheads = params.num_q_heads;
    uint64_t numKheads = params.num_kv_heads;
    uint64_t headSize = params.head_size;

    auto cosSinSize = context->GetInputShape(INPUT_COSSINCACHE_INDEX)->GetStorageShape().GetDimNum();
    uint64_t rotaryDim = static_cast<uint64_t>(
        context->GetInputShape(INPUT_COSSINCACHE_INDEX)->GetStorageShape().GetDim(cosSinSize - 1));

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t coreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNum());

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    auto maxUbSize = static_cast<uint64_t>(ubSizePlatForm);

    uint64_t totalDataNum = numTokens;
    uint64_t dataTypeSize = FP32_DTYPE_SIZE;
    uint64_t front_core = totalDataNum % coreNum != 0 ? static_cast<uint64_t>(totalDataNum % coreNum) : coreNum;
    uint64_t tail_core = totalDataNum <= coreNum ? 0 : coreNum - front_core;
    uint64_t blockDim = front_core + tail_core;

    uint64_t numHeadsMax = numQheads > numKheads ? numQheads : numKheads;
    uint64_t allSize = params.isNeoxStyle == 1UL ?
                           static_cast<uint64_t>(numHeadsMax * (rotaryDim * 8UL + headSize) * dataTypeSize) :
                           static_cast<uint64_t>(numHeadsMax * (rotaryDim * 10UL + headSize) * dataTypeSize);
    uint64_t maxNPerLoopForUb = maxUbSize / allSize; // ub每次能载入最大行数（包括所有计算数据）;

    uint64_t num_tokens_each_front_core = (totalDataNum + coreNum - 1) / coreNum;
    uint64_t loop_time_each_front_core =
        (num_tokens_each_front_core + maxNPerLoopForUb - 1UL) / static_cast<uint64_t>(maxNPerLoopForUb);

    uint64_t num_tokens_front_core_each_loop =
        loop_time_each_front_core == 1UL ? num_tokens_each_front_core : maxNPerLoopForUb;
    uint64_t num_tokens_front_core_last_loop =
        loop_time_each_front_core == 1UL ?
            0 :
            num_tokens_each_front_core - num_tokens_front_core_each_loop * (loop_time_each_front_core - 1UL);

    uint64_t num_tokens_each_tail_core = totalDataNum / coreNum;
    uint64_t loop_time_each_tail_core = (num_tokens_each_tail_core + maxNPerLoopForUb - 1) / maxNPerLoopForUb;
    uint64_t num_tokens_tail_core_each_loop =
        loop_time_each_tail_core <= 1UL ? num_tokens_each_tail_core : maxNPerLoopForUb;
    uint64_t num_tokens_tail_core_last_loop =
        static_cast<uint64_t>(loop_time_each_front_core) == 1UL ?
            0 :
            num_tokens_each_tail_core - num_tokens_tail_core_each_loop * (loop_time_each_tail_core - 1UL);

    params.num_tokens = numTokens;
    params.rotary_dim = rotaryDim;
    params.core_num_use = blockDim;
    params.front_core = front_core;
    params.tail_core = tail_core;

    params.num_tokens_each_front_core = num_tokens_each_front_core;
    params.loop_time_each_front_core = loop_time_each_front_core;
    params.num_tokens_front_core_each_loop = num_tokens_front_core_each_loop;
    params.num_tokens_front_core_last_loop = num_tokens_front_core_last_loop;
    params.num_tokens_each_tail_core = num_tokens_each_tail_core;
    params.loop_time_each_tail_core = loop_time_each_tail_core;
    params.num_tokens_tail_core_each_loop = num_tokens_tail_core_each_loop;
    params.num_tokens_tail_core_last_loop = num_tokens_tail_core_last_loop;

    context->SetBlockDim(blockDim);
    context->SetTilingKey(params.tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingRopeWithSinCosCache(gert::TilingContext* context)
{
    TilingParams params;
    RopeWithSinCosCacheTilingData tiling;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(totalCoreNum);
    params.core_num_use = context->GetBlockDim();

    auto* attrs = context->GetAttrs();

    const uint64_t* attrNumQHeads = attrs->GetAttrPointer<uint64_t>(0);
    params.num_q_heads = static_cast<uint64_t>(*attrNumQHeads);

    const uint64_t* attrNumKVHeads = attrs->GetAttrPointer<uint64_t>(1);
    params.num_kv_heads = static_cast<uint64_t>(*attrNumKVHeads);

    const uint64_t* attrHeadSize = attrs->GetAttrPointer<uint64_t>(2);
    params.head_size = static_cast<uint64_t>(*attrHeadSize);

    const auto attrMRopeSection = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    const uint64_t* attrMRopeSectionData = reinterpret_cast<const uint64_t*>(attrMRopeSection->GetData());
    if (attrMRopeSectionData != nullptr) {
        params.mrope_section0 = attrMRopeSectionData[0];
        params.mrope_section1 = attrMRopeSectionData[1];
        params.mrope_section2 = attrMRopeSectionData[2];
    }

    const uint64_t* attrQStride = attrs->GetAttrPointer<uint64_t>(4);
    params.q_leading_dimension = static_cast<uint64_t>(*attrQStride);

    const uint64_t* attrKStride = attrs->GetAttrPointer<uint64_t>(5);
    params.k_leading_dimension = static_cast<uint64_t>(*attrKStride);

    const bool* attrIsNeoxStyle = attrs->GetAttrPointer<bool>(6);
    params.isNeoxStyle = static_cast<bool>(*attrIsNeoxStyle);

    TilingKeyChose(context, params);
    TilingCompute(context, params);
    SetTiling(params, tiling);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context->GetWorkspaceSizes(1);

    size_t UserWorkspaceSize = 0;

    workspaces[0] = sysWorkspaceSize + UserWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

struct TilingmropeCompileInfo {
};

static ge::graphStatus TilingPreparemrope(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RopeWithSinCosCache)
    .Tiling(TilingRopeWithSinCosCache)
    .TilingParse<TilingmropeCompileInfo>(TilingPreparemrope);
} // namespace optiling
