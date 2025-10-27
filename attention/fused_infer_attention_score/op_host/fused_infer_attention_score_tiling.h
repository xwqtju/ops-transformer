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
 * \file fused_infer_attention_score_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_impl.h"
#include "register/tilingdata_base.h"
#include "fused_infer_attention_score_tiling_compile_info.h"
#include "fused_infer_attention_score_tiling_index.h"

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif

namespace optiling {
const uint32_t FIA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
// 基础参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, headDimRope)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS1Dims)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS2Dims)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, needInit)
TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionBaseParamsOp, FusedInferAttentionBaseParams)

// PageAttention 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionPageAttentionParams)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPageAttentionParamsOp, FusedInferAttentionPageAttentionParams)

// AttenMask 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionMaskParams)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(int32_t, preToken)
TILING_DATA_FIELD_DEF(int32_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionMaskParamsOp, FusedInferAttentionMaskParams)

// 内切基本块参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionInnerSplitParamsOp, FusedInferAttentionInnerSplitParams)

// workspace参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionWorkspaceParams)
TILING_DATA_FIELD_DEF(uint32_t, mm1ResSize)
TILING_DATA_FIELD_DEF(uint32_t, mm2ResSize)
TILING_DATA_FIELD_DEF(uint32_t, fdAccumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, fdLogSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionWorkspaceParamsOp, FusedInferAttentionWorkspaceParams)

// 外切分核参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionOuterSplitParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2End)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionOuterSplitParamsOp, FusedInferAttentionOuterSplitParams)

// FlashDecode规约参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionFlashDecodeParams)
TILING_DATA_FIELD_DEF(uint32_t, numOfFdHead)
TILING_DATA_FIELD_DEF(uint32_t, reserved)
TILING_DATA_FIELD_DEF(uint32_t, gS1BaseSizeOfFd)                                    // FD负载均衡中，每个FD任务按gS1切分的基本size
TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd)                                     // FD负载均衡中，用到的vector数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitNumOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitStartIdxOfCore)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1SplitNumOfFdHead)          // FD负载均衡中，每个FD任务按gS1基本size切分后的份数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1LastPartSizeOfFdHead)      // FD负载均衡中，每个FD任务按gS1基本size切分后，最后一份的gS1大小，即尾块大小
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHead)        // FD负载均衡中，每个vector核处理的最后一个FD任务的序号
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHeadSplit)   // FD负载均衡中，每个vector核处理的最后一个FD任务的子划分的序号
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionFlashDecodeParamsOp, FusedInferAttentionFlashDecodeParams)

//MLA非量化模板TilingData
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionFlashDecodeParams, fdParams);
END_TILING_DATA_DEF

// Left Padding 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionLeftPaddingParams)
TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionLeftPaddingParamsOp, FusedInferAttentionLeftPaddingParams)

// Pse 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionPseParams)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftB)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPseParamsOp, FusedInferAttentionPseParams)

// 后量化 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionPostQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPostQuantParamsOp, FusedInferAttentionPostQuantParams)

// 公共前缀 
BEGIN_TILING_DATA_DEF(FusedInferAttentionPrefixParams)
TILING_DATA_FIELD_DEF(uint64_t, prefixAttenOutOffset) //临时输出偏移
TILING_DATA_FIELD_DEF(uint64_t, userPromptAttenOutOffset)
TILING_DATA_FIELD_DEF(uint64_t, tmpLseOffset)
TILING_DATA_FIELD_DEF(uint64_t, prefixLen)
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range)
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange)
TILING_DATA_FIELD_DEF(uint32_t, batchSizeQ)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPrefixParamsOp, FusedInferAttentionPrefixParams)

// 全量化 参数 当前无
BEGIN_TILING_DATA_DEF(FusedInferAttentionFullQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, placeHolder)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionFullQuantParamsOp, FusedInferAttentionFullQuantParams)

// L2 Cache 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionL2CacheParams)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionL2CacheParamsOp, FusedInferAttentionL2CacheParams)

// MSD 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionMsdParams)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionMsdParamsOp, FusedInferAttentionMsdParams)

// 伪量化 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionAntiqParams)
TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionAntiqParamsOp, FusedInferAttentionAntiqParams)

extern "C" {
ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context);
ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(gert::TilingContext *context);
}
ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context);
FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
