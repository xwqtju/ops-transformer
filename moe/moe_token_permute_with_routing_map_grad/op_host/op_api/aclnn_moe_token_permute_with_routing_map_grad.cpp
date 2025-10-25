/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_permute_with_routing_map_grad.h"
#include "level0/inplace_index_add.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_moe_token_permute_with_routing_map_grad.h"
#include "level0/sort.h"
#include "level0/zero_op.h"
#include "level0/masked_scatter.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;
static constexpr int64_t INDEX_SHAPE_1 = 1;
static constexpr int64_t TRANSPOSE_SHAPE_SIZE = 2;
static constexpr int64_t INPUT_MAX_GROUP = 2048;
// FLOAT类型在2139095040时为inf，不能sort
static constexpr int64_t MAX_SORT_SHAPE_DIM = 2139095040;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,  op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT16,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_95_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_BF16,    op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64,   op::DataType::DT_BOOL};

static bool IsAICoreSupport(const aclTensor* self)
{
    // 根据芯片类型和输入self类型判断是否走aicore
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), ASCEND910_95_AICORE_DTYPE_SUPPORT_LIST);
    } else if (
        GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        if (CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
            return true;
        }
    } else {
        if (CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST)) {
            return true;
        }
    }
    return false;
}
static const std::initializer_list<DataType> dtype_list = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType> routing_map_dtype_list = {op::DataType::DT_INT8, op::DataType::DT_BOOL};

static const std::initializer_list<DataType> indice_dtype_list = {op::DataType::DT_INT32};

static inline bool CheckNotNull(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, aclTensor* tokensGradOut,
    aclTensor* probsGradOutOptional)
{
    OP_CHECK_NULL(permutedTokenOutputGrad, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    OP_CHECK_NULL(tokensGradOut, return false);

    if (routingMapOptional == nullptr && permutedProbsOutputGradOptional != nullptr) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when routingMapOptional is nullptr, permutedProbsOutputGradOptional should be nullptr.");
        return false;
    }
    (void)probsGradOutOptional;
    (void)tokensGradOut;
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, aclTensor* tokensGradOut,
    aclTensor* probsGradOutOptional)
{
    // 检查permutedTokenOutputGrad的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(permutedTokenOutputGrad, dtype_list, return false);
    if (permutedProbsOutputGradOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutedProbsOutputGradOptional, dtype_list, return false);
        OP_CHECK_DTYPE_NOT_MATCH(permutedProbsOutputGradOptional, permutedTokenOutputGrad->GetDataType(), return false);
    }

    // 检查out的数据类型是否在支持列表内
    if (routingMapOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(routingMapOptional, routing_map_dtype_list, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, indice_dtype_list, return false);
    (void)probsGradOutOptional;
    (void)tokensGradOut;
    return true;
}

static bool CheckShapeValid(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* routingMap, const aclTensor* probsOptional)
{
    if (permutedTokenOutputGrad != nullptr) {
        auto permutedTokenOutputGradDimNum = permutedTokenOutputGrad->GetViewShape().GetDimNum();
        OP_CHECK(
            permutedTokenOutputGradDimNum == TRANSPOSE_SHAPE_SIZE,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimensions of permutedTokenOutputGrad should be two, but got %lu.",
                permutedTokenOutputGradDimNum),
            return false);
    }
    if (routingMap != nullptr) {
        auto routingMapDimNum = routingMap->GetViewShape().GetDimNum();
        OP_CHECK(
            routingMapDimNum == TRANSPOSE_SHAPE_SIZE,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimensions of routingMap should be two, but got %lu.", routingMapDimNum),
            return false);
    }
    if (probsOptional != nullptr) {
        auto probsDimNum = probsOptional->GetViewShape().GetDimNum();
        OP_CHECK(
            probsDimNum == INDEX_SHAPE_1,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimensions of probsOptional should be one, but got %lu.", probsDimNum),
            return false);
    }
    return true;
}

static bool checkAttrValid(int64_t numExperts, int64_t tokensNum)
{
    if (numExperts <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numExperts should be large than 0, but current is %ld", numExperts);
        return false;
    }
    if (tokensNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tokensNum should be large than 0, but current is %ld", tokensNum);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, int64_t numExperts, int64_t tokensNum,
    bool dropAndPad, aclTensor* tokensGradOut, aclTensor* probsGradOutOptional)
{
    // 1. 检查Attr是否合法
    CHECK_RET(checkAttrValid(numExperts, tokensNum), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查参数是否为空指针
    CHECK_RET(
        CheckNotNull(
            permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional, tokensGradOut,
            probsGradOutOptional),
        ACLNN_ERR_PARAM_NULLPTR);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(
        CheckDtypeValid(
            permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional, tokensGradOut,
            probsGradOutOptional),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckShapeValid(permutedTokenOutputGrad, routingMapOptional, permutedProbsOutputGradOptional),
        ACLNN_ERR_PARAM_INVALID);

    (void)dropAndPad;
    return ACLNN_SUCCESS;
}
static void ViewDataType(const aclTensor* input, const op::DataType dtype)
{
    if (input == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "view data type error! it is null.");
        return;
    }
    auto tmpTensor = const_cast<aclTensor*>(input);
    tmpTensor->SetDataType(dtype);
    input = tmpTensor;
}
} // namespace

aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, int64_t numExperts, int64_t tokensNum,
    bool dropAndPad, aclTensor* tokensGradOut, aclTensor* probsGradOutOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnMoeTokenPermuteWithRoutingMapGrad,
        DFX_IN(
            permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional, numExperts,
            tokensNum, dropAndPad),
        DFX_OUT(tokensGradOut, probsGradOutOptional));

    auto ret = CheckParams(
        permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional, numExperts,
        tokensNum, dropAndPad, tokensGradOut, probsGradOutOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 如果是空tensor，直接返回
    if (permutedTokenOutputGrad->IsEmpty() || tokensGradOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto permutedTokenOutputGradContiguous = l0op::Contiguous(permutedTokenOutputGrad, uniqueExecutor.get());
    CHECK_RET(permutedTokenOutputGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* permutedProbsOutputGradOptionalContiguous = nullptr;
    const aclTensor* zeroPermutedProbsOutputGrad = nullptr;
    FVector<int64_t> transposeDim = {1, 0};
    auto perm = uniqueExecutor->AllocIntArray(transposeDim.data(), transposeDim.size());
    if (permutedProbsOutputGradOptional != nullptr) {
        permutedProbsOutputGradOptionalContiguous =
            l0op::Contiguous(permutedProbsOutputGradOptional, uniqueExecutor.get());
    }
    // 处理第一个输出
    auto TokensGradOutContiguous = l0op::Contiguous(tokensGradOut, uniqueExecutor.get());
    CHECK_RET(TokensGradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto zeroTokensGradOut = l0op::ZerosLike(TokensGradOutContiguous, uniqueExecutor.get());
    CHECK_RET(zeroTokensGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 处理第二个输出
    if (probsGradOutOptional != nullptr && permutedProbsOutputGradOptional != nullptr) {
        auto probsGradOutOptionalContiguous = l0op::Contiguous(probsGradOutOptional, uniqueExecutor.get());
        CHECK_RET(probsGradOutOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 先清零
        zeroPermutedProbsOutputGrad = l0op::ZerosLike(probsGradOutOptionalContiguous, uniqueExecutor.get());
        CHECK_RET(zeroPermutedProbsOutputGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto sortedIndicesContiguous = l0op::Contiguous(sortedIndices, uniqueExecutor.get());
    CHECK_RET(sortedIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* routingMapOptionalContiguous = nullptr;
    if (routingMapOptional != nullptr) {
        routingMapOptionalContiguous = l0op::Contiguous(routingMapOptional, uniqueExecutor.get());
        CHECK_RET(routingMapOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (dropAndPad == false) {
        // 算子kernel内部只会处理第一个输出，但是第二个输出传空指针时会影响第一个输出，所以这里暂时传zeroTokensGradOut
        auto MoeTokenpermuteGradWithRoutingMapOut = l0op::MoeTokenPermuteWithRoutingMapGrad(
            permutedTokenOutputGradContiguous, permutedProbsOutputGradOptionalContiguous, sortedIndicesContiguous,
            routingMapOptionalContiguous, numExperts, tokensNum, dropAndPad, zeroTokensGradOut, zeroTokensGradOut,
            uniqueExecutor.get());
        auto permuteGradTokensOpOut = MoeTokenpermuteGradWithRoutingMapOut[0];
        CHECK_RET(permuteGradTokensOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto permuteGradTokensResult = l0op::ViewCopy(permuteGradTokensOpOut, tokensGradOut, uniqueExecutor.get());
        CHECK_RET(permuteGradTokensResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (probsGradOutOptional != nullptr && permutedProbsOutputGradOptional != nullptr) {
            CHECK_RET(zeroPermutedProbsOutputGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto maskBool = l0op::Cast(routingMapOptionalContiguous, DataType::DT_BOOL, uniqueExecutor.get());
            CHECK_RET(maskBool != nullptr, ACLNN_ERR_INNER_NULLPTR);
            // mask scator
            zeroPermutedProbsOutputGrad = l0op::Transpose(zeroPermutedProbsOutputGrad, perm, uniqueExecutor.get());
            CHECK_RET(zeroPermutedProbsOutputGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            maskBool = l0op::Transpose(maskBool, perm, uniqueExecutor.get());
            CHECK_RET(maskBool != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto maskedScatterOpOut = l0op::MaskedScatter(
                zeroPermutedProbsOutputGrad, maskBool, permutedProbsOutputGradOptionalContiguous, uniqueExecutor.get());
            CHECK_RET(maskedScatterOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            maskedScatterOpOut = l0op::Transpose(maskedScatterOpOut, perm, uniqueExecutor.get());
            CHECK_RET(maskedScatterOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
            auto permuteProbsResult = l0op::ViewCopy(maskedScatterOpOut, probsGradOutOptional, uniqueExecutor.get());
            CHECK_RET(permuteProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        constexpr bool descending = false;
        constexpr bool stable = true;

        if (nullptr != zeroPermutedProbsOutputGrad) {
            auto MoeTokenpermuteGradWithRoutingMapOut = l0op::MoeTokenPermuteWithRoutingMapGrad(
                permutedTokenOutputGradContiguous, permutedProbsOutputGradOptionalContiguous, sortedIndicesContiguous,
                routingMapOptionalContiguous, numExperts, tokensNum, dropAndPad, zeroTokensGradOut,
                zeroPermutedProbsOutputGrad, uniqueExecutor.get());
            auto probsGradOpOut = MoeTokenpermuteGradWithRoutingMapOut[1];
            CHECK_RET(probsGradOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto probsGradOutOptionalResult =
                l0op::ViewCopy(probsGradOpOut, probsGradOutOptional, uniqueExecutor.get());
            CHECK_RET(probsGradOutOptionalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        const aclTensor* indexAddOut = nullptr;
        // 当设备类型为A2或A3且index为int32类型时，切为InplaceIndexAddWithSorted算子
        bool useNewOp = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) &&
                        tokensNum < MAX_SORT_SHAPE_DIM &&
                        (zeroTokensGradOut->GetDataType() == op::DataType::DT_BF16 ||
                         zeroTokensGradOut->GetDataType() == op::DataType::DT_FLOAT16);
        if (useNewOp) {
            const aclTensor* indicesViewFloat =
                uniqueExecutor.get()->CreateView(sortedIndicesContiguous, sortedIndicesContiguous->GetViewShape(), 0);
            ViewDataType(indicesViewFloat, op::DataType::DT_FLOAT);
            auto sortResult = l0op::Sort(indicesViewFloat, -1, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
            auto sortValues = std::get<0>(sortResult);
            auto sortIndex = std::get<1>(sortResult);
            CHECK_RET(sortValues != nullptr && sortIndex != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto sortValuesI32 = uniqueExecutor.get()->CreateView(
                sortValues, sortedIndicesContiguous->GetViewShape(), sortValues->GetViewOffset());
            ViewDataType(sortValuesI32, op::DataType::DT_INT32);
            // inplace index add
            indexAddOut = l0op::InplaceIndexAddWithSorted(
                zeroTokensGradOut, 0, sortValuesI32, sortIndex, permutedTokenOutputGradContiguous, nullptr,
                uniqueExecutor.get());
        } else if (IsAICoreSupport(zeroTokensGradOut)) {
            indexAddOut = l0op::InplaceIndexAddAiCore(
                zeroTokensGradOut, 0, sortedIndicesContiguous, permutedTokenOutputGradContiguous, nullptr,
                uniqueExecutor.get());
        } else {
            indexAddOut = l0op::InplaceIndexAddAiCpu(
                zeroTokensGradOut, 0, sortedIndicesContiguous, permutedTokenOutputGradContiguous, nullptr,
                uniqueExecutor.get());
        }

        CHECK_RET(indexAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto tokensGradOutResult = l0op::ViewCopy(indexAddOut, tokensGradOut, uniqueExecutor.get());
        CHECK_RET(tokensGradOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnMoeTokenPermuteWithRoutingMapGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif