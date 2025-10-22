/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include <new>
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "grouped_matmul_swiglu_quant.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz.h"
#include "aclnn_grouped_matmul_swiglu_quant.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t SPLIT = 2L;
static constexpr int64_t K_LIMIT_A8W8 = 65536L;
static constexpr int64_t K_LIMIT_A8W4 = 20000L;
static constexpr int64_t N_LIMIT = 10240L;
static constexpr int64_t NZ_DIM_4_INT8 = 32L;
static constexpr int64_t NZ_DIM_4_INT4 = 64L;
static constexpr int64_t NZ_DIM_3 = 16L;
static constexpr int64_t OUTPUT_IDX_0 = 0L;
static constexpr int64_t OUTPUT_IDX_1 = 1L;
static constexpr int64_t DIM_IDX_0 = 0L;
static constexpr int64_t DIM_IDX_1 = 1L;
static constexpr int64_t DIM_IDX_2 = 2L;
static constexpr int64_t DIM_IDX_3 = 4L;
static constexpr size_t X_DIM_LIMIT = 2UL;
static constexpr size_t WEIGHT_ND_DIM_LIMIT = 3UL;
static constexpr size_t WEIGHT_NZ_DIM_LIMIT = 5UL;
static constexpr size_t WEIGHT_SCALE_DIM_LIMIT = 2UL;
static constexpr size_t WEIGHT_SCALE_PERGROUP_DIM_LIMIT = 3UL;
static constexpr size_t WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT = 2UL;
static constexpr size_t TOKEN_SCALE_DIM_LIMIT = 1UL;
static constexpr size_t BIAS_DIM_LIMIT = 2UL;
static constexpr size_t GROUP_LIST_DIM_LIMIT = 1UL;
static constexpr size_t QUANTOUT_DIM_LIMIT = 2UL;
static constexpr size_t QUANTSCALEOUT_DIM_LIMIT = 1UL;
static constexpr size_t INT4_PER_INT32 = 8UL;
bool isEnableWeightAssistanceMatrix = false;
int dequantMode = 0;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8, DataType::DT_INT4};
static const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> WEIGHT_SCALE_A8W4_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
static const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                           DataType::DT_BF16};
static const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
static const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> BIAS_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static bool CheckNotNull(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, const aclTensor *offset,
                         const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *groupList,
                         const aclTensor *output, const aclTensor *outputScale, const aclTensor *outputOffset)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(weightScale, return false);
    OP_CHECK_NULL(xScale, return false);
    OP_CHECK_NULL(groupList, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(outputScale, return false);
    if (x->GetDataType() == DataType::DT_INT8 && weight->GetDataType() == DataType::DT_INT8 && bias != nullptr) {
        OP_LOGW("aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario that bias is not 0. "
                "Features and accuracy are not guaranteed if inputting bias with values other than 0.");
    } else if (x->GetDataType() == DataType::DT_INT8 && weight->GetDataType() == DataType::DT_INT4 && bias == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario that without bias. "
                "When x is Int8 and weight is int4, bias serves as an auxiliary matrix to weight, and this parameter "
                "cannot be nullptr.");
        return false;
    }
    if (offset != nullptr) {
        OP_LOGW(
            "aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario where offset is not 0. "
            "Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
    }
    if (outputOffset != nullptr) {
        OP_LOGW("aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario where outputOffset "
                "is not 0. Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
    }
    return true;
}

static bool CheckInputOutDims_A8W8(const aclTensor *x, const aclTensor *weight, const aclTensor *weightScale,
                                   const aclTensor *xScale, const aclTensor *groupList, const aclTensor *output,
                                   const aclTensor *outputScale)
{
    OP_CHECK_WRONG_DIMENSION(x, X_DIM_LIMIT, return false);
    op::Format weightViewFormat = weight->GetViewFormat();
    if (IsPrivateFormat(weightViewFormat)) {
        OP_CHECK_WRONG_DIMENSION(weight, WEIGHT_NZ_DIM_LIMIT, return false);
    } else {
        OP_CHECK_WRONG_DIMENSION(weight, WEIGHT_ND_DIM_LIMIT, return false);
    }
    OP_CHECK_WRONG_DIMENSION(weightScale, WEIGHT_SCALE_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(xScale, TOKEN_SCALE_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(output, QUANTOUT_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
    return true;
}

static bool CheckInputOutDims_A8W4(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
                                   const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *groupList,
                                   const aclTensor *output, const aclTensor *outputScale)
{
    OP_CHECK_WRONG_DIMENSION(x, X_DIM_LIMIT, return false);
    op::Format weightViewFormat = weight->GetViewFormat();
    if (IsPrivateFormat(weightViewFormat)) {
        OP_CHECK_WRONG_DIMENSION(weight, WEIGHT_NZ_DIM_LIMIT, return false);
    } else {
        OP_CHECK_WRONG_DIMENSION(weight, WEIGHT_ND_DIM_LIMIT, return false);
    }
    // 支持pergroup、perchannel量化weightScale分别为2维和3维
    OP_CHECK_MAX_DIM(weightScale, WEIGHT_SCALE_PERGROUP_DIM_LIMIT, return false);
    OP_CHECK_MIN_DIM(weightScale, WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(bias, BIAS_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(xScale, TOKEN_SCALE_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(output, QUANTOUT_DIM_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
    return true;
}

static bool CheckInputOutShape_A8W8(const aclTensor *x, const aclTensor *weight, const aclTensor *weightScale,
                                    const aclTensor *xScale, const aclTensor *groupList, const aclTensor *output,
                                    const aclTensor *outputScale)
{
    int64_t m = x->GetViewShape().GetDim(0);
    int64_t k = x->GetViewShape().GetDim(1);
    int64_t n = weightScale->GetViewShape().GetDim(1);
    int64_t e = weight->GetViewShape().GetDim(0);
    if (n % SPLIT != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulSwiGluQuant, N is %ld , not an even number.", n);
        return false;
    }
    int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
    // x的shape期望为[M, K]
    op::Shape xExpectShape = {m, k};
    // weight的NDshape期望为[E, K, N]
    op::Shape weightNDExpectShape = {e, k, n};
    // weight的NZshape期望为[E, N // 32, K // 16, 16, 32]
    op::Shape weightNZExpectShape = {e, static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                     NZ_DIM_3, NZ_DIM_4_INT8};
    // weightScale的shape期望为[E, N]
    op::Shape weightScaleExpectShape = {e, n};
    // xScale的shape期望为[E, N]
    op::Shape xScaleExpectShape = {m};
    // output的shape期望为[M, N]
    op::Shape outputExpectShape = {m, nAfterHalve};
    // outputScale的shape期望为[M]
    op::Shape outputScaleExpectShape = {m};
    op::Format weightViewFormat = weight->GetViewFormat();
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
    if (IsPrivateFormat(weightViewFormat)) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightNZExpectShape, return false);
    } else {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightNDExpectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScale, xScaleExpectShape, return false);

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outputExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputScale, outputScaleExpectShape, return false);
    // groupList的长度应小于等于weight的专家数
    int64_t groupListLen = groupList->GetViewShape().GetDim(0);
    if (groupListLen > e) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant A8W8, Length of 'groupList' out of range (expected to be in range of [1, "
                "%ld], but got %ld)",
                e, groupListLen);
        return false;
    }
    if (n > N_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant A8W8: The current version does not support the scenario that "
                "N(%ld) is greater than %ld.",
                n, N_LIMIT);
        return false;
    }
    if (k >= K_LIMIT_A8W8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant A8W8, The current version does not support the scenario."
                "The tail axis dimension of input0(x) is %ld, which need lower than %ld.",
                k, K_LIMIT_A8W8);
        return false;
    }
    return true;
}

static bool CheckInputOutShape_A8W4(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
                                    const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *groupList,
                                    const aclTensor *output, const aclTensor *outputScale)
{
    int64_t e = weight->GetViewShape().GetDim(0);
    int64_t m = x->GetViewShape().GetDim(0);
    int64_t k = x->GetViewShape().GetDim(1);
    int64_t n = 1;
    int64_t KGroupCount = 1; // K轴的组数，perchannel场景相当于pergroup场景中的组数为1
    int64_t KGroupSize = k;  // K轴每组的元素个数
    op::Shape weightScaleExpectShape;
    // 通过weightScale的维度判断是否为perchannel 或 pergroup量化模式
    if (weightScale->GetViewShape().GetDimNum() == WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT) {
        // weightScale入参在perchannel场景期望shape [E, N]
        n = weightScale->GetViewShape().GetDim(DIM_IDX_1);
        weightScaleExpectShape = {e, n};
    } else if (weightScale->GetViewShape().GetDimNum() == WEIGHT_SCALE_PERGROUP_DIM_LIMIT) {
        // weightScale入参在pergroup场景期望shape [E, KGroupCount, N]
        n = weightScale->GetViewShape().GetDim(DIM_IDX_2);
        KGroupCount = weightScale->GetViewShape().GetDim(DIM_IDX_1);
        KGroupSize = KGroupCount > 0 ? k / KGroupCount : k;
        weightScaleExpectShape = {e, KGroupCount, n};
    }
    if (KGroupCount == 0 || k % KGroupCount != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant, "
                "The number of groups along the k-axis is %ld, and the length of the k-axis is %ld, which is illegal. "
                "The number of groups must be greater than 0, and k-axis length %% number of groups == 0 must be true.",
                KGroupCount, k);
        return false;
    }
    if (n % SPLIT != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulSwiGluQuant, N is %ld , which must even number.", n);
        return false;
    }
    int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
    // x的shape期望为[M, K]
    op::Shape xExpectShape = {m, k};
    // weight的NDshape期望为[E, K, N]
    op::Shape weightNDExpectShape = {e, k, n};
    op::Shape biasExpectShape = {e, n};
    // weight的NZshape期望为[E, N // 64, K // 16, 16, 64]
    op::Shape weightNZExpectShape = {e, static_cast<int64_t>(n / NZ_DIM_4_INT4), static_cast<int64_t>(k / NZ_DIM_3),
                                     NZ_DIM_3, NZ_DIM_4_INT4};
    // xScale的shape期望为[E, N]
    op::Shape xScaleExpectShape = {m};
    // output的shape期望为[M, N]
    op::Shape outputExpectShape = {m, nAfterHalve};
    // outputScale的shape期望为[M]
    op::Shape outputScaleExpectShape = {m};
    op::Format weightViewFormat = weight->GetViewFormat();
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
    if (IsPrivateFormat(weightViewFormat)) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightNZExpectShape, return false);
    } else {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightNDExpectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScale, xScaleExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outputExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputScale, outputScaleExpectShape, return false);
    // groupList的长度应小于等于weight的专家数
    int64_t groupListLen = groupList->GetViewShape().GetDim(0);
    if (groupListLen > e) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant A8W4, Length of 'groupList' out of range (expected to be in range of [1, "
                "%ld], but got %ld)",
                e, groupListLen);
        return false;
    }
    if (n > N_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant A8W4, The current version does not support the scenario."
                "where N after halve is %ld greater than %ld.",
                n, N_LIMIT);
        return false;
    }
    if (k >= K_LIMIT_A8W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant A8W4, The current version does not support the scenario."
                "The tail axis dimension of input0(x) is %ld, which need lower than %ld.",
                k, K_LIMIT_A8W4);
        return false;
    }
    (void)KGroupSize;
    return true;
}

static bool CheckDtypeValid(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
                            const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *groupList,
                            const aclTensor *output, const aclTensor *outputScale)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST, return false);
    if (weight->GetDataType() == DataType::DT_INT4) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
    }
    if (weight->GetDataType() == DataType::DT_INT4) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_A8W4_DTYPE_SUPPORT_LIST, return false);
    } else if (weight->GetDataType() == DataType::DT_INT8) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, QUANTOUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckFormat(const aclTensor *x, const aclTensor *weight, const aclTensor *output)
{
    bool isNZ = weight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
    if ((x->GetDataType() == DataType::DT_INT8 && weight->GetDataType() == DataType::DT_INT8) && !isNZ) {
        // fp16 in fp32 out that is split k template, not precision-advanced now
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario."
                "weight Format expect is FRACTAL_NZ, but got [%s].",
                op::ToString(weight->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario."
                "x Format Not support Private Format.");
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulSwiGluQuant, The current version does not support the scenario."
                "output Format Not support Private Format.");
        return false;
    }
    return true;
}

static void UnpackInt32ToInt4(const aclTensor *&tensorS32, const std::string &tensorType)
{
    OP_LOGD("Unpack %s from int32 to int4 start.", tensorType.c_str());
    auto tensorS4 = const_cast<aclTensor *>(tensorS32);
    op::Shape tensorShape = tensorS4->GetViewShape();
    auto viewShapeDim = tensorShape.GetDimNum();
    tensorShape[viewShapeDim - 1] = tensorShape[viewShapeDim - 1] * INT4_PER_INT32;
    tensorS4->SetViewShape(tensorShape);
    tensorS4->SetStorageShape(tensorShape);
    tensorS4->SetDataType(DataType::DT_INT4);
    OP_LOGD("Unpack %s from int32 to int4 finished.", tensorType.c_str());
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
                               const aclTensor *offset, const aclTensor *weightScale, const aclTensor *xScale,
                               const aclTensor *groupList, const aclTensor *output, const aclTensor *outputScale,
                               const aclTensor *outputOffset)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, weight, bias, offset, weightScale, xScale, groupList, output, outputScale, outputOffset),
              ACLNN_ERR_PARAM_NULLPTR);
    // A8W8场景
    if (x->GetDataType() == DataType::DT_INT8 && weight->GetDataType() == DataType::DT_INT8) {
        // 2. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims_A8W8(x, weight, weightScale, xScale, groupList, output, outputScale),
                  ACLNN_ERR_PARAM_INVALID);

        // 3. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape_A8W8(x, weight, weightScale, xScale, groupList, output, outputScale),
                  ACLNN_ERR_PARAM_INVALID);
    }
    // A8W4场景 INT32为兼容torch_npu考虑，实际计算时，1个INT32数据会被视为8个INT4数据
    if ((x->GetDataType() == DataType::DT_INT8 && weight->GetDataType() == DataType::DT_INT4) ||
        (x->GetDataType() == DataType::DT_INT8 && weight->GetDataType() == DataType::DT_INT32)) {
        // 将INT32视为8个Int4数据，调整viewShape和dtype便于后续统一校验
        if (weight->GetDataType() == DataType::DT_INT32) {
            UnpackInt32ToInt4(weight, "weight");
        }
        if (weightScale->GetDataType() == DataType::DT_INT64) {
            auto weightScale_fix = const_cast<aclTensor *>(weightScale);
            weightScale_fix->SetDataType(DataType::DT_UINT64);
        }
        // 2. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims_A8W4(x, weight, bias, weightScale, xScale, groupList, output, outputScale),
                  ACLNN_ERR_PARAM_INVALID);

        // 3. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape_A8W4(x, weight, bias, weightScale, xScale, groupList, output, outputScale),
                  ACLNN_ERR_PARAM_INVALID);
    }
    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, weight, bias, weightScale, xScale, groupList, output, outputScale),
              ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据形状是否支持
    CHECK_RET(CheckFormat(x, weight, output), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(
    const aclTensor *x, const aclTensor *weight, const aclTensor *bias, const aclTensor *offset,
    const aclTensor *weightScale, const aclTensor *xScale, const aclTensor *groupList, aclTensor *output,
    aclTensor *outputScale, aclTensor *outputOffset, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查

    auto ret = CheckParams(x, weight, bias, offset, weightScale, xScale, groupList, output, outputScale, outputOffset);

    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空Tensor场景
    if (output->IsEmpty() || groupList->IsEmpty() || outputScale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 转连续
    x = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 若weight为私有格式，则不应该做连续性转换 （l0op::Contiguous接口会把viewShape赋值给storageShape）
    if (IsPrivateFormat(weight->GetStorageFormat())) {
        weight->SetOriginalShape(weight->GetViewShape());
    } else {
        weight = l0op::Contiguous(weight, uniqueExecutor.get());
    }
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    weightScale = l0op::Contiguous(weightScale, uniqueExecutor.get());
    CHECK_RET(weightScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    xScale = l0op::Contiguous(xScale, uniqueExecutor.get());
    CHECK_RET(xScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
    CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用L0算子能力
    if (bias != nullptr) {
        isEnableWeightAssistanceMatrix = true;
    }
    if (isEnableWeightAssistanceMatrix && weightScale->GetViewShape().GetDimNum() == WEIGHT_SCALE_PERGROUP_DIM_LIMIT) {
        dequantMode = 1;
    }
    auto ret_0 = l0op::GroupedMatmulSwigluQuant(x, weight, weightScale, xScale, groupList, bias,
                                                isEnableWeightAssistanceMatrix, dequantMode, uniqueExecutor.get());
    CHECK_RET(ret_0 != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    auto out0 = std::get<OUTPUT_IDX_0>(ret_0);
    auto ret_1 = l0op::ViewCopy(out0, output, uniqueExecutor.get());
    CHECK_RET(ret_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto out1 = std::get<OUTPUT_IDX_1>(ret_0);
    auto ret_2 = l0op::ViewCopy(out1, outputScale, uniqueExecutor.get());
    CHECK_RET(ret_2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                          const aclTensor *bias, const aclTensor *offset,
                                                          const aclTensor *weightScale, const aclTensor *xScale,
                                                          const aclTensor *groupList, aclTensor *output,
                                                          aclTensor *outputScale, aclTensor *outputOffset,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuant, DFX_IN(x, weight, bias, offset, weightScale, xScale, groupList),
                   DFX_OUT(output, outputScale, outputOffset));
    // 固定写法，创建OpExecutor
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(x, weight, bias, offset, weightScale, xScale, groupList,
                                                               output, outputScale, outputOffset, workspaceSize,
                                                               executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                                  const aclTensor *bias, const aclTensor *offset,
                                                                  const aclTensor *weightScale, const aclTensor *xScale,
                                                                  const aclTensor *groupList, aclTensor *output,
                                                                  aclTensor *outputScale, aclTensor *outputOffset,
                                                                  uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantWeightNZ,
                   DFX_IN(x, weight, bias, offset, weightScale, xScale, groupList),
                   DFX_OUT(output, outputScale, outputOffset));
    // weight在该场景下强制绑定StorageFormat 和 ViewFormat 为NZ
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto storgeShape = weight->GetStorageShape();
    auto viewShape = weight->GetViewShape();
    aclTensor *weightNZ = const_cast<aclTensor *>(weight);
    CHECK_COND((storgeShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT), ACLNN_ERR_PARAM_INVALID,
               "aclnnGroupedMatmulSwigluQuantWeightNZ, The dimnum of storageShape for second input (weight)"
               "must be 5. \n But StorageShape got %s , and dimNum is %lu.",
               op::ToString(storgeShape).GetString(), storgeShape.GetDimNum());
    // weight的StorageFormat无条件视为NZ
    weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    if (viewShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT) {
        // 若weight的viewShape为5维则视为NZ
        weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
    } else if (viewShape.GetDimNum() == WEIGHT_ND_DIM_LIMIT) {
        // 若weight的viewShape为3维则视为ND
        weightNZ->SetViewFormat(op::Format::FORMAT_ND);
    }
    // 调用公共接口
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(x, weight, bias, offset, weightScale, xScale, groupList,
                                                               output, outputScale, outputOffset, workspaceSize,
                                                               executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuant);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuant launch aicore");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantWeightNZ);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantWeightNZ launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif