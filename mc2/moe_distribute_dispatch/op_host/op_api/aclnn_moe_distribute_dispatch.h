/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_DISPATCH_
#define OP_API_INC_MOE_DISTRIBUTE_DISPATCH_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoeDistributeDispatch功能，对Token数据先进行量化，再进行EP域的alltoallv通信，再进行TP域的allgatherv通信。
 * @brief aclnnMoeDistributeDispatch的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。输入的token数据。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。每个token的topK个专家索引。
 * @param [in] scales: 计算可选输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。每个专家的smooth权重。
 * @param [in] xActiveMask: 计算输入，Tensor，数据类型Bool，必须为1维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] expertScales: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。同一个EP通信域中各卡的epRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。
 * @param [in] groupTp: 计算可选输入，str。tp通信域名称，数据并行的通信域。无tp通信域时传空。
 * @param [in] tpWorldSize: 计算可选输入，int。tp通信域size。
 * @param [in] tpRankId: 计算可选输入，int。tp本卡Id。
 * @param [in] expertShardType: 计算可选输入，int。专家共享类型。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家卡数量。
 * @param [in] quantMode: 计算可选输入，int，量化模式。
 * @param [in] globalBs: 计算可选输入，int。EP域全局的batch size大小。
 * @param [in] expertTokenNumsType: 计算可选输入，int。输出expertTokenNums中的值语义类型。
 * @param [out] expandX: 计算输出，Tensor，必选输出，数据类型支持float16, bfloat16, int8，仅支持2维，数据格式支持ND。根据
    expertIdx进行扩展过的token特征。
 * @param [out] dynamicScales: 计算输出，Tensor，必选输出，数据类型float32，仅支持1维，数据格式支持ND。quantMode为0时输出为空。
 * @param [out] expandIdx: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。给同一专家发送的token个数。
 * @param [out] expertTokenNums: 计算输出，Tensor，必选输出，数据类型int64，仅支持1维，数据格式支持ND。每个专家收到的token个数。
 * @param [out] epRecvCounts: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。表示从各卡接收的token数。
 * @param [out] tpRecvCounts: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。无tp通信域时输出为空。
 * @param [out] expandScales: 计算输出，Tensor，必选输出，数据类型为float32，仅支持1维，数据格式支持ND。输出的weights，每个
    输出token附带有topK个weight信息。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds,
    const aclTensor* scales, const aclTensor* xActiveMask,
    const aclTensor* expertScales, const aclTensor* waitCost,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, 
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType,
    aclTensor* expandX, aclTensor* dynamicScales,
    aclTensor* expandIdx, aclTensor* expertTokenNums,
    aclTensor* epRecvCounts, aclTensor* tpRecvCounts,
    aclTensor* expandScales, uint64_t* workspaceSize, 
    aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeDispatch的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMoeDistributeDispatchGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatch(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_DISPATCH_