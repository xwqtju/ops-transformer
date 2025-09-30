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
 * \file test_aclnn_moe_distribute_dispatch.cpp
 * \brief
 */

#include <thread>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "../op_host/op_api/aclnn_moe_distribute_dispatch.h"
#include "../../moe_distribute_combine/op_host/op_api/aclnn_moe_distribute_combine.h"
#include "aclnn/opdev/fp16_t.h"
#include <random>

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

#define ACLCHECK(ret) do { \
    if(ret != ACL_SUCCESS)\
    {\
        printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);\
    }\
} while(0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, 
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
    );
    return 0;
}

struct Args_A2 {
    int rankId;
    int epRankId;
    char* groupEpName;
    HcclComm hcclEpComm;
    aclrtStream stream;
};

struct Args_A3 {
    uint32_t rankId;
    uint32_t epRankId;
    uint32_t tpRankId;
    HcclComm hcclEpComm;
    HcclComm hcclTpComm;
    aclrtStream dispatchStream;
    aclrtStream combineStream;
    aclrtContext context;
};

// A2
constexpr int EP_WORLD_SIZE_A2 = 16;
constexpr int TP_WORLD_SIZE_A2 = 0;
int FIRST_RANK_ID = 0;
// A3
constexpr uint32_t EP_WORLD_SIZE_A3 = 8;
constexpr uint32_t TP_WORLD_SIZE_A3 = 2;
constexpr uint32_t DEV_NUM = EP_WORLD_SIZE_A3 * TP_WORLD_SIZE_A3;

int launchOneProcessMoeDistributeDispatchA2(Args_A2 &args)
{
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 8;
    int64_t shardType = 0; // dispatch need
    int64_t quantMode = 0; // dispatch need
    bool isQuant = false;  // dispatch need
    int64_t expertTokenNumsType = 0; // dispatch need
    int64_t expertShardType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t sharedExpertNum = 0;
    int64_t moeExpertNum = 16;
    int64_t globalBS = BS * EP_WORLD_SIZE_A2;      // tiling里处理成BS*world_size
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    const char* groupTpName = "";
    int64_t tpWorldSize = 0;
    int64_t tpRankId = 0;

    int64_t localMoeExpertNum = moeExpertNum / (EP_WORLD_SIZE_A2 - sharedExpertRankNum);
    int64_t A = 0;
    if (args.epRankId < sharedExpertRankNum) { // 共享专家
        A = BS * EP_WORLD_SIZE_A2 / sharedExpertRankNum;
        localMoeExpertNum = 1;
    } else { // Moe专家
        A = BS * EP_WORLD_SIZE_A2 * localMoeExpertNum;
    }
    int64_t epWorldSize = EP_WORLD_SIZE_A2;
    auto outDataType = aclDataType::ACL_BF16;
    if (isQuant) {
        outDataType = aclDataType::ACL_INT8;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    std::vector<int64_t> scalesShape{moeExpertNum, H};              // dispatch need
    std::vector<int64_t> dynamicScalesShape{A};                     // dispatch need
    std::vector<int64_t> expertTokenNumsShape{localMoeExpertNum};   // dispatch need
    std::vector<int64_t> expandScalesShape{A}; // dispatch & combine
    std::vector<int64_t> expandXShape{A, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> expandIdxShape{BS * K};
    std::vector<int64_t> epSendCountsShape{localMoeExpertNum * EP_WORLD_SIZE_A2};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> tpSendCountsShape{1};
    std::vector<int64_t> xActiveMaskShape{BS};
    std::vector<int64_t> activationScaleShape{A};
    std::vector<int64_t> weightScaleShape{1, H};
    std::vector<int64_t> groupListShape{1};
    std::vector<int64_t> xShape{BS, H};

    void *scalesDeviceAddr = nullptr;           // dispatch need
    void *dynamicScalesDeviceAddr = nullptr;    // dispatch need
    void *expertTokenNumsDeviceAddr = nullptr;  // dispatch need
    void *expandScalesDeviceAddr = nullptr;     // dispatch & combine need
    void *expandXDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *epSendCountsDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *tpSendCountsDeviceAddr = nullptr;
    void *xActiveMaskDeviceAddr = nullptr; 
    void *activationScaleDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;

    aclTensor *scales = nullptr;            // dispatch need
    aclTensor *dynamicScales = nullptr;     // dispatch need
    aclTensor *expertTokenNums = nullptr;   // dispatch need
    aclTensor *expandScales = nullptr;      // dispatch & combine need
    aclTensor *expandX = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *epSendCounts = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *tpSendCounts = nullptr;
    aclTensor *xActiveMask = nullptr; 
    aclTensor *activationScale = nullptr;
    aclTensor *weightScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *x = nullptr;

    long long scalesShapeSize = GetShapeSize(scalesShape);                      // dispatch need
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);        // dispatch need
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);    // dispatch need
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);          // dispatch & combine need
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long epSendCountsShapeSize = GetShapeSize(epSendCountsShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long tpSendCountsShapeSize = GetShapeSize(tpSendCountsShape);
    long long xActiveMaskShapeSize = GetShapeSize(xActiveMaskShape);
    long long activationScaleShapeSize = GetShapeSize(activationScaleShape);
    long long weightScaleShapeSize = GetShapeSize(weightScaleShape);
    long long groupListShapeSize = GetShapeSize(groupListShape);
    long long xShapeSize = GetShapeSize(xShape);

    std::vector<float> scalesHostData(scalesShapeSize, 0);                      // dispatch need
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);        // dispatch need
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);  // dispatch need
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);          // dispatch & combine need
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<int32_t> expertIdsHostData(expertIdsShapeSize, 0);
    std::random_device rd; // 随机数设备
    std::mt19937 gen(rd()); // 以随机数设备作为种子的Mersenne Twister生成器
    std::uniform_int_distribution<> dis(sharedExpertRankNum, EP_WORLD_SIZE_A2 - 1);
    for (auto& val : expertIdsHostData) {
        val = dis(gen); // 为每个元素生成一个2到15之间的随机数
    }
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int32_t> epSendCountsHostData(epSendCountsShapeSize, 0);
    std::vector<int32_t> tpSendCountsHostData(tpSendCountsShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<int8_t> xActiveMaskHostData(xActiveMaskShapeSize, 0);
    std::vector<float> activationScaleHostData(activationScaleShapeSize,0);
    std::vector<float> weightScaleHostData(weightScaleShapeSize,0);
    std::vector<int32_t> groupListHostData(groupListShapeSize,0);
    std::vector<op::fp16_t> xHostData(xShapeSize, 0);

    auto ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);                                // dispatch need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         // dispatch need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); // dispatch need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             // dispatch & combine need
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epSendCountsHostData, epSendCountsShape, &epSendCountsDeviceAddr, aclDataType::ACL_INT32, &epSendCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpSendCountsHostData, tpSendCountsShape, &tpSendCountsDeviceAddr, aclDataType::ACL_INT32, &tpSendCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xActiveMaskHostData, xActiveMaskShape, &xActiveMaskDeviceAddr, aclDataType::ACL_BOOL, &xActiveMask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(activationScaleHostData, activationScaleShape, &activationScaleDeviceAddr, aclDataType::ACL_FLOAT, &activationScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT32, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    /******************************先调用dispatch,因为combine需要使用dispatch的数据********************************************/
    ret = aclnnMoeDistributeDispatchGetWorkspaceSize(x, expertIds, 
            (isQuant? scales : nullptr), xActiveMask, 
            expertScales, args.groupEpName, epWorldSize, args.epRankId, moeExpertNum, groupTpName, tpWorldSize, tpRankId, expertShardType, 
            sharedExpertNum,sharedExpertRankNum, quantMode, globalBS, expertTokenNumsType, expandX, dynamicScales, 
            expandIdx, expertTokenNums, epSendCounts, tpSendCounts, expandScales, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d \n", ret);
        return ret;
    }
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMoeDistributeDispatch(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatch failed. ret = %d \n", ret);
        return ret);

    /**************************************** 然后调用combine ********************************************/
    // 调用第一阶段接口
    ret = aclnnMoeDistributeCombineGetWorkspaceSize(expandX, expertIds,
                                                        expandIdx, epSendCounts,
                                                        expertScales, tpSendCounts,
                                                        xActiveMask, activationScale,
                                                        weightScale, groupList, expandScales, 
                                                        args.groupEpName, EP_WORLD_SIZE_A2, 
                                                        args.epRankId, moeExpertNum,
                                                        groupTpName, tpWorldSize, tpRankId,
                                                        expertShardType, sharedExpertNum, sharedExpertRankNum,globalBS, outDtype, commQuantMode,
                                                        groupListType, x,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributeCombineGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // 调用第二阶段接口
    ret = aclnnMoeDistributeCombine(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombine failed. ret = %d \n", ret);
        return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombine execute successfully.\n", args.rankId);

    // 释放device资源，需要根据具体API的接口定义修改
    if (scales != nullptr) {                // dispatch need
        aclDestroyTensor(scales);
    }
    if (dynamicScales != nullptr) {         // dispatch need
        aclDestroyTensor(dynamicScales);
    }
    if (expertTokenNums != nullptr) {       // dispatch need
        aclDestroyTensor(expertTokenNums);
    }
    if (expandScales != nullptr) {          // dispatch & combine need
        aclDestroyTensor(expandScales);
    }
    if (expandX != nullptr) {
        aclDestroyTensor(expandX);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (expandIdx != nullptr) {
        aclDestroyTensor(expandIdx);
    }
    if (epSendCounts != nullptr) {
        aclDestroyTensor(epSendCounts);
    }
    if (tpSendCounts != nullptr) {
        aclDestroyTensor(tpSendCounts);
    }
    if (expertScales != nullptr) {
        aclDestroyTensor(expertScales);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (epSendCountsDeviceAddr != nullptr) {
        aclrtFree(epSendCountsDeviceAddr);
    }
    if (tpSendCountsDeviceAddr != nullptr) {
        aclrtFree(tpSendCountsDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(args.stream);
    HcclCommDestroy(args.hcclEpComm);
    aclrtResetDevice(args.rankId);
    return 0;
}

void RunInProcessA2(int rank, int rankSize)
{
    // 1. acl init
    Args args;
    aclrtStream stream;
    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(rank));
    ACLCHECK(aclrtCreateStream(&stream));

    // 2. create HcclComm by rankFile
    char commName[128] = "";
    HcclComm hcclComm = nullptr;
    char *rankTableFile = getenv("RANK_TABLE_FILE");

    std::string rankTableFileStr(rankTableFile);
    std::cout << "rankTableFilePath is :" << rankTableFileStr << std::endl;
    int rank_id = rank + FIRST_RANK_ID;
    auto ret = HcclCommInitClusterInfo(rankTableFile, rank_id, &hcclComm);
    if (ret != HCCL_SUCCESS || hcclComm == nullptr) {
        std::cout << "HCCL CommInitClusterInfo ERROR" << ret << " should check rankTableFile config" << std::endl;
        return;
    }
    std::cout << "HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << rankSize
                    << ", hcclComm:" << hcclComm;
    HcclGetCommName(hcclComm, commName);
    if (commName == "") { std::cout << "rankTableFile CommName should not be null" << std::endl;}

    // 3. launch one process for MoeDistributeCombine
    args.rankId = rank;
    args.groupEpName = commName;
    args.hcclEpComm = hcclComm;
    args.epRankId = rank_id;
    args.stream = stream;
    LOG_PRINT("[INFO] rank = %d, groupEpName = %s, stream = %p\n", args.rankId, commName, args.stream);

    int res = launchOneProcessMoeDistributeDispatchA2(args);
    if (res != ACL_SUCCESS) {
        std::cout << "run launchOneProcessMoeDistributeDispatchA2 failed, ret = " << res << std::endl;
        return;
    }
}

int launchOneProcessMoeDistributeDispatchA3(Args_A3 &args){
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    char hcomTpName[128] = {0};
    ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT(
        "[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, context = %p\n",
        args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream, args.context
    );

    // 设置场景
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 3;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t quantMode = 0;
    int64_t globalBS = BS * EP_WORLD_SIZE_A3;
    int64_t expertTokenNumsType = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupList_type = 1;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        A = globalBS / sharedExpertRankNum;
    } else { 
        // Moe专家卡
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE_A3 - sharedExpertRankNum);
        A = globalBS * (localExpertNum < K ? localExpertNum : K);
    }

    /* 根据当前场景，构造device侧输入输出变量*/
    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandXDeviceAddr = nullptr;
    void *dynamicScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    void *tpRecvCountsDeviceAddr = nullptr;
    void *expandScalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *expertTokenNums = nullptr;
    aclTensor *epRecvCounts = nullptr;
    aclTensor *tpRecvCounts = nullptr;
    aclTensor *expandScales = nullptr;
    
    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> expandXShape{TP_WORLD_SIZE_A3 * A, H};
    std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE_A3 * A};
    std::vector<int64_t> expandIdxShape{BS * K};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE_A3 * localExpertNum * EP_WORLD_SIZE_A3};
    std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE_A3 * localExpertNum};
    std::vector<int64_t> expandScalesShape{A};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);
    int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
    int64_t expandXShapeSize = GetShapeSize(expandXShape);
    int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
    int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

    // 构造host侧变量
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        // 每个token发给moe专家{0, 1, ... k - 1}
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> scalesHostData(scalesShapeSize, 0.1);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
    std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);  
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); 
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    /* 声明算子执行必需变量 */
    uint64_t dispatchWorkspaceSize = 0;
    aclOpExecutor *dispatchExecutor = nullptr;
    void *dispatchWorkspaceAddr = nullptr;

    uint64_t combineWorkspaceSize = 0;
    aclOpExecutor *combineExecutor = nullptr;
    void *combineWorkspaceAddr = nullptr;   

    /* 依次执行dispatch及combine算子 */
    // 调用dispatch算子第一阶段接口
    ret = aclnnMoeDistributeDispatchGetWorkspaceSize(
        x, expertIds, 
        (quantMode > 0 ? scales : nullptr), nullptr, 
        expertScales, 
        hcomEpName, EP_WORLD_SIZE_A3, args.epRankId,
        moeExpertNum, hcomTpName, TP_WORLD_SIZE_A3,
        args.tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBS,
        expertTokenNumsType,
        expandX, dynamicScales,
        expandIdx, expertTokenNums,
        epRecvCounts, tpRecvCounts,
        expandScales, &dispatchWorkspaceSize,
        &dispatchExecutor
    );
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d\n", ret); return ret
    );
    // 根据dispatch算子第一阶段接口计算出的workspaceSize申请device内存
    if (dispatchWorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用dispatch算子第二阶段接口
    ret = aclnnMoeDistributeDispatch(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor, args.dispatchStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatch failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
    );

    // 调用combine算子第一阶段接口
    ret = aclnnMoeDistributeCombineGetWorkspaceSize(expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        hcomEpName, EP_WORLD_SIZE_A3, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE_A3, args.tpRankId,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupList_type,
        x, &combineWorkspaceSize, &combineExecutor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeDistributeCombineGetWorkspaceSize failed. ret = %d\n", ret); return ret
    );
    // 根据combine算子第一阶段接口计算出的workspaceSize申请device内存
    if (combineWorkspaceSize > 0) {
        ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用combine算子第二阶段接口
    ret = aclnnMoeDistributeCombine(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombine failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
    CHECK_RET(
        ret == ACL_SUCCESS, 
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
    );

    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatch and aclnnMoeDistributeCombine execute successfully.\n", args.rankId);

    // 释放device资源
    if (dispatchWorkspaceSize > 0) {
        aclrtFree(dispatchWorkspaceAddr);
    }
    if (combineWorkspaceSize > 0) {
        aclrtFree(combineWorkspaceAddr);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (scales != nullptr) {                
        aclDestroyTensor(scales);
    }
    if (expertScales != nullptr) {
        aclDestroyTensor(expertScales);
    }
    if (expandX != nullptr) {
        aclDestroyTensor(expandX);
    }
    if (dynamicScales != nullptr) {  
        aclDestroyTensor(dynamicScales);
    }
    if (expandIdx != nullptr) {
        aclDestroyTensor(expandIdx);
    }
    if (expertTokenNums != nullptr) {     
        aclDestroyTensor(expertTokenNums);
    }
    if (epRecvCounts != nullptr) {
        aclDestroyTensor(epRecvCounts);
    }
    if (tpRecvCounts != nullptr) {
        aclDestroyTensor(tpRecvCounts);
    }   
    if (expandScales != nullptr) {         
        aclDestroyTensor(expandScales);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }
    if (expertScalesDeviceAddr != nullptr) {
        aclrtFree(expertScalesDeviceAddr);
    }
    if (expandXDeviceAddr != nullptr) {
        aclrtFree(expandXDeviceAddr);
    }
    if (dynamicScalesDeviceAddr != nullptr) {
        aclrtFree(dynamicScalesDeviceAddr);
    }
    if (expandIdxDeviceAddr != nullptr) {
        aclrtFree(expandIdxDeviceAddr);
    }
    if (expertTokenNumsDeviceAddr != nullptr) {
        aclrtFree(expertTokenNumsDeviceAddr);
    }
    if (epRecvCountsDeviceAddr != nullptr) {
        aclrtFree(epRecvCountsDeviceAddr);
    }
    if (expandScalesDeviceAddr != nullptr) {
        aclrtFree(expandScalesDeviceAddr);
    }
    if (tpRecvCountsDeviceAddr != nullptr) {
        aclrtFree(tpRecvCountsDeviceAddr);
    }

    HcclCommDestroy(args.hcclEpComm);
    HcclCommDestroy(args.hcclTpComm);
    aclrtDestroyStream(args.dispatchStream);
    aclrtDestroyStream(args.combineStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    
    return 0;
}

int main(int argc, char *argv[])
{
    #ifndef ASCEND910_93
        CHECK_RET(false, LOG_PRINT("[INFO] This example is implemented based on Atlas A3 and must be run on Atlas A3 \n"); return -1);

        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE_A3][EP_WORLD_SIZE_A3];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE_A3; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE_A3; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE_A3 + tpId;
            }
        }
        // 初始化ep通信域，ep = 8 {0,2,4,6,8,10,12,14} {1,3,5,7,9,11,13,15}.
        HcclComm commsEp[TP_WORLD_SIZE_A3][EP_WORLD_SIZE_A3];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE_A3; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE_A3, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d\n", tpId, ret); return ret
            );
        }

        int32_t devicesTp[EP_WORLD_SIZE_A3][TP_WORLD_SIZE_A3];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE_A3; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE_A3; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE_A3 + tpId;
            }
        }
        // 初始化tp通信域，tp = 2 {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,13} {14,15}.
        HcclComm commsTp[EP_WORLD_SIZE_A3][TP_WORLD_SIZE_A3];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE_A3; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE_A3, devicesTp[epId], commsTp[epId]);
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d\n", epId, ret); return ret
            );
        }

        Args args[DEV_NUM];
        // 各线程调用各卡执行算子
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE_A3;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE_A3;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneProcessMoeDistributeDispatchA3, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        _exit(0);
    #endif

    char* env_rankID = getenv("FIRST_RANK_ID");
    if (!env_rankID) {
        std::cerr << "FIRST_RANK_ID环境变量未设置！\n";
        return 1;
    }
    FIRST_RANK_ID = std::stoi(std::string(env_rankID));
    std::cout << "FIRST_RANK_ID is: " << FIRST_RANK_ID << std::endl;

    // 所需的进程数量
    const int processCount = EP_WORLD_SIZE_A3;
    pid_t pids[processCount];

    for (int i = 0; i < processCount; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            // 子进程，完成任务后退出
            RunInProcessA2(i, processCount);
            exit(0);
        }
    }

    // 父进程等待所有子进程完成
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], NULL, 0);
    }

    return 0;
}
