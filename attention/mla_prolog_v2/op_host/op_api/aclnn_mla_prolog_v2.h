/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACLNN_MLA_PROLOG_V2_H
 #define ACLNN_MLA_PROLOG_V2_H
 
 #include "aclnn/acl_meta.h"
 #include "aclnn/aclnn_base.h"
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /**
  * @brief The first interface of aclnnMlaPrologV2WeightNz calculates the workspace size based on the specific calculation process.
  * @domain aclnn_ops_infer
  */
 __attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologV2GetWorkspaceSize(
     const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
     const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
     const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cacheIndex,
     aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *dequantScaleXOptional,
     const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional,
     const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional,
     const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
     double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
     const aclTensor *queryOut, const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOutOptional,
     uint64_t *workspaceSize, aclOpExecutor **executor);
 
 /**
  * @brief The second interface of aclnnMlaPrologV2WeightNz is used to perform calculations.
  */
 __attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologV2(void *workspace,
                                                                     uint64_t workspaceSize,
                                                                     aclOpExecutor *executor,
                                                                     const aclrtStream stream);
 
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // ACLNN_MLA_PROLOG_V2_H