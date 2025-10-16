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
 * \file ts_mla_prolog_v3_tc_full_quant.cpp
 * \brief MlaPrologV3 全量化KvCache非量化用例.
 */

#include "ts_mla_prolog_v3.h"

using Tensor = ops::adv::tests::utils::Tensor;

class Ts_MlaPrologV3_Ascend910B2_tc_full_quant : public Ts_MlaPrologV3_WithParam_Ascend910B2 {};

TEST_P(Ts_MlaPrologV3_Ascend910B2_tc_full_quant, Tc_Case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_MlaPrologV3_Full_Quant_Case = ::testing::Values(
    //DataType不符合要求
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00000", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {48, 16, 7168}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00001", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00002", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 12544}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00003", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00004", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 576}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00005", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {Tensor("rmsnormGammaCq", {1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00006", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {Tensor("rmsnormGammaCkv", {512}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00007", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {Tensor("ropeSin", {48, 16, 64}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00008", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {Tensor("ropeCos", {48, 16, 64}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00009", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {Tensor("cacheIndex", {48, 16}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00010", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00011", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {16, 128, 1, 64}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00012", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {768, 1}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00013", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00014", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 12544}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00015", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1 ,576}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00016", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT16, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    //维数不符合要求
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00017", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {Tensor("tokenX", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00018", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {Tensor("weightDq", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00019", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {Tensor("weightUqQr", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00020", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {Tensor("weightUk", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00021", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00022", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {Tensor("rmsnormGammaCq", {1, 1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00023", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {Tensor("rmsnormGammaCkv", {1, 1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00024", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {Tensor("ropeSin", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00025", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {Tensor("ropeCos", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00026", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {Tensor("cacheIndex", {1, 1, 1}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00027", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00028", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {1}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00029", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00030", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00031", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00032", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00033", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    //输入shape不符合要求
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00034", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {Tensor("weightDq", {8000, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00035", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {Tensor("weightDq", {7168, 1536}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDq */
                   {Tensor("weightUqQr", {1536, 20000}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00036", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {Tensor("weightUk", {64, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00037", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(48, 16, 7168, 1536, 512, 128, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {Tensor("weightUk", {64, 128, 256}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {48, 16, 128, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00038", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {Tensor("weightDkvKr", {7168, 577}, "1", ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ, Tensor::TensorType::REQUIRED_INPUT)},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00039", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {Tensor("rmsnormGammaCq", {1537}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00040", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {Tensor("rmsnormGammaCkv", {513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00041", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {Tensor("ropeSin", {2, 1, 64}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00042", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(2, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {Tensor("ropeCos", {2, 1, 65}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {2, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00043", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(11, 1, 1, 7168, 1536, 512, 64, 128, 64,             /* T, B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {Tensor("cacheIndex", {12}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {11, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00044", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 1024,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {1024, 128, 1, 513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {1024, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00045", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 1024,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {1024, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {Tensor("krCache", {1024, 16, 1, 513}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {1024, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00046", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {Tensor("dequantScaleX", {8, 1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00047", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {Tensor("dequantScaleWDq", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00048", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 32, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {Tensor("dequantScaleWUqQr", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 32, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00049", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {Tensor("dequantScaleWDkvKr", {1, 20000}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1536}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           ),
    MlaPrologV3Case("MlaPrologV3_Tc_Full_Quant_00050", true,                    /* CaseName, Enable */
           "",                                                     /* DebugInfo */
           OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                  ExpectInfo(false,                                 /* ExpectSuccess */
                             ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           MlaPrologV3Param(1, 1, 7168, 1536, 512, 64, 128, 64,             /* B, S, He, Hcq, Hckv, N, D, Dr */
                   1024, 1, 128, 16,                                   /* Skv, Nkv, BlockSize, BlockNum */
                   CacheModeType::PA_BSND,                         /* CacheModeType */
                   0.00001f, 0.00001f,                             /* rmsnormEpsilonCq, rmsnormEpsilonCkv */
                   {},                                             /* tokenX */
                   {},                                             /* weightDq */
                   {},                                             /* weightUqQr */
                   {},                                             /* weightUk */
                   {},                                             /* weightDkvKr */
                   {},                                             /* rmsnormGammaCq */
                   {},                                             /* rmsnormGammaCkv */
                   {},                                             /* ropeSin */
                   {},                                             /* ropeCos */
                   {},                                             /* cacheIndex */
                   {Tensor("kvCache", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT)},                                             /* kvCache */
                   {},                                             /* krCache */
                   {},                                             /* dequantScaleX */
                   {},                                             /* dequantScaleWDq */
                   {},                                             /* dequantScaleWUqQr */
                   {},                                             /* dequantScaleWDkvKr */
                   {},                                             /* quantScaleCkv */
                   {},                                             /* quantScaleCkr */
                   {Tensor("smoothScalesCq", {1, 1537}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT)},                                             /* smoothScalesCq */
                   {Tensor("kvCacheOut", {16, 128, 1, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* kvCacheOut */
                   {},                                             /* krCacheOut */
                   {Tensor("query", {1, 1, 64, 512}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT)},                                             /* queryOut */
                   {},                                             /* queryRopeOut */
                   {})                                             /* dequantScaleQNopeOut */
           )
);
INSTANTIATE_TEST_SUITE_P(MlaPrologV3, Ts_MlaPrologV3_Ascend910B2_tc_full_quant, Tc_MlaPrologV3_Full_Quant_Case);