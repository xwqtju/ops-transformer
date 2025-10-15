/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FALLBACK_MLA_PROLOG_V3_H
#define FALLBACK_MLA_PROLOG_V3_H

#include <vector>
#include "log/log.h"
#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "../../mla_prolog/op_host/fallback_mla_prolog.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

constexpr size_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;


constexpr size_t ATTR_QUERY_NORM_FLAG_INDEX = 3;
constexpr size_t ATTR_WEIGHT_QUANT_MODE_INDEX = 4;
constexpr size_t ATTR_KV_QUANT_MODE_INDEX = 5;
constexpr size_t ATTR_QUERY_QUANT_MODE_INDEX = 6;
constexpr size_t ATTR_CKVKR_REPO_MODE_INDEX = 7;
constexpr size_t ATTR_QUANT_SCALE_REPO_MODE_INDEX = 8;
constexpr size_t ATTR_TILE_SIZE_INDEX = 9;
constexpr size_t ATTR_K_NOPE_CLIP_ALPHA_INDEX = 10;
constexpr size_t ATTR_QC_QR_SCALE_INDEX = 11;
constexpr size_t ATTR_KC_SCALE_INDEX = 12;

struct MlaPrologV3FallBackParam : MlaPrologFallBackParam {
    const gert::Tensor *dequantScaleQNope = nullptr;
    const gert::Tensor *queryNormOptional = nullptr;
    const gert::Tensor *dequantScaleQNormOptional = nullptr;
    int queryNormFlag = 0;
    int weightQuantMode = 0;
    int kvQuantMode = 0;
    int queryQuantMode = 0;
    int ckvkrRepoMode = 0;
    int quantScaleRepoMode = 0;
    int tileSize = 0;
    double kNopeClipAlpha = 1.0f;
    double qcQrScale = 1.0f;
    double kcScale = 1.0f;
};

graphStatus GetMlaPrologV3OutputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param);
graphStatus MlaV3HostExecuteFunc(OpExecuteContext *host_api_ctx);

} // namespace fallback

#ifdef __cplusplus
}
#endif

#endif // FALLBACK_MLA_PROLOG_V3_H