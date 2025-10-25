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
 * \file nsa_selected_attention_infer_tiling_register.cpp
 * \brief
 */

#include "nsa_selected_attention_infer_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingPrepareForNsaSelectAttentionInfer(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(NsaSelectedAttentionInfer)
    .Tiling(TilingNsaSelectAttentionInfer)
    .TilingParse<NsaSelectAttentionCompileInfo>(TilingPrepareForNsaSelectAttentionInfer)
    .TilingInputsDataDependency({6}, {gert::TilingPlacement::TILING_ON_HOST})
    .TilingInputsDataDependency({7}, {gert::TilingPlacement::TILING_ON_HOST});
} // namespace optiling
