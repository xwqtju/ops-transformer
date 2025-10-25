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
 * \file all_gather_add_infershape.cc
 * \brief
 */
#include "mc2_log.h"
#include "context_util.h"
#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_common_infershape.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeAllGatherAdd(gert::InferShapeContext* context)
{
    OP_LOGE_IF(
        InferShapeAllGatherAddCommon(context) != GRAPH_SUCCESS, GRAPH_FAILED, context->GetNodeName(),
        "infer shape excute failed.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAllGatherAdd(gert::InferDataTypeContext* context) {
    auto d_type = context->GetInputDataType(0);
    context->SetOutputDataType(0, d_type);
    context->SetOutputDataType(1, d_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AllGatherAdd).InferShape(InferShapeAllGatherAdd).InferDataType(InferDataTypeAllGatherAdd);
} // namespace ops
