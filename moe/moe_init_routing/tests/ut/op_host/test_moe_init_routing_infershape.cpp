/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_init_routing_proto.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeInitRouting : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRouting SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRouting TearDown" << std::endl;
    }
};

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3}, {8}, {8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}