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

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_2)
{
gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, 2}, {-1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 3}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, 2}, {-1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 3}, {-1, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 2}, {6}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, -1}, {6}, {6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_10)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 3}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_11)
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
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 3}, {8}, {8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_12)
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

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_13)
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
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 3}, {8}, {8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_14)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_15)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_17)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_18)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_19)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_20)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_21)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, -1}, {3, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_22)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, -1}, {3, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_23)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, -1}, {3, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_24)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 2}, {-1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, 3}, {-1, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_25)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_26)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_27)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 5}, {2, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_28)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_29)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_30)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_31)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_33)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_34)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_35)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_36)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{-3}, {-3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_37)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-3}, {-3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_38)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-3}, {-3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_39)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, -3}, {2, -3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_40)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, -3}, {2, -3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeInitRouting, moe_init_routing_infer_shape_41)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRouting",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, -3}, {2, -3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
