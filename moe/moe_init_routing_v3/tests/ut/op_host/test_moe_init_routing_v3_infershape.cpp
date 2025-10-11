// /**
//  * This program is free software, you can redistribute it and/or modify.
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /* !
//  * \file test_moe_init_routing_v3_infershape.cpp
//  * \brief
//  */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MoeInitRoutingV3 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3 TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 30})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {29}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{3, 128}, {3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 128}, {24}, {7}, {24}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_4)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_5)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_6)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {8 * 512 * 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_7)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7, 1}, {7, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_8)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8, 1024}, {8 , 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_9)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_10)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// int_64:  [-9223372036854775808, 9223372036854775807]
// uint_64: [                   0, 18446744073709551615]
TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_11)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{9223372036854775807, 1}, {9223372036854775807, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{-1, 1}, {-1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{9223372036854775807, 1}, {9223372036854775807}, {135}, {9223372036854775807}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_12)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_13)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{-1, 7242}, {-1, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_14)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_15)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_16)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_17)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{135, 1}, {135, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {135}, {15114054}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_18)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{4, 14}, {4, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{20, 14}, {20}, {256, 2}, {40}}; 
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_19)
// {
//     ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3"), nullptr);
//     auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MoeInitRoutingV3")->infer_datatype;

//     if (data_type_func != nullptr) {
//         ge::DataType fp16_ref = ge::DT_FLOAT16;
//         ge::DataType fp32_ref = ge::DT_FLOAT;
//         ge::DataType int32_ref = ge::DT_INT32;
//         ge::DataType int64_ref = ge::DT_INT64;
//         vector<int64_t> active_expert_range{1, 8};
//         auto context_holder = gert::InferDataTypeContextFaker()
//                                   .IrInputNum(4)
//                                   .NodeIoNum(4, 5)
//                                   .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
//                                               {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(-1)},
//                                               {"expert_num", ge::AnyValue::CreateFrom<int64_t>(-1)},
//                                               {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(0)},
//                                               {"expert_tokens_num_type", ge::AnyValue::CreateFrom<int64_t>(0)},
//                                               {"expert_tokens_num_flag", ge::AnyValue::CreateFrom<bool>(false)},
//                                               {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(-1)},
//                                               {"active_expert_range",
//                                                ge::AnyValue::CreateFrom<std::vector<int64_t>>(active_expert_range)},
//                                               {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
//                                   .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                                   .InputDataTypes({&fp16_ref, &int32_ref, &fp32_ref})
//                                   .OutputDataTypes({&fp16_ref, &int32_ref, &int64_ref, &fp32_ref})
//                                   .Build();
//         auto context = context_holder.GetContext<gert::InferDataTypeContext>();
//         EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
//         ASSERT_NE(context, nullptr);
//         EXPECT_EQ(context->GetInputDataType(0), fp16_ref);
//         printf("expended_x: %d", context->GetInputDataType(0));
//         EXPECT_EQ(context->GetOutputDataType(0), fp16_ref);
//         EXPECT_EQ(context->GetOutputDataType(1), int32_ref);
//         EXPECT_EQ(context->GetOutputDataType(2), int64_ref);
//         EXPECT_EQ(context->GetOutputDataType(4), fp32_ref);
//     }
// }

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_00)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_01)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{135, 1}, {135, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_02)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{135, 1}, {135, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_03)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_04)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_05)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_06)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2087}, {2087}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_07)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{2, 192}, {2, 192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV3, moe_init_routing_v3_infer_shape_08)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV3",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7, 191}, {7, 191}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(20)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)}, 
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}, 
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}, 
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})}, 
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, 
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}