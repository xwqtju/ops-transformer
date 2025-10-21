/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_permute_with_routing_map_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenPermuteWithRoutingMapTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithRoutingMapTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteWithRoutingMapTiling, MoeTokenPermuteWithRoutingMap_tiling_float) {
    optiling::MoeTokenPermuteWithRoutingMapCompileInfo compileInfo = {40, 192 * 1024, 0};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteWithRoutingMap",
                                            {
                                                {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                {"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 8 3 0 0 0 1 6 1 6 6 6 1 6 6 8160 0 0 0 0 0 0 1024 2 2 0 1 0 20 2 2976 5952 17856 1 1 1 1 1 0 5952 2 2976 1 1 6 95232 17856 3 2 1 2 2 0 0 0 0 0 2 3 ";
    std::vector<size_t> expectWorkspaces = {16777376};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
