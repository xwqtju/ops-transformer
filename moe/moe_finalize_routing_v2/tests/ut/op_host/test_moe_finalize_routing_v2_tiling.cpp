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
#include "../../../op_host/moe_finalize_routing_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeFinalizeRoutingTilingV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingTilingV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingTilingV2 TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_float) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20015;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_bfloat16) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_float16) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_cut_h_float) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{4096 * 4, 8934}, {4096 * 4, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8934}, {8, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 4}, {4096, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);

    int64_t expectTilingKey = 20009;
    string expectTilingData = "64 64 0 8 4096 8934 5432 3502 1 0 0 0 4 64 128 1 1 64 128 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_cut_h_float16) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{4096 * 2, 15080}, {4096 * 2, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096 * 2}, {4096 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 15080}, {16, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 2}, {4096, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 2}, {4096, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 64 0 16 4096 15080 15080 15080 0 0 0 0 2 64 64 1 1 64 64 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_cut_hk_float) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{1024 * 258, 8936}, {1024 * 258, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{340, 8936}, {340, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 8936 8088 848 1 256 2 2 258 16 32 1 1 16 32 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_cut_k_float) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{1024 * 258, 2560}, {1024 * 258, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{340, 2560}, {340, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 2560 2560 2560 0 256 2 2 258 16 4 5 1 16 4 5 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRouting_tiling_float_newwork) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{1024 * 258, 2560}, {1024 * 258, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{340, 2560}, {340, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 2560 2560 2560 0 256 2 2 258 16 4 5 1 16 4 5 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_10000_001) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{884, 5120}, {884, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{884}, {884}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{68, 5120}, {68, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{68, 5120}, {68, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 5120}, {16, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{68, 13}, {68, 13}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{68, 13}, {68, 13}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{68, 5120}, {68, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 34 0 16 68 5120 5120 5120 0 0 0 0 13 2 2 1 1 2 2 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20000_001) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{380, 10240}, {380, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{380}, {380}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{95, 10240}, {95, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{95, 10240}, {95, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 10240}, {16, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{95, 4}, {95, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{95, 4}, {95, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{95, 10240}, {95, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20009;
    string expectTilingData = "64 48 0 16 95 10240 5432 4808 1 0 0 0 4 2 4 1 1 1 2 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_30000_001) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{512, 46}, {512, 46}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{128, 46}, {128, 46}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{128, 46}, {128, 46}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1191, 46}, {1191, 46}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{128, 4}, {128, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{128, 46}, {128, 46}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_40000_001) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    // gert::TilingContextPara::TensorDescription emptyTensorDes{};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{18142, 68}, {18142, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{18142}, {18142}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{386, 68}, {386, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 56 1 0 386 68 68 68 0 0 0 0 47 7 1 7 7 1 1 1 1 0 1 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_drop_pad_mode) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{18142, 68}, {18142, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{18142}, {18142}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{386, 68}, {386, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(11)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_dtype_0) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{18142, 68}, {18142, 68}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{18142}, {18142}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{386, 68}, {386, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_dtype_1) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{18142, 68}, {18142, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{18142}, {18142}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{386, 68}, {386, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_shape_001) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{18142, 68}, {18142, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{18142}, {18142}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{386, 68}, {386, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_shape_002) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{18142, 68}, {18142, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{18142}, {18142}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 47}, {386, 47}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{386, 48}, {386, 48}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{386, 68}, {386, 68}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_x1_x2) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{512, 46}, {512, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{128, 46}, {128, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1191, 46}, {1191, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{128, 4}, {128, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{128, 46}, {128, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_bias_expert_idx) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{512, 46}, {512, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1191, 46}, {1191, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{128, 4}, {128, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{128, 46}, {128, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_invalid_e_k) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{3, 2, 46}, {3, 2, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{3, 46}, {3, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{128, 4}, {128, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{128, 46}, {128, 46}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 64 0 1191 128 46 46 46 0 0 0 0 4 2 1 2 2 2 1 2 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20038_bigK_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{466, 385, 131}, {466, 385, 131}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{331585}, {331585}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{799, 131}, {799, 131}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{799, 131}, {799, 131}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{466, 131}, {466, 131}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{799, 415}, {799, 415}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{799, 415}, {799, 415}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{799, 131}, {799, 131}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20002;
    string expectTilingData = "64 62 0 466 799 131 131 131 0 256 159 2 415 13 1 13 13 6 1 6 6 0 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20039_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{21573, 5046}, {21573, 5046}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{21573}, {21573}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{423, 5046}, {423, 5046}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{423, 5046}, {423, 5046}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{463, 5046}, {463, 5046}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{423, 51}, {423, 51}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{423, 51}, {423, 51}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{423, 5046}, {423, 5046}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 61 0 463 423 5046 5046 5046 0 0 0 0 51 7 7 1 1 3 3 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20040_cutH_k2_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{456, 242, 6108}, {456, 242, 6108}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{872}, {872}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{436, 6108}, {436, 6108}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{436, 6108}, {436, 6108}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{456, 6108}, {456, 6108}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{436, 2}, {436, 2}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{436, 2}, {436, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{436, 6108}, {436, 6108}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 63 0 456 436 6108 6108 6108 0 0 0 0 2 7 7 1 1 2 2 1 1 0 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20041_cutH_k4_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{189, 72, 6896}, {189, 72, 6896}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{24}, {24}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{6, 6896}, {6, 6896}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{6, 6896}, {6, 6896}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{189, 6896}, {189, 6896}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{6, 4}, {6, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{6, 4}, {6, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{6, 6896}, {6, 6896}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20011;
    string expectTilingData = "64 6 0 189 6 6896 4192 2704 1 0 0 0 4 1 2 1 1 1 2 1 1 0 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20042_cutH_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{248, 1, 6825}, {248, 1, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{157042}, {157042}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{674, 6825}, {674, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{674, 6825}, {674, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{248, 6825}, {248, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{674, 233}, {674, 233}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{674, 233}, {674, 233}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{674, 6825}, {674, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20014;
    string expectTilingData = "64 62 0 248 674 6825 6816 9 1 0 0 0 233 11 22 1 1 3 6 1 1 0 0 3 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20043_allBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{316, 96, 52}, {316, 96, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{60088}, {60088}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{259, 52}, {259, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{259, 52}, {259, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{316, 52}, {316, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{259, 232}, {259, 232}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{259, 232}, {259, 232}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{259, 52}, {259, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 52 0 316 259 52 52 52 0 0 0 0 232 5 1 5 5 4 1 4 4 0 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20044_bigK_noBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{22099, 616}, {22099, 616}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{22099}, {22099}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{77, 616}, {77, 616}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{77, 616}, {77, 616}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{77, 287}, {77, 287}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{77, 287}, {77, 287}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{77, 616}, {77, 616}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20021;
    string expectTilingData = "64 39 0 0 77 616 616 616 0 256 31 2 287 2 1 2 2 1 1 1 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20045_noBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{20382, 7380}, {20382, 7380}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{20382}, {20382}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{237, 7380}, {237, 7380}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{237, 7380}, {237, 7380}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{237, 86}, {237, 86}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{237, 86}, {237, 86}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{237, 7380}, {237, 7380}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData =  "64 60 0 0 237 7380 7380 7380 0 0 0 0 86 4 4 1 1 1 1 1 1 0 0 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20046_cutH_k2_noBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{1712, 8072}, {1712, 8072}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1712}, {1712}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{856, 8072}, {856, 8072}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{856, 8072}, {856, 8072}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{856, 2}, {856, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{856, 2}, {856, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{856, 8072}, {856, 8072}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20049;
    string expectTilingData = "64 62 0 0 856 8072 8072 8072 0 0 0 0 2 14 14 1 1 2 2 1 1 0 0 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20047_cutH_k4_noBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{2740, 7887}, {2740, 7887}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{2740}, {2740}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{685, 7887}, {685, 7887}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{685, 7887}, {685, 7887}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{685, 4}, {685, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{685, 4}, {685, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{685, 7887}, {685, 7887}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20049;
    string expectTilingData = "64 63 0 0 685 7887 7887 7887 0 0 0 0 4 11 11 1 1 3 3 1 1 0 0 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20048_cutH_noBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{6794, 7507}, {6794, 7507}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{6794}, {6794}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{43, 7507}, {43, 7507}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{43, 7507}, {43, 7507}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{43, 158}, {43, 158}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{43, 158}, {43, 158}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{43, 7507}, {43, 7507}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20036;
    string expectTilingData = "64 43 0 0 43 7507 7507 7507 0 0 0 0 158 1 1 1 1 1 1 1 1 0 0 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTilingV2, MoeFinalizeRoutingTilingV2_20049_allBias_noBias_mixPrec) {
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                            {
                                                {{{259, 251, 142}, {259, 251, 142}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{139360}, {139360}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{670, 142}, {670, 142}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{670, 142}, {670, 142}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{670, 208}, {670, 208}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{670, 208}, {670, 208}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{670, 142}, {670, 142}}, ge::DT_BF16, ge::FORMAT_ND},
                                            },
                                            {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)}},
                                            &compileInfo);
    int64_t expectTilingKey = 20049;
    string expectTilingData = "64 61 0 0 670 142 142 142 0 0 0 0 208 11 1 11 11 10 1 10 10 0 0 3 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}