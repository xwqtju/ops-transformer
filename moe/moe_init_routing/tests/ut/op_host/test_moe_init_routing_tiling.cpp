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
#include "../../../op_host/moe_init_routing_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeInitRoutingTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_01)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_02)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 1234}, {32, 1234}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 1234}, {32, 1234}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{7404, 1024}, {7404, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39488}, {39488}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39488}, {39488}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_03)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_04)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_05)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_06)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{32, 1024}, {32, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{32, 256}, {32, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{1536, 1024}, {1536, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_07)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{40, 5}, {40, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{120000, 5}, {120000, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_08)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{40, 5000}, {40, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{40, 3000}, {40, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{30000, 5000}, {30000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{120000}, {120000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 4; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_09)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 500}, {117000, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_10)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 500}, {39, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 500}, {117000, 500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_11)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000, 2}, {117000, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_12)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117}, {117}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_13)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117}, {117}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_14)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5000}, {117000, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_15)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{39, 5000}, {39, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{39, 3000}, {39, 3000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{117000, 5}, {117000, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{117000}, {117000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(39)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_16)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{1, 577999}, {1, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{0, 577999}, {0, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_17)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{8, 2}, {8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingTiling, moe_init_routing_tiling_18)
{
    optiling::MoeInitRoutingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRouting", // op_name
                                              {
                                                  // input info
                                                  {{{1, 577999}, {1, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 1000000}, {1, 1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{0, 577999}, {0, 577999}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1000000}, {1000000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {
                                                  {{"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},

                                              },

                                              &compileInfo);
    int64_t expectTilingKey = 2; // tilngkey
    string expectTilingData = "64 2 5 3 1 6 1 6 6 6 1 6 6 6848 0 2048 6 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 3 6 2 1 1 1 2 2 "
                              "2 1 1 1 2 2 5 1 ";      // tilingData
    std::vector<size_t> expectWorkspaces = {16781480}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

