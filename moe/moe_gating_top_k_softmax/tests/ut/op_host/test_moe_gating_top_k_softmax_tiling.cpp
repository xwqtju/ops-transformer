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
#include "../../../op_host/moe_gating_top_k_softmax_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeGatingTopKSoftmaxTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxTiling TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_001) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_002) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 5;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_003) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_004) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_005) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 14;
    string expectTilingData = "1099511627790 68719476752 34359738400 274877906952 17179869188 4294967297 17179869188 93823560581124 549755857578 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_006) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_007) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24}, {1, 24}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_008) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24}, {1, 24}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_009) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 1, 5000}, {1, 1, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_010) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 24, 8}, {1, 24, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 8}, {1, 24, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 8}, {1, 24, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 8}, {1, 24, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "103079215113 34359738376 17179869216 103079215112 4294967297 4294967297 4294967297 365072220161 137438953642 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_011) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "1099511627785 34359738376 17179869216 274877906952 17179869188 4294967297 17179869188 365072220164 549755814058 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_012) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "1099511627785 34359738376 17179869216 274877906952 17179869188 4294967297 17179869188 365072220164 549755814058 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_013) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 13;
    string expectTilingData = "1099511627789 68719476752 34359738400 274877906952 17179869188 4294967297 17179869188 93823560581124 549755857578 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_014) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 15;
    string expectTilingData = "1099511627791 1099511628032 34359738624 274877906952 17179869188 4294967297 17179869188 93823560581124 4398046554794 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_015) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 16;
    string expectTilingData = "1099511627792 1099511628032 34359738624 274877906952 17179869188 4294967297 17179869188 93823560581124 4398046554794 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxTiling, moe_gating_top_k_softmax_tiling_016) {
    optiling::MoeGatingTopKSoftmaxCompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmax",
                                              {
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 256}, {1, 256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 17;
    string expectTilingData = "1099511627793 1099511628032 34359738624 274877906952 17179869188 4294967297 17179869188 93823560581124 4398046554794 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}