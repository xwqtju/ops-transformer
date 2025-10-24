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
 * \file test_moe_token_unpermute_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_unpermute_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenUnpermuteGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteGradTiling TearDown" << std::endl;
    }
};

struct MoeTokenUnpermuteGradCompileInfo {
};

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_bf16_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 256 1 64 3 3 16 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_bf16_002) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 256 1 64 3 3 8 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_none_bf16_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "10 1 64 30 10 54 1 0 1 0 64 1 64 992 992 992 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_split_h_bf16_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 8192}, {30, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 8192}, {10, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 8192}, {30, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 8192 1 8192 1 3 16 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_fp16_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 256 1 64 3 3 16 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_none_fp16_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "10 1 64 30 10 54 1 0 1 0 64 1 64 992 992 992 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_split_h_fp16_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 8192 1 8192 1 3 16 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_split_h_fp16_002) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 8192 1 8192 1 3 8 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_fp32_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 64 30 10 54 1 0 3 0 128 1 64 3 3 8 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_none_fp32_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 64}, {10, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 64}, {30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData = "10 1 64 30 10 54 1 0 1 0 64 1 64 496 496 496 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_split_h_fp32_001) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 7168 2 1024 1 3 8 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteGradTiling, test_tiling_prob_not_none_split_h_fp32_002) {
    MoeTokenUnpermuteGradCompileInfo compileInfo = {}; // 根据tiling头文件中的compileInfo填入对应值，一般原先的用例里能找到
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteGrad", // op_name
                                            {
                                            // input info
                                            // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 8192}, {10, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{30,}, {30,}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // output info
                                            {
                                                {{{30, 8192}, {30, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{10, 3}, {10, 3}}, ge::DT_BF16, ge::FORMAT_ND}
                                            },
                                            // attr
                                            {{"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(0)}},
                                            &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData = "10 3 8192 30 10 54 1 0 3 0 7168 2 1024 1 3 16 261888 "; // tilingData（不确定的话跑下对应用例打印看看）
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}