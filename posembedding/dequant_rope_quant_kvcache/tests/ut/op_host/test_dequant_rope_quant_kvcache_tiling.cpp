/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * The code snippet comes from Huawei's open-source Ascend project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/dequant_rope_quant_kvcache_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

struct DequantRopeQuantKvcacheCompileInfo {};

class DequantRopeQuantKvcacheTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantRopeQuantKvcacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantRopeQuantKvcacheTiling TearDown" << std::endl;
    }
};


TEST_F(DequantRopeQuantKvcacheTiling, DequantRopeQuantKvcachetiling_01) {
  DequantRopeQuantKvcacheCompileInfo compileInfo = {};
  gert::TilingContextPara tilingContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{1, 320, 1280}, {1, 320, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                              {{{1, 320, 8, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 320, 1, 128}, {1, 320, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{1, 1024, 1, 128}, {1, 1024, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({8 * 128, 1 * 128, 1 * 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            },
                                            &compileInfo);

  uint64_t expectTilingKey = 0;
  string expectTilingData = "8 1 128 128 256 512 44 1024 320 1024 128 128 64 64 5 5 1 0 0 0 0 ";
  std::vector<size_t> expectWorkspaces = {16777216};
  ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
