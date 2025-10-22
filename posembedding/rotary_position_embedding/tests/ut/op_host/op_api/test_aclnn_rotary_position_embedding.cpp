/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_rotary_position_embedding.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_rotary_position_embedding_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_rotary_position_embedding_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_rotary_position_embedding_test TearDown" << endl;
    }
};

// dtype fp32
TEST_F(l2_rotary_position_embedding_test, Ascend910B2_rotary_position_embedding_fp32)
{
    auto x = TensorDesc({8, 1, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto cos = TensorDesc({1, 1, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    auto sin = TensorDesc({1, 1, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 8);
    int64_t mode = 2;
    auto out = TensorDesc({8, 1, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotaryPositionEmbedding, INPUT(x, cos, sin, mode), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}