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
 * \file experiment_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief MoeDistributeDispatch operator interface implementation.

* @par Inputs
* Five inputs, including:
* @li x: A tensor. Support dtype: float16,bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expertIds: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape supports (BS, K), support format: ND.
* @li scales: An optional tensor. Support dtype: float32, dimension must be 2, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li expert_scales: An optional tensor. Support dtype: float32. Shape supports (BS, K), support format: ND.

* @par Attributes
* @li group_ep: Required. Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Required. Input ep comm world size, dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, dtype: int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: int64.
* @li expert_shard_type: Input moe shard type, dtype: int64.
* @li shared_expert_num: Input shared expert num, dtype: int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: int64.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), 1 (static quantization), and 2 (dynamic quantization). dtype: int64.
* @li global_bs: Input global batch size, dtype: int64.
* @li expert_token_nums_type: Input expert token nums type, dtype: int64.

* @par Outputs
* Seven outputs, including:
* @li expand_x: A tensor. Result of each expert after dispatching. Support dtype: float16,bfloat16,int8. Shape supports (A, H), support format: ND.
* @li dynamic_scales: If quant is enabled, scale value of each token. A tensor. Support dtype: float32. Shape supports (A, ), support format: ND.
* @li expand_idx: A tensor. Support dtype: int32. Shape supports (BS*K, ), support format: ND.
* @li expert_token_nums: A tensor. Tokens nums of expand_x. Support dtype: int64, support format: ND.
* @li ep_recv_count: A tensor. Received token nums after dispatching. Support dtype: int32, support format: ND.
* @li tp_recv_count: A tensor. Received token nums after allgather. Support dtype: int32, support format: ND.
* @li expand_scales: A tensor. Scales of each token to sum for combine. Support dtype: float32. Shape supports (A, ), support format: ND.
*/
REG_OP(MoeDistributeDispatch)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(wait_cost, TensorType({DT_INT64}))
    .OUTPUT(expand_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16}))
    .OUTPUT(dynamic_scales, TensorType({DT_FLOAT}))
    .OUTPUT(expand_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_token_nums, TensorType({DT_INT64}))
    .OUTPUT(ep_recv_count, TensorType({DT_INT32}))
    .OUTPUT(tp_recv_count, TensorType({DT_INT32}))
    .OUTPUT(expand_scales, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(expert_token_nums_type, Int, 1)
    .OP_END_FACTORY_REG(MoeDistributeDispatch)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
