#!/usr/bin/python3
# -*- coding:utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import sys
import os
import numpy as np
import random
np.random.seed(0)

def gen_input_data(group_num, e, m, k, n):
    e = int(e)
    m = int(m)
    k = int(k)
    n = int(n)
    group_num = int(group_num)
    x = np.random.uniform(-5, 5, [m, k]).astype(np.int8)
    weight = np.random.uniform(-5, 5, [e, k, n]).astype(np.int8)
    xScale = np.random.uniform(-5, 5, [m,]).astype(np.float32)
    weightScale = np.random.uniform(-5, 5, [e, n]).astype(np.float32)
    groupedList = np.linspace(1, m, group_num, dtype=np.int64)
    return x, weight, xScale, weightScale, y, groupedList

def gen_golden_data(group_num, e, m, k, n):
    x, weight, xScale, weightScale, y, groupedList = gen_input_data(group_num, e, m, k, n)
    x.tofile(f"x.bin")
    weight.tofile(f"weight.bin")
    weightScale.tofile(f"weightScale.bin")
    xScale.tofile(f"xScale.bin")
    y.tofile(f"y.bin")
    groupedList.tofile(f"groupedList.bin")

if __name__ == "__main__": 
    os.system("rm -rf *.bin")
    gen_golden_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

