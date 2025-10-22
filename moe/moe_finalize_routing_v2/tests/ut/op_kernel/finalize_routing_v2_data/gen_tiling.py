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
import numpy as np
import sys

case0_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 2004, 0, 1] #db float
case1_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 2005, 0, 1] #db float16
case2_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 2003, 0, 1] #db bfloat16
case3_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 2, 26, 52, 1, 1, 10, 20, 1, 1, 2006, 0, 1] #cuthk2 float
case4_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 2, 26, 52, 1, 1, 10, 20, 1, 1, 2007, 0, 1] #cuthk2 float16
case5_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2008, 0, 1] #cuthk2 bfloat16
case6_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 4, 26, 52, 1, 1, 10, 20, 1, 1, 2009, 0, 1] #cuthk4 float
case7_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 4, 26, 52, 1, 1, 10, 20, 1, 1, 20010, 0, 1] #cuthk4 float16
case8_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0,
                0, 0, 4, 1, 3, 1, 1, 1, 3, 1, 1, 20011, 0, 1] #cuthk4 bfloat16
case9_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 3, 26, 52, 1, 1, 10, 20, 1, 1, 20012, 0, 1] #cuth float
case10_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 3, 26, 52, 1, 1, 10, 20, 1, 1, 20013, 0, 1] #cuth float16
case11_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 3, 1, 2, 1, 1, 1, 2, 1, 1, 20014, 0, 1] #cuth bfloat16
case12_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20000, 0, 1] #CUTK float
case13_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20001, 0, 1] #CUTK float16
case14_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20002, 0, 1] #CUTK bfloat16

case15_params = [40, 39, 0, 471, 933, 30, 30, 30, 0, 0, 0, 0, 4, 24,
                4, 7, 3, 21, 3, 7, 7, 20015, 0, 1] #AllBias float32

case16_params = [40, 38, 0, 702, 340, 30, 30, 30, 0, 0, 0, 0, 5, 9,
                1, 9, 9, 7, 1, 7, 7, 20016, 0, 1] #AllBias float16

case17_params = [40, 35, 0, 592, 105, 30, 30, 30, 0, 0, 0, 0, 6, 3,
                1, 3, 3, 3, 1, 3, 3, 20017, 0, 1] #AllBias bfloat16

case18_params = [40, 12, 0, 16, 12, 20, 10, 10, 1, 0, 0, 0, 4, 1,
                1, 1, 1, 1, 1, 1, 1, 20018, 0, 1] #optimized float32

case19_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 20023, 1, 1] #db float
case20_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 20024, 1, 1] #db float16
case21_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 20022, 1, 1] #db bfloat16
case22_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 2, 26, 52, 1, 1, 10, 20, 1, 1, 20025, 1, 1] #cuthk2 float
case23_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 2, 26, 52, 1, 1, 10, 20, 1, 1, 20026, 1, 1] #cuthk2 float16
case24_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 20027, 1, 1] #cuthk2 bfloat16
case25_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 4, 26, 52, 1, 1, 10, 20, 1, 1, 20028, 1, 1] #cuthk4 float
case26_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 4, 26, 52, 1, 1, 10, 20, 1, 1, 20029, 1, 1] #cuthk4 float16
case27_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0,
                0, 0, 4, 1, 3, 1, 1, 1, 3, 1, 1, 20030, 1, 1] #cuthk4 bfloat16
case28_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 3, 26, 52, 1, 1, 10, 20, 1, 1, 20031, 1, 1] #cuth float
case29_params = [40, 40, 0, 10, 1024, 30, 20, 10, 1, 0, 
                0, 0, 3, 26, 52, 1, 1, 10, 20, 1, 1, 20032, 1, 1] #cuth float16
case30_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 3, 1, 2, 1, 1, 1, 2, 1, 1, 20033, 1, 1] #cuth bfloat16
case31_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20019, 1, 1] #CUTK float
case32_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20020, 1, 1] #CUTK float16
case33_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20021, 1, 1] #CUTK bfloat16

case34_params = [40, 39, 0, 471, 933, 30, 30, 30, 0, 0, 0, 0, 4, 24,
                4, 7, 3, 21, 3, 7, 7, 20034, 1, 1] #AllBias float32

case35_params = [40, 38, 0, 702, 340, 30, 30, 30, 0, 0, 0, 0, 5, 9,
                1, 9, 9, 7, 1, 7, 7, 20035, 1, 1] #AllBias float16

case36_params = [40, 35, 0, 592, 105, 30, 30, 30, 0, 0, 0, 0, 6, 3,
                1, 3, 3, 3, 1, 3, 3, 20036, 1, 1] #AllBias bfloat16

# for bfloat16+float32 testcases
case37_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20037, 0, 1] #CUTK bfloat16+float32
case38_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 20038, 0, 1] #db bfloat16+float32
case39_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 20039, 0, 1] #cuthk2 bfloat16+float32
case40_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0,
                0, 0, 4, 1, 3, 1, 1, 1, 3, 1, 1, 20040, 0, 1] #cuthk4 bfloat16+float32
case41_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 3, 1, 2, 1, 1, 1, 2, 1, 1, 20041, 0, 1] #cuth bfloat16+float32
case42_params = [40, 35, 0, 592, 105, 30, 30, 30, 0, 0, 0, 0, 6, 3,
                1, 3, 3, 3, 1, 3, 3, 20042, 0, 1] #AllBias bfloat16+float32
case43_params = [12, 12, 0, 300, 12, 7, 7, 7, 0, 256, 1, 1, 257, 1,
                1, 1, 1, 1, 1, 1, 1, 20043, 1, 1] #CUTK bfloat16+float32
case44_params = [12, 12, 0, 3, 12, 7, 7, 7, 0, 0, 0, 0, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 20044, 1, 1] #db bfloat16+float32
case45_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 20045, 1, 1] #cuthk2 bfloat16+float32
case46_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0,
                0, 0, 4, 1, 3, 1, 1, 1, 3, 1, 1, 20046, 1, 1] #cuthk4 bfloat16+float32
case47_params = [40, 10, 0, 10, 10, 30, 20, 10, 1, 0, 
                0, 0, 3, 1, 2, 1, 1, 1, 2, 1, 1, 20047, 1, 1] #cuth bfloat16+float32
case48_params = [40, 35, 0, 592, 105, 30, 30, 30, 0, 0, 0, 0, 6, 3,
                1, 3, 3, 3, 1, 3, 3, 20048, 1, 1] #AllBias bfloat16+float32


params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
    "case6": case6_params,
    "case7": case7_params,
    "case8": case8_params,
    "case9": case9_params,
    "case10": case10_params,
    "case11": case11_params,
    "case12": case12_params,
    "case13": case13_params,
    "case14": case14_params,
    "case15": case15_params,
    "case16": case16_params,
    "case17": case17_params,
    "case18": case18_params,
    "case19": case19_params,
    "case20": case20_params,
    "case21": case21_params,
    "case22": case22_params,
    "case23": case23_params,
    "case24": case24_params,
    "case25": case25_params,
    "case26": case26_params,
    "case27": case27_params,
    "case28": case28_params,
    "case29": case29_params,
    "case30": case30_params,
    "case31": case31_params,
    "case32": case32_params,
    "case33": case33_params,
    "case34": case34_params,
    "case35": case35_params,
    "case36": case36_params,
    "case37": case37_params,
    "case38": case38_params,
    "case39": case39_params,
    "case40": case40_params,
    "case41": case41_params,
    "case42": case42_params,
    "case43": case43_params,
    "case44": case44_params,
    "case45": case45_params,
    "case46": case46_params,
    "case47": case47_params,
    "case48": case48_params
}


def main():
    case = sys.argv[1]
    print("moe finalize tiling case: ", case)
    tiling = np.array(params_info.get(case), dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    tiling.tofile(tiling_file)


if __name__ == '__main__':
    main()
