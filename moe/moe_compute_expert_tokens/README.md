# aclnnMoeComputeExpertTokens

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

-   **算子功能**：MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。
-   **计算公式**：

    $$
    out_{i}=BinarySearch(sortedExperts,numExperts)
    $$

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sortedExperts</td>
      <td>输入</td>
      <td>公式中的sortedExperts。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numExperts</td>
      <td>属性</td>
      <td>总专家数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>



## 约束说明

-   sortedExperts的shape大小需要小于2\*\*24。
-   numExperts的输入常值需要大于0，但不能超过2048。
-   输入shape大小不要超过device可分配的内存上限，否则会导致异常终止。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_moe_compute_expert_tokens](examples/test_aclnn_moe_compute_expert_tokens.cpp) | 通过接口方式调用[MoeComputeExpertTokens](docs/aclnnMoeComputeExpertTokens.md)算子。 |