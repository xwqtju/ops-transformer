# FlashAttentionScore

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |

## 功能说明

- 算子功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算：

    -   psetype=1时，需要先add再mul。
    -   psetype≠1时，需要先mul再add。

- 计算公式：

  注意力的正向计算公式如下：

    - psetype=1时，公式如下：
      $$
      attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
      $$

    - psetype≠1时，公式如下：
      $$
      attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
      $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入query。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入key。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入value。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>realShiftOptional</td>
      <td>可选输入</td>
      <td>公式中的pse，表示位置编码。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的Dropout，表示数据丢弃掩码。取值为1代表保留该数据，为0代表丢弃该数据。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的atten_mask，表示注意力掩码，取值为1代表该位不参与计算（不生效），为0代表该位参与计算。</td>
      <td>BOOL、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的scale，表示缩放系数，作为计算流中Muls的scalar值。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepProb</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的keep_prob，表示数据需要保留的概率。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pseType</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>控制add与mul的执行次序，支持配置值为0、1、2、3。</li>
          <li>默认值为1。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>Softmax计算的Max中间结果，用于反向计算。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>Softmax计算的Sum中间结果，用于反向计算。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOutOut</td>
      <td>输出</td>
      <td>公式中的attention_out。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 关于数据shape的约束，以inputLayout的BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
    -   B：取值范围为1\~2M。带prefixOptional的时候B最大支持2K。
    -   N：取值范围为1\~256。
    -   S：取值范围为1\~1M。
    -   D：取值范围为1\~512。
- keepProb的取值范围为(0, 1]。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- pseType为2或3的时候，当前只支持Sq和Skv等长。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_flash_attention_score](./examples/test_aclnn_flash_attention_score.cpp) | 非TND场景，通过[aclnnFlashAttentionScore](./docs/aclnnFlashAttentionScoreV2.md)接口方式调用FlashAttention算子。             |
