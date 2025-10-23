# aclnnMlaProlog

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

## 功能说明
-  **接口功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分为四路，首先对输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后分为两路，第一路乘以$W^{UQ}$和$W^{UK}$经过两次上采样后得到$q^N$；第二路乘以$W^{QR}$后经过旋转位置编码（ROPE）得到$q^R$；第三路是输入$x$乘以$W^{DKV}$进行下采样和RmsNorm后传入Cache中得到$k^C$；第四路是输入$x$乘以$W^{KR}$后经过旋转位置编码后传入另一个Cache中得到$k^R$。
-  **计算公式**：

    RmsNorm公式

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query计算公式，包括下采样，RmsNorm和两次上采样

    $$
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    对Query的进行ROPE旋转位置编码

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    Key计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$


## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnMlaPrologGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnMlaProlog”接口执行计算。
```cpp
aclnnStatus aclnnMlaPrologGetWorkspaceSize(
  const aclTensor *tokenX, 
  const aclTensor *weightDq, 
  const aclTensor *weightUqQr, 
  const aclTensor *weightUk, 
  const aclTensor *weightDkvKr, 
  const aclTensor *rmsnormGammaCq, 
  const aclTensor *rmsnormGammaCkv, 
  const aclTensor *ropeSin, 
  const aclTensor *ropeCos, 
  const aclTensor *cacheIndex, 
  const aclTensor *kvCacheRef, 
  const aclTensor *krCacheRef, 
  const aclTensor *dequantScaleXOptional, 
  const aclTensor *dequantScaleWDqOptional, 
  const aclTensor *dequantScaleWUqQrOptional, 
  const aclTensor *dequantScaleWDkvKrOptional, 
  const aclTensor *quantScaleCkvOptional, 
  const aclTensor *quantScaleCkrOptional, 
  const aclTensor *smoothScalesCqOptional, 
  double           rmsnormEpsilonCq, 
  double           rmsnormEpsilonCkv, 
  char            *cacheModeOptional, 
  const aclTensor *queryOut, 
  const aclTensor *queryRopeOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```
``` cpp
aclnnStatus aclnnMlaProlog(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```



## aclnnMlaPrologGetWorkspaceSize
-  参数说明

    | 参数名 | 输入/输出 | 描述 | 使用说明   | 数据类型 | 数据格式   | 维度(shape) | 非连续Tensor |
    |----------------------------|-----------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|------------|---------------------------|-------------------------|
    | tokenX                     | 输入      | 公式中用于计算Query和Key的输入tensor，Device侧的aclTensor。          | 支持B=0,S=0,T=0的空Tensor                                                                                                              | BFLOAT16、INT8 | ND         | 2维：(T,He)、3维：(B,S,He) | -                       |
    | weightDq                   | 输入      | 公式中用于计算Query的下采样权重矩阵$W^{DQ}$，Device侧的aclTensor      |  不支持空Tensor                                                                                                                         | BFLOAT16、INT8 | FRACTAL_NZ | 2维：(He,Hcq)             | -                       |
    | weightUqQr                 | 输入      | 公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$，Device侧的aclTensor。 |  不支持空Tensor <br> dtype为INT8（量化场景）：<br> 1. 需为per-tensor量化输入 <br>2. 非量化输出时必传dequantScaleWUqQrOptional <br>3. 量化输出时必传dequantScaleWUqQrOptional、quantScaleCkvOptional、quantScaleCkrOptional <br>4. smoothScalesCqOptional可选传 <br> dtype为BFLOAT16（非量化场景）： <br>1. dequantScaleWUqQrOptional、quantScaleCkvOptional、quantScaleCkrOptional、smoothScalesCqOptional必须传空指针 | BFLOAT16、INT8 | FRACTAL_NZ | 2维：(Hcq,N*(D+Dr))       | -                       |
    | weightUk                   | 输入      | 公式中用于计算Key的上采样权重$W^{UK}$，Device侧的aclTensor。           |  不支持空Tensor       | BFLOAT16       | ND         | 3维：(N,D,Hckv)           | -                       |
    | weightDkvKr                | 输入      | 公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$，Device侧的aclTensor。 |  不支持空Tensor                                        | BFLOAT16、INT8 | FRACTAL_NZ | 2维：(He,Hckv+Dr)         | -                      |
    | rmsnormGammaCq             | 输入      | 计算$c^Q$的RmsNorm公式中的$\gamma$参数，Device侧的aclTensor。          |  不支持空Tensor                                                  | BFLOAT16       | ND         | 1维：(Hcq)                | -                       |
    | rmsnormGammaCkv            | 输入      | 计算$c^{KV}$的RmsNorm公式中的$\gamma$参数，Device侧的aclTensor。        |  不支持空Tensor                                                         | BFLOAT16       | ND         | 1维：(Hckv)               | -                       |
    | ropeSin                    | 输入      | 用于计算旋转位置编码的正弦参数矩阵，Device侧的aclTensor。              |  支持B=0,S=0,T=0的空Tensor                                                  | BFLOAT16       | ND         | 2维：(T,Dr)、3维：(B,S,Dr) | -                       |
    | ropeCos                    | 输入      | 用于计算旋转位置编码的余弦参数矩阵，Device侧的aclTensor。              |  支持B=0,S=0,T=0的空Tensor                                                | BFLOAT16       | ND         | 2维：(T,Dr)、3维：(B,S,Dr) | -                       |
    | cacheIndex                 | 输入      | 用于存储kvCache和krCache的索引，Device侧的aclTensor。                  |  支持B=0,S=0,T=0的空Tensor <br> 取值范围需在[0,BlockNum*BlockSize)内                                                                   | INT64          | ND         | 1维：(T)、2维：(B,S)      | -                       |
    | kvCacheRef                 | 输入      | 用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$）。       |  支持B=0,Skv=0的空Tensor；Nkv与N关联，N是超参，故Nkv不支持dim=0                                                                         | BFLOAT16、INT8 | ND         | 4维：(BlockNum,BlockSize,Nkv,Hckv) | -                       |
    | krCacheRef                 | 输入      | 用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$），Device侧的aclTensor。 | 支持B=0,Skv=0的空Tensor；Nkv与N关联，N是超参，故Nkv不支持dim=0                                                                          | BFLOAT16、INT8 | ND         | 4维：(BlockNum,BlockSize,Nkv,Dr) | -                       |
    | dequantScaleXOptional      | 输入      | tokenX的反量化参数。   |  必须传入空指针      | FLOAT          | ND         | -                         | -                       |
    | dequantScaleWDqOptional    | 输入      | weightDq的反量化参数。 |  必须传入空指针   | FLOAT          | ND         | -                         | -                       |
    | dequantScaleWUqQrOptional  | 输入      | 用于MatmulQcQr矩阵乘后反量化操作的per-channel参数，Device侧的aclTensor。 |  支持非空Tensor（仅INT8 dtype场景需传）   | FLOAT    ND         | 2维：(1,N*(D+Dr))         | -                       |
    | dequantScaleWDkvKrOptional | 输入      | weightDkvKr的反量化参数。|  必须传入空指针 | FLOAT          | ND         | -                         | -                       |
    | quantScaleCkvOptional      | 输入      | 用于对KVCache输出数据做量化操作的参数，Device侧的aclTensor。            |  支持非空Tensor（仅INT8 dtype量化输出场景需传）                                                                                         | FLOAT          | ND         | 2维：(1,Hckv)             | -                       |
    | quantScaleCkrOptional      | 输入      | 用于对KRCache输出数据做量化操作的参数，Device侧的aclTensor。            |  支持非空Tensor（仅INT8 dtype量化输出场景需传）                                                                                         | FLOAT          | ND         | 2维：(1,Dr)               | -                       |
    | smoothScalesCqOptional     | 输入      | 用于对RmsNormCq输出做动态量化操作的参数，Device侧的aclTensor。         |  支持非空Tensor（仅INT8 dtype场景可选传）                                                                                               | FLOAT          | ND         | 2维：(1,Hcq)              | -                       |
    | rmsnormEpsilonCq           | 输入      | 计算$c^Q$的RmsNorm公式中的$\epsilon$参数，Host侧参数。                  |  用户未特意指定时，建议传入1e-05 <br> 仅支持double类型                                                                                 | double         | -          | -                         | -                       |
    | rmsnormEpsilonCkv          | 输入      | 计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数，Host侧参数。                |  用户未特意指定时，建议传入1e-05 <br> 仅支持double类型                                                                                 | double         | -          | -                         | -                       |
    | cacheModeOptional          | 输入      | 表示kvCache的模式，Host侧参数。                                        |  用户未特意指定时，建议传入"PA_BSND" <br> 仅支持char*类型 <br> 可选值为"PA_BSND"、"PA_NZ"                                             | char*          | -          | -                         | -                       |
    | queryOut                   | 输出      | 公式中Query的输出tensor（对应$q^N$），Device侧的aclTensor。             | 不支持空Tensor                                                                                                                         | BFLOAT16、INT8 | ND         | 3维：(T,N,Hckv)、4维：(B,S,N,Hckv) | -                       |
    | queryRopeOut               | 输出      | 公式中Query位置编码的输出tensor（对应$q^R$），Device侧的aclTensor。      | 不支持空Tensor                                                                                                                         | BFLOAT16       | ND         | 3维：(T,N,Dr)、4维：(B,S,N,Dr) | -                       |
    | workspaceSize              | 输出      | 返回需在Device侧申请的workspace大小。                                  | 仅用于输出结果，无需输入配置 <br> 数据类型为uint64_t*                                                                                 | -              | -          | -                         | -                       |
    | executor                   | 输出      | 返回op执行器，包含算子计算流程。                                      |  仅用于输出结果，无需输入配置 <br> 数据类型为aclOpExecutor**                                                                           | -              | -          | -                         | -                       |

-  返回值

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。</br>
    第一段接口完成入参校验，出现以下场景时报错：

    | 返回值                 | 错误码               | 描述      |
    |------------------------|----------------------|----------------------|
    | ACLNN_ERR_PARAM_NULLPTR | 161001   | 必须传入的参数（如接口核心依赖的输入/输出参数）中存在空指针。         |
    | ACLNN_ERR_PARAM_INVALID | 161002   | 输入参数的 shape（维度/尺寸）、dtype（数据类型）不在接口支持的范围内。 |
    | ACLNN_ERR_RUNTIME_ERROR | 361001   | API 内存调用 NPU Runtime 接口时发生异常（如 Runtime 服务未启动、内存申请失败等）。 |
    | ACLNN_ERR_INNER_TILING_ERROR | 561002  | tiling发生异常，入参的dtype类型或者shape错误。 |
## aclnnMlaProlog
- 参数说明

  | 参数名        | 参数类型         | 含义                                                                 |
  |---------------|------------------|----------------------------------------------------------------------|
  | workspace     | void\*           | 在Device侧申请的workspace内存地址。                                  |
  | workspaceSize | uint64_t         | 在Device侧申请的workspace大小，由第一段接口aclnnMlaPrologGetWorkspaceSize获取。 |
  | executor      | aclOpExecutor\*  | op执行器，包含了算子计算流程。                                       |
  | stream        | aclrtStream      | 指定执行任务的AscendCL Stream流。|

-  返回值
    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。



## 约束说明
-  shape 格式字段含义说明
    | 字段名       | 英文全称/含义                  | 取值规则与说明                                                                 |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B            | Batch（输入样本批量大小）      | 取值范围：0~65536                                                           |
    | S            | Seq-Length（输入样本序列长度） | 取值范围：0~16                                                              |
    | He           | Head-Size（隐藏层大小）        | 取值固定为：7168、7680                                                     |
    | Hcq          | q 低秩矩阵维度                 | 取值固定为：1536                                                           |
    | N            | Head-Num（多头数）             | 取值范围：1、2、4、8、16、32、64、128                                       |
    | Hckv         | kv 低秩矩阵维度                | 取值固定为：512                                                             |
    | D            | qk 不含位置编码维度            | 取值固定为：128                                                             |
    | Dr           | qk 位置编码维度                | 取值固定为：64                                                              |
    | Nkv          | kv 的 head 数                  | 取值固定为：1                                                               |
    | BlockNum     | PagedAttention 场景下的块数    | 取值为计算 `B*Skv/BlockSize` 的结果后向上取整（Skv 表示 kv 的序列长度，允许取 0） |
    | BlockSize    | PagedAttention 场景下的块大小  | 取值范围：16、128                                                           |
    | T            | BS 合轴后的大小                | 取值范围：0~1048576；注：若采用 BS 合轴，此时 tokenX、ropeSin、ropeCos 均为 2 维，cacheIndex 为 1 维，queryOut、queryRopeOut 为 3 维 |

- aclnnMlaProlog接口支持场景：
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">场景</th>
      <th>含义</th>
    </tr>
    <tr>
      <td colspan="2">非量化</td>
      <td>
          入参：所有入参皆为非量化数据 <br> 
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="2">部分量化</td>
      <td>kv_cache非量化 </td>
      <td>
          入参：weightUqQr传入pertoken量化数据，其余入参皆为非量化数据 <br> 
          出参：所有出参返回非量化数据 
      </td>
    </tr>
    <tr>
      <td>kv_cache量化 </td>
      <td> 
          入参：weightUqQr传入pertoken量化数据，kvCacheRef、krCacheRef传入perchannel量化数据，其余入参皆为非量化数据 <br> 
          出参：kvCacheRef、krCacheRef返回perchannel量化数据，其余出参返回非量化数据 
      </td>
    </tr>
  </table>

- 在不同量化场景下，参数的dtype和shape组合需要满足如下条件：
  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">参数名</th>
      <th rowspan="2" colspan="2">非量化场景</th>
      <th colspan="4">部分量化场景</th>
    </tr>
    <tr>
      <th colspan="2">kv_cache非量化</th>
      <th colspan="2">kv_cache量化</th>
    </tr>
    <tr>
      <th>dtype</th>
      <th>shape</th>
      <th>dtype</th>
      <th>shape</th>
      <th>dtype</th>
      <th>shape</th>
    </tr>
    <tr>
      <td>tokenX</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
    </tr>
    <tr>
      <td>weightDq</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
    </tr>
    <tr>
      <td>weightUqQr</td>
      <td>BFLOAT16</td>
      <td> (Hcq, N*(D+Dr))</td>
      <td>INT8</td>
      <td> (Hcq, N*(D+Dr))</td>
      <td>INT8</td>
      <td> (Hcq, N*(D+Dr))</td>
    </tr>
    <tr>
      <td>weightUk</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
    </tr>
    <tr>
      <td>weightDkvKr</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
    </tr>
    <tr>
      <td> rmsnormGammaCq </td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
    </tr>
    <tr>
      <td> rmsnormGammaCkv </td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
    </tr>
    <tr>
      <td> ropeSin </td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
    </tr>
    <tr>
      <td> ropeCos </td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
    </tr>
    <tr>
      <td> cacheIndex </td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
    </tr>
    <tr>
      <td> kvCacheRef </td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      <td>INT8</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
    </tr>
    <tr>
      <td> krCacheRef </td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      <td>INT8</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
    </tr>
    <tr>
      <td> dequantScaleXOptional </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequantScaleWDqOptional </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequantScaleWUqQrOptional </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
    </tr>
    <tr>
      <td> dequantScaleWDkvKrOptional </td>
      <td> 无需赋值 </td>
      <td> / </td>
      <td> 无需赋值 </td>
      <td> / </td>
      <td> 无需赋值 </td>
      <td> / </td>
    </tr>
    <tr>
      <td> quantScaleCkvOptional </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Hckv) </td>
    </tr>
    <tr>
      <td> quantScaleCkrOptional </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Dr) </td>
    </tr>
    <tr>
      <td> smoothScalesCqOptional </td>
      <td>无需赋值</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Hcq) </td>
      <td>FLOAT</td>
      <td> (1, Hcq) </td>
    </tr>
    <tr>
      <td> queryOut </td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
    </tr>
    <tr>
      <td> queryRopeOut </td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
    </tr>
    <tr>
      <td> dequantScaleQNopeOutOptional </td>
      <td>无需赋值</td>
      <td>/</td>
      <td>无需赋值</td>
      <td>/</td>
      <td>无需赋值</td>
      <td>/</td>
    </tr>
  </table>
  </div>


  <!-- 参数解释请参见**算子执行接口**。 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mla_prolog.h"
  
  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)
  
  #define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)
  
  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shape_size = 1;
      for (auto i : shape) {
          shape_size *= i;
      }
      return shape_size;
  }
  
  int Init(int32_t deviceId, aclrtStream* stream) {
      // 固定写法，AscendCL初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  template <typename T>
  int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMalloc申请host侧内存
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  template <typename T>
  int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMalloc申请host侧内存
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  int TransToNZShape(std::vector<int64_t> &shapeND) {
      int64_t inputParam1 = shapeND[0];
      int64_t inputParam2 = shapeND[1];
      int64_t h0 = 16;
      int64_t newParam1 = inputParam2 / h0;
      int64_t newParam2 = inputParam1 / h0;
      shapeND[0] = newParam1;
      shapeND[1] = newParam2;
      shapeND.emplace_back(h0);
      shapeND.emplace_back(h0);
      return 0;
  }
  
  int main() {
      // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
      // 根据实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. 构造输入与输出，需要根据API的接口定义构造
      std::vector<int64_t> tokenXShape = {8, 1, 7168};  // B,S,He
      std::vector<int64_t> weightDqShape = {7168, 1536};  // He,Hcq
      std::vector<int64_t> weightUqQrShape = {1536, 6144};  // Hcq,N*(D+Dr)
      std::vector<int64_t> weightUkShape = {32, 128, 512};  // N,D,Hckv
      std::vector<int64_t> weightDkvKrShape = {7168, 576};  // He,Hckv+Dr
      std::vector<int64_t> rmsnormGammaCqShape = {1536};  // Hcq
      std::vector<int64_t> rmsnormGammaCkvShape = {512};  // Hckv
      std::vector<int64_t> ropeSinShape = {8, 1, 64};  // B,S,Dr
      std::vector<int64_t> ropeCosShape = {8, 1, 64};  // B,S,Dr
      std::vector<int64_t> cacheIndexShape = {8, 1};  // B,S
      std::vector<int64_t> kvCacheShape = {16, 128, 1, 512};  // BolckNum,BlockSize,Nkv,Hckv
      std::vector<int64_t> krCacheShape = {16, 128, 1, 64};  // BolckNum,BlockSize,Nkv,Dr
      std::vector<int64_t> queryShape = {8, 1, 32, 512};  // B,S,N,Hckv
      std::vector<int64_t> queryRopeShape = {8, 1, 32, 64};  // B,S,N,Dr
      double rmsnormEpsilonCq = 1e-5;
      double rmsnormEpsilonCkv = 1e-5;
      char cacheMode[] = "PA_BSND";
  
      void* tokenXDeviceAddr = nullptr;
      void* weightDqDeviceAddr = nullptr;
      void* weightUqQrDeviceAddr = nullptr;
      void* weightUkDeviceAddr = nullptr;
      void* weightDkvKrDeviceAddr = nullptr;
      void* rmsnormGammaCqDeviceAddr = nullptr;
      void* rmsnormGammaCkvDeviceAddr = nullptr;
      void* ropeSinDeviceAddr = nullptr;
      void* ropeCosDeviceAddr = nullptr;
      void* cacheIndexDeviceAddr = nullptr;
      void* kvCacheDeviceAddr = nullptr;
      void* krCacheDeviceAddr = nullptr;
      void* queryDeviceAddr = nullptr;
      void* queryRopeDeviceAddr = nullptr;
  
      void* tokenXHostAddr = nullptr;
      void* weightDqHostAddr = nullptr;
      void* weightUqQrHostAddr = nullptr;
      void* weightUkHostAddr = nullptr;
      void* weightDkvKrHostAddr = nullptr;
      void* rmsnormGammaCqHostAddr = nullptr;
      void* rmsnormGammaCkvHostAddr = nullptr;
      void* ropeSinHostAddr = nullptr;
      void* ropeCosHostAddr = nullptr;
      void* cacheIndexHostAddr = nullptr;
      void* kvCacheHostAddr = nullptr;
      void* krCacheHostAddr = nullptr;
      void* queryHostAddr = nullptr;
      void* queryRopeHostAddr = nullptr;
  
      aclTensor* tokenX = nullptr;
      aclTensor* weightDq = nullptr;
      aclTensor* weightUqQr = nullptr;
      aclTensor* weightUk = nullptr;
      aclTensor* weightDkvKr = nullptr;
      aclTensor* rmsnormGammaCq = nullptr;
      aclTensor* rmsnormGammaCkv = nullptr;
      aclTensor* ropeSin = nullptr;
      aclTensor* ropeCos = nullptr;
      aclTensor* cacheIndex = nullptr;
      aclTensor* kvCache = nullptr;
      aclTensor* krCache = nullptr;
      aclTensor* query = nullptr;
      aclTensor* queryRope = nullptr;
  
      // 转换三个NZ格式变量的shape
      ret = TransToNZShape(weightDqShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
  
      // 创建tokenX aclTensor
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_BF16, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightDq aclTensor
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_BF16, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightUqQr aclTensor
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_BF16, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightUk aclTensor
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightDkvKr aclTensor
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_BF16, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建ropeSin aclTensor
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建ropeCos aclTensor
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rmsnormGammaCq aclTensor
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rmsnormGammaCkv aclTensor
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建cacheIndex aclTensor
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建kvCache aclTensor
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_BF16, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建krCache aclTensor
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建query aclTensor
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_BF16, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建queryRope aclTensor
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  
      // 3. 调用CANN算子库API，需要修改为具体的API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // 调用aclnnMlaProlog第一段接口
      ret = aclnnMlaPrologGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, cacheIndex, kvCache, krCache, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode, query, queryRope, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      // 调用aclnnMlaProlog第二段接口
      ret = aclnnMlaProlog(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaProlog failed. ERROR: %d\n", ret); return ret);
  
      // 4. 固定写法，同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  
      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(tokenX);
      aclDestroyTensor(weightDq);
      aclDestroyTensor(weightUqQr);
      aclDestroyTensor(weightUk);
      aclDestroyTensor(weightDkvKr);
      aclDestroyTensor(rmsnormGammaCq);
      aclDestroyTensor(rmsnormGammaCkv);
      aclDestroyTensor(ropeSin);
      aclDestroyTensor(ropeCos);
      aclDestroyTensor(cacheIndex);
      aclDestroyTensor(kvCache);
      aclDestroyTensor(krCache);
      aclDestroyTensor(query);
      aclDestroyTensor(queryRope);
  
      // 7. 释放device 资源
      aclrtFree(tokenXDeviceAddr);
      aclrtFree(weightDqDeviceAddr);
      aclrtFree(weightUqQrDeviceAddr);
      aclrtFree(weightUkDeviceAddr);
      aclrtFree(weightDkvKrDeviceAddr);
      aclrtFree(rmsnormGammaCqDeviceAddr);
      aclrtFree(rmsnormGammaCkvDeviceAddr);
      aclrtFree(ropeSinDeviceAddr);
      aclrtFree(ropeCosDeviceAddr);
      aclrtFree(cacheIndexDeviceAddr);
      aclrtFree(kvCacheDeviceAddr);
      aclrtFree(krCacheDeviceAddr);
      aclrtFree(queryDeviceAddr);
      aclrtFree(queryRopeDeviceAddr);
  
      // 8. 释放host 资源
      aclrtFree(tokenXHostAddr);
      aclrtFree(weightDqHostAddr);
      aclrtFree(weightUqQrHostAddr);
      aclrtFree(weightUkHostAddr);
      aclrtFree(weightDkvKrHostAddr);
      aclrtFree(rmsnormGammaCqHostAddr);
      aclrtFree(rmsnormGammaCkvHostAddr);
      aclrtFree(ropeSinHostAddr);
      aclrtFree(ropeCosHostAddr);
      aclrtFree(cacheIndexHostAddr);
      aclrtFree(kvCacheHostAddr);
      aclrtFree(krCacheHostAddr);
      aclrtFree(queryHostAddr);
      aclrtFree(queryRopeHostAddr);
  
      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  
      return 0;
  }
  ```



