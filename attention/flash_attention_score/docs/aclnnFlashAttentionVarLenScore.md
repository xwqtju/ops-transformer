# aclnnFlashAttentionVarLenScore

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |


## 功能说明

-   算子功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。**跟[FlashAttentionScore](./aclnnFlashAttentionScore.md)接口的区别是该接口支持可变长S的计算，即可以一次传入多个长度不相等的sequence。使用此接口时，query，key和value使用TND的格式传入数据，其中T意为total number，表示其所有sequence的length总和，同时使用actualSeqQLenOptional与actualSeqKvLenOptional输入传入每个sequence依次的累积长度以区分不同sequence。每个sequence单独计算其注意力结果**。
- 计算公式：

    注意力的正向计算公式如下：

    $$
    attention\_out=Dropout(Softmax(Mask(scale*(pse+query*key^T),atten\_mask)),keep\_prob)*value
    $$



## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnFlashAttentionVarLenScoreGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionVarLenScore”接口执行计算。

```c++
aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *realShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *prefixOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualSeqKvLenOptional,
  double             scaleValue,
  double             keepProb,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            innerPrecise,
  int64_t            sparseMode,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *softmaxOutOut,
  const aclTensor   *attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```
```c++
aclnnStatus aclnnFlashAttentionVarLenScore(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```


### aclnnFlashAttentionVarLenScoreGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>输入</td>
        <td>Device侧的aclTensor，公式中的query。</td>
        <td>数据类型与key/value的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>Device侧的aclTensor，公式中的key。</td>
        <td>数据类型与query/value的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>Device侧的aclTensor，公式中的value。</td>
        <td>数据类型与query/key的数据类型一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>可选输入</td>
        <td>Device侧的aclTensor，公式中的pse。</td>
        <td>
          <ul>
              <li>数据类型与query的数据类型一致。</li>
              <li>输入shape类型需为[B,N,S,S]、[B,N,1,Skv]、[1,N,S,S]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>输入</td>
        <td>Device侧的aclTensor，公式中的Dropout。</td>
        <td>-</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0、1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>Device侧的aclTensor，公式中的atten_mask。</td>
        <td>
          <ul>
              <li>取值为1代表该位不参与计算，为0代表该位参与计算。</li>
              <li>输入shape类型需为[B,N,S,S]、[B,1,S,S]、[1,1,S,S]、[S,S]。</li>
          </ul>
        </td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
        <td>0、2、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>输入</td>
        <td>Host侧的aclIntArray，代表prefix稀疏计算场景每个Batch的N值。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>输入</td>
        <td>Host侧的aclIntArray，描述了每个Batch对应的query的sequence length。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>输入</td>
        <td>Host侧的aclIntArray，描述了每个Batch对应的key/value的sequence length。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>Host侧的double，公式中的scale，代表缩放系数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>输入</td>
        <td>Host侧的double，代表dropMaskOptional中1的比例。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>Host侧的int64_t，用于稀疏计算 ，表示slides window的左边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>Host侧的int64_t，用于稀疏计算，表示slides window的右边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>Host侧的int64_t，代表单卡的head个数，即输入query的N轴长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>Host侧的string，代表输入query、key、value的数据排布格式。</td>
        <td>支持TND。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>Host侧的int64_t，用于提升精度。</td>
        <td>暂未使用。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>Host侧的int64_t，表示sparse的模式。</td>
        <td>支持配置值为0、1、2、3、4、5、6。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>输出</td>
        <td>Device侧的aclTensor，Softmax计算的Max中间结果，用于反向计算。</td>
        <td>输出的shape类型为[N,T,8]。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>输出</td>
        <td>Device侧的aclTensor，Softmax计算的Sum中间结果，用于反向计算。</td>
        <td>输出的shape类型为[N,T,8]。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>输出</td>
        <td>Device侧的aclTensor，计算公式的最终输出。</td>
        <td>数据类型和shape类型与query保持一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0、3、4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、key、value、realShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOut、softmaxSumOut、softmaxOutOut、attentionOutOut的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>query、key、value、realShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOut、softmaxSumOut、softmaxOutOut、attentionOutOut的数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>


## aclnnFlashAttentionVarLenScore

-   **参数说明：**
  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 144px">
  <col style="width: 125px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionVarLenScoreGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的AscendCL stream流。</td>
    </tr>
  </tbody>
  </table>


-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 输入query、key、value的B：batchsize必须相等。
- 输入query、key、value的D：Head-Dim必须满足(qD == kD && kD >= vD)。
- 输入query、key、value的input_layout必须一致。
- 输入query、key、value、realShiftOptional的数据类型必须一致。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。当Nq/Nkv > 1时，即为GQA(grouped-query attention)；当Nkv=1时，即为MQA(multi-query attention)。本文如无特殊说明，N表示的是Nq。
- 关于数据shape的约束，其中：
    - T：取值范围为1\~1M。
    - N：取值范围为1\~256。
    - D：取值范围为1\~768。
- query、key、value数据排布格式仅支持TND，T是B和S合轴紧密排列的数据（每个batch的SeqLenQ和SeqLenKV），其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- sparseMode: 当所有的attenMaskOptional的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量；sparseMode配置为1、2、3、5时，用户配置的preTokens、nextTokens不会生效；sparseMode配置为0、4时，须保证attenMaskOptional与preTokens、nextTokens的范围一致。用户不特意指定时建议传入0。sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/context/sparse_mode参数说明.md)。
- 不同数据格式详情请参见[数据格式](../../../docs/context/数据格式.md)。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- keepProb的取值范围为(0, 1]。
- prefixOptional稀疏计算场景即sparseMode=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。
- band场景，preTokens和nextTokens之间必须要有交集。
- sparseMode配置为1、2、3、6时，用户配置的preTokens、nextTokens不会生效；sparseMode配置为0、4、7、8时，须保证attenMaskOptional与preTokens、nextTokens的范围一致。
- sparseMode为1、2、3、4、6、7、8时，应传入对应正确的attenMaskOptional，否则将导致计算结果错误。当attenMaskOptional输入为None时，sparseMode、preTokens、nextTokens参数不生效，固定为全计算。
- attenMaskOptional输入不支持补pad，即attenMaskOptional中不能存在某一行全1的场景。
- sparse_mode=3时，不支持无效行计算，需要满足每个batch的Sq<=Skv。
- actualSeqQLenOptional输入支持某个Batch上的S长度为0，此时不支持可选输入realShiftOptional，假设真实的S长度为\[2,2,0,2,2\]，则传入的actualSeqQLenOptional为\[2,4,4,6,8\]。actualSeqQLenOptional的长度取值范围为1\~2K。当存在prefixOptional输入的时候，其长度最大支持1K。
- sparse_mode=7时，不支持可选输入realShiftOptional。
- sparse_mode=8时，当每个sequence的q、kv等长时支持可选输入realShiftOptional，针对全局做pse生成。支持q方向进行外切，需要外切前每个sequence的q、kv等长，外切后传入的actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0（本功能属实验性功能）。



## 调用示例

该融合算子有两种调用方式：

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_fusion_attention](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_flash_attention_score.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
  }

  int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> qShape = {256, 1, 128};
    std::vector<int64_t> kShape = {256, 1, 128};
    std::vector<int64_t> vShape = {256, 1, 128};
    std::vector<int64_t> attenmaskShape = {256, 256};

    std::vector<int64_t> attentionOutShape = {256, 1, 128};
    std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
    std::vector<int64_t> softmaxSumShape = {256, 1, 8};

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* attenmaskDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* pse = nullptr;
    aclTensor* dropMask = nullptr;
    aclTensor* padding = nullptr;
    aclTensor* attenmask = nullptr;
    aclTensor* attentionOut = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* softmaxOut = nullptr;

    std::vector<float> qHostData(32768, 1);
    std::vector<float> kHostData(32768, 1);
    std::vector<float> vHostData(32768, 1);
    std::vector<uint8_t> attenmaskHostData(65536, 0);
    std::vector<float> attentionOutHostData(32768, 0);
    std::vector<float> softmaxMaxHostData(2048, 3.0);
    std::vector<float> softmaxSumHostData(2048, 3.0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    std::vector<int64_t> prefixOp = {0};
    aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
    std::vector<int64_t>  acSeqQLenOp = {256};
    std::vector<int64_t>  acSeqKvLenOp = {256};
    aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
    aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
    double scaleValue = 0.088388;
    double keepProb = 1;
    int64_t preTokens = 65536;
    int64_t nextTokens = 65536;
    int64_t headNum = 1;
    int64_t innerPrecise = 0;
    int64_t sparseMod = 0;
    
    char layOut[5] = {'T', 'N', 'D', 0};
    
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    
    // 调用aclnnFlashAttentionVarLenScore第一段接口
    ret = aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
              q, k, v, pse, dropMask, padding, attenmask, prefix, acSeqQLen, acSeqKvLen,
              scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
              sparseMod, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    
    // 调用aclnnFlashAttentionVarLenScore第二段接口
    ret = aclnnFlashAttentionVarLenScore(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScore failed. ERROR: %d\n", ret); return ret);
    
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
    PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
    PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
    
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(attenmask);
    aclDestroyTensor(attentionOut);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    
    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(attenmaskDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    return 0;
  }
  ```
