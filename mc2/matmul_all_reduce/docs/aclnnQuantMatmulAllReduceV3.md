# aclnnQuantMatmulAllReduceV3
## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | x |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √ |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

- **算子功能**：aclnnQuantMatmulAllReduceV3接口是对aclnnQuantMatmulAllReduceV2接口的功能扩展, 新增支持低比特通信：matmul的计算结果依次进行all_to_all通信、reduceSum计算、allgather通信、dequant反量化，代替先dequant和pertoken计算、再all_reduce通信的原流程。支持pertensor、perchannel、pertoken[量化方式](common/量化介绍.md)。

- **计算公式**：

    为以下3种情形：

    - 情形1：对量化后的入参x1、x2进行matmul计算后，接着进行dequant计算，接着与x3进行add操作，最后做all_reduce计算。

  $$
  output= allReduce(dequantScale*(x1_{int8}@x2_{int8} + bias_{int32}) + x3)
  $$

    - 情形2：对量化后的入参x1、x2进行mm计算后，接着进行dequant和pertoken计算，接着与x3进行add操作，最后做all_reduce计算。

  $$
  output= allReduce(dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional)
  $$

    - 情形3：对量化后的入参x1、x2进行matmul、dequant和pertoken计算，接着与x3进行add操作，再对输出进行perchannel量化，然后进行all_to_all通信，对第一次通讯结果进行reduceSum计算，接着进行all_gather通信，最后对第二次通信结果进行dequant，得到最终输出。

  $$
  matmulAddOutPut = (dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
  $$

  $$
  alltoallOutPut_{int8} = alltoall(matmulAddOutPut / commQuantScale1Optional);
  $$

  $$
  reduceSumOutPut_{int8} = (add(alltoallOutPut_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
  $$

  $$
  outPut = (allgather(reduceSumOutPut_{int8}) * commQuantScale2Optional);
  $$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnQuantMatmulAllReduceV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulAllReduceV3”接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x3Optional,
    const aclTensor *dequantScale,
    const aclTensor *pertokenScaleOptional,
    const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```
```cpp
aclnnStatus aclnnQuantMatmulAllReduceV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantMatmulAllReduceV3GetWorkspaceSize

-   **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
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
          <td>x1</td>
          <td>输入</td>
          <td>Device侧的aclTensor，MatMul计算的左矩阵，即计算公式中的x1。</td>
          <td><li>当前版本仅支持二维或者三维输入。</li><li>支持不转置场景。</li></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>输入</td>
          <td>Device侧的aclTensor，MatMul计算的右矩阵，即计算公式中的x2。</td>
          <td><li>当前版本仅支持两维输入。</li><li>支持转置/不转置场景。</li></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>输入</td>
          <td>Device侧的aclTensor，bias偏移，即计算公式中的biasOptional。</td>
          <td>当前版本仅支持一维输入。</td>
          <td>INT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>输入</td>
          <td>Device侧的aclTensor，MatMul计算后的add计算，即计算公式中的x3Optional。</td>
          <td><li>维度与output一致。</li><li>目前仅支持输出为BFLOAT16场景，且仅支持非空输入。</li></td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dequantScale</td>
          <td>输入</td>
          <td>Device侧的aclTensor，MatMul计算后的去量化系数，即计算公式中的dequantScale。</td>
          <td><li>shape在pertensor场景为(1)，perchannel场景为(n)/(1, n)</li><li>输出为BFLOAT16时，直接将BFLOAT16类型的dequantScale传入本接口。</li><li>输出为FLOAT16时，如果pertokenScale不为空，可直接将FLOAT32类型的dequantScale传入本接口，如果pertokenScale为空，则需提前调用TransQuantParamV2算子的aclnn接口来将dequantScale转成INT64/UINT64数据类型。</li></td>
          <td>INT64、UINT64、FLOAT32、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>pertokenScaleOptional</td>
          <td>输入</td>
          <td>Device侧的aclTensor，MatMul计算后的pertoken去量化系数，即计算公式中的pertokenScaleOptional。</td>
          <td>x1为(b, s, k)时，shape为(b*s)；x1为(m, k)时shape为(m)。</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale1Optional</td>
          <td>输入</td>
          <td>Device侧的aclTensor，matmulAdd计算后的perchannel量化系数，即计算公式中的commQuantScale1。</td>
          <td>x2为(k, n)时, shape可为(n)或者(1,n)</td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale2Optional</td>
          <td>输入</td>
          <td>Device侧的aclTensor，allGather计算后的perchannel量化系数，即计算公式中的commQuantScale2。</td>
          <td>x2为(k, n)时, shape可为(n)或者(1,n)</td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>输入</td>
          <td>Host侧标识通信域的字符串，通信域名称。</td>
          <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>输入</td>
          <td>Host侧标识操作类型的字符串，reduce操作类型。</td>
          <td>当前仅支持输入"sum"。</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>输入</td>
          <td>Host侧的整型，通信数据切分数，即总数据量/单次通信量。</td>
          <td>当前版本仅支持输入0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>输入</td>
          <td>Host侧的整型，AscendCL流模式的枚举。</td>
          <td>当前只支持枚举值1。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>输出</td>
          <td>Device侧的aclTensor，MatMul计算与AllReduce通信的结果，即计算公式中的output。</td>
          <td>output的维数与x1一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
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

    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：输入x2的数据格式支持ND和FRACTAL_NZ格式。
    输入的shape规则如下：
        - 当x2的数据格式为FRACTAL_NZ时，当前版本仅支持四维输入，配合aclnnCalculateMatmulWeightSizeV2和aclnnTransMatmulWeight完成输入ND到NZ的转换，非连续的tensor仅支持transpose场景。
        - 当x2的数据格式为ND时，当前版本仅支持二维输入。

-   **返回值：**

    返回aclnnStatus状态码，具体参见aclnn返回码。
    第一段接口完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x1、x2、dequantScale或output是空指针。</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>x1、x2、bias、dequantScale、pertokenScaleOptional、x3、commQuantScale1Optional、commQuantScale2Optional或output的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
        <td>streamMode不在合法范围内。</td>
    </tr>
    <tr>
        <td>x1、x2、bias、dequantScale、pertokenScaleOptional、x3、commQuantScale1Optional、commQuantScale2Optional或output的shape不符合约束要求。</td>
    </tr>
    </tbody>
    </table>
## aclnnQuantMatmulAllReduceV3

-   **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1312px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 750px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulAllReduceV3GetWorkspaceSize获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
    </tr>
    </tbody></table>
-   **返回值：**

    返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

- 增量场景不使能MC2，全量场景使能MC2。
- 输入x1可为2维或者3维，且不为空Tensor，其shape为(b, s, k)或者(m, k)。x2必须是2维，且不为空Tensor。其shape为(k, n)，k轴满足mm算子入参要求，k轴相等。
- m大小不超过2147483647，x1与x2的最后一维大小不超过65535，x1的最后一维指k，x2的最后一维指转置时的k或非转置时的n。
- 传入的x1、x2、dequantScale或者output不为空指针。
- x1和x2、dequantScale、output、bias（非空场景）、x3（非空场景）的数据类型和数据格式需要在支持的范围之内。
- 若输出output类型为FLOAT16，当pertokenScale为空时，dequantScale的类型为INT64、UINT64，当pertokenScale不为空时，dequantScale的类型为FLOAT32；若输出output类型为BFLOAT16，dequantScale的类型为BFLOAT16，x3的类型为BFLOAT16。
- 传入的commQuantScale1与commQuantScale2需要同时为空指针或同时不为空指针，若传入的commQuantScale1与commQuantScale2同时不为空指针，两个量化参数shape需保持一致，类型需与算子输出类型保持一致，且每张卡输入保持一致。
- x1的shape为(b, s, k)时，pertokenScaleOptional的shape为(b*s)；当x1的shape为(m, k)时，pertokenScaleOptional的shape为(m)。
- 仅支持hccs链路all mesh组网。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持1、2、4、8卡。
- 不支持空tensor。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：一个模型中的通算融合MC2算子，仅支持相同通信域。
- int8低bit通信仅在通信bound的情况下存在性能收益，计算bound的情况不建议使能int8低bit通信，即不建议输入commQuantScale1和commQuantScale2。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "aclnnop/aclnn_trans_matmul_weight.h"
    #include "../op_host/op_api/aclnn_quant_matmul_all_reduce_v3.h"

    int ndev = 8;

    #define ACL_CHECK(ret)                                                                                     \
        do {                                                                                                   \
            auto retcode = ret;                                                                                \
            if (retcode != ACL_SUCCESS) {                                                                      \
                printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
                return retcode;                                                                                \
            }                                                                                                  \
        } while (0)

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
        std::string format;
      };

    template<typename T>
    int CreateWeightNzAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                                aclDataType dataType, aclTensor **tensor, Args &args) {
        auto size = GetShapeSize(shape) * sizeof(T);
        const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
        auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, ACL_INT8, &size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return    ret);
        auto tensorSize = size * sizeof(T);

        // 调用aclrtMalloc申请device内存
        ret = aclrtMalloc(deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
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

        uint64_t transWorkspaceSize;
        aclOpExecutor *executor;
        void *transWorkspaceAddr = nullptr;
        ret = aclnnTransMatmulWeightGetWorkspaceSize(*tensor, &transWorkspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS && transWorkspaceSize > 0,
                  printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));

        return 0;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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

    int launchOneThreadQuantMatmulAllReduce(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                  args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> dequantScaleShape = {128};
        std::vector<int64_t> pertokenScaleShape = {32};
        std::vector<int64_t> commQuantScale1Shape = {128};
        std::vector<int64_t> commQuantScale2Shape = {128};
        std::vector<int64_t> x3Shape = {32, 128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *dequantScaleDeviceAddr = nullptr;
        void *pertokenScaleDeviceAddr = nullptr;
        void *commQuantScale1DeviceAddr = nullptr;
        void *commQuantScale2DeviceAddr = nullptr;
        void *x3DeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *dequantScale = nullptr;
        aclTensor *pertokenScale = nullptr;
        aclTensor *commQuantScale1 = nullptr;
        aclTensor *commQuantScale2 = nullptr;
        aclTensor *x3 = nullptr;
        aclTensor *out = nullptr;

        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
        long long pertokenScaleShapeSize = GetShapeSize(pertokenScaleShape);
        long long commQuantScale1ShapeSize = GetShapeSize(commQuantScale1Shape);
        long long commQuantScale2ShapeSize = GetShapeSize(commQuantScale2Shape);
        long long x3ShapeSize = GetShapeSize(x3Shape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<int32_t> biasHostData(biasShapeSize, 1);
        std::vector<int32_t> dequantScaleHostData(dequantScaleShapeSize, 1);
        std::vector<int32_t> pertokenScaleHostData(pertokenScaleShapeSize, 1);
        std::vector<int16_t> commQuantScale1HostData(commQuantScale1ShapeSize, 1);
        std::vector<int16_t> commQuantScale2HostData(commQuantScale2ShapeSize, 1);
        std::vector<int16_t> x3HostData(x3ShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // 创建 tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        if (args.format == "NZ") {
            ret = CreateWeightNzAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2, args);
        } else {
            ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        }
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                              aclDataType::ACL_FLOAT, &dequantScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                              aclDataType::ACL_FLOAT, &pertokenScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(commQuantScale1HostData, commQuantScale1Shape, &commQuantScale1DeviceAddr,
                              aclDataType::ACL_FLOAT16, &commQuantScale1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(commQuantScale2HostData, commQuantScale2Shape, &commQuantScale2DeviceAddr,
                              aclDataType::ACL_FLOAT16, &commQuantScale2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // 调用第一段接口
        ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(x1, x2, bias, x3, dequantScale, pertokenScale,
                                                          commQuantScale1, commQuantScale2, hcom_name,
                                                          "sum", commTurn, streamMode, out,
                                                          &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnQuantMatmulAllReduceV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用第二段接口
        ret = aclnnQuantMatmulAllReduceV3(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV3 failed. ERROR: %d\n", ret); return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAllReduceV3 execute success \n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (dequantScale != nullptr) {
            aclDestroyTensor(dequantScale);
        }
        if (pertokenScale != nullptr) {
            aclDestroyTensor(pertokenScale);
        }
        if (commQuantScale1 != nullptr) {
            aclDestroyTensor(commQuantScale1);
        }
        if (commQuantScale2 != nullptr) {
            aclDestroyTensor(commQuantScale2);
        }
        if (x3 != nullptr) {
            aclDestroyTensor(x3);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (dequantScaleDeviceAddr != nullptr) {
            aclrtFree(dequantScaleDeviceAddr);
        }
        if (pertokenScaleDeviceAddr != nullptr) {
            aclrtFree(pertokenScaleDeviceAddr);
        }
        if (commQuantScale1DeviceAddr != nullptr) {
            aclrtFree(commQuantScale1DeviceAddr);
        }
        if (commQuantScale2DeviceAddr != nullptr) {
            aclrtFree(commQuantScale2DeviceAddr);
        }
        if (x3DeviceAddr != nullptr) {
            aclrtFree(x3DeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[]) {
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // 初始化集合通信域
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(
                    new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduce, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

