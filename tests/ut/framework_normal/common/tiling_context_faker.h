/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H

#include "op_tiling_context_builder.h"
#include "any_value.h"

namespace gert {

class TilingContextPara {
public:
    class TensorDescription {
    public:
        TensorDescription(const gert::StorageShape& shape, ge::DataType dtype, ge::Format format) :
            shape_(shape), dtype_(dtype), format_(format) {}
    public:
        gert::StorageShape shape_;
        ge::DataType dtype_ = ge::DT_FLOAT;
        ge::Format format_ = ge::FORMAT_ND;
    };

    class OpAttr {
    public:
        OpAttr(const std::string& attrName, const Ops::Transformer::AnyValue& attr) : attrName_(attrName), attr_(attr) {}
    public:
        std::string attrName_;
        Ops::Transformer::AnyValue attr_;
    };
public:
    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      const std::vector<OpAttr>& attrs,
                      void* compileInfo = nullptr,
                      const std::string& socVersion = "Ascend910B",
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      attrs_(attrs),
                      socVersion_(socVersion),
                      compileInfo_(compileInfo),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}

    TilingContextPara(const std::string& opName,
                      const std::vector<TensorDescription>& inputTensorDesc,
                      const std::vector<TensorDescription>& outputTensorDesc,
                      void* compileInfo = nullptr,
                      const std::string& socVersion = "Ascend910B",
                      uint64_t coreNum = 64,
                      uint64_t ubSize = 262144,
                      uint64_t tilingDataSize = 4096) : 
                      opName_(opName),
                      inputTensorDesc_(inputTensorDesc),
                      outputTensorDesc_(outputTensorDesc),
                      compileInfo_(compileInfo),
                      socVersion_(socVersion),
                      coreNum_(coreNum),
                      ubSize_(ubSize),
                      tilingDataSize_(tilingDataSize) {}

public:
    std::string opName_;
    std::vector<TensorDescription> inputTensorDesc_;
    std::vector<TensorDescription> outputTensorDesc_;
    std::vector<OpAttr> attrs_;
    uint64_t coreNum_        = 64;
    uint64_t ubSize_         = 262144;
    uint64_t tilingDataSize_ = 4096;
    void* compileInfo_ = nullptr;
    std::string socVersion_;
};

class TilingContextFaker : public OpTilingContextBuilder {
public:
    TilingContextFaker& SetOpType(const std::string opType);

    /* only one can be choosed from IrInstanceNum */
    TilingContextFaker& NodeIoNum(size_t inputNum, size_t outputNum);

    /* can be used for dynamic inputs/outputs
     * only one can be choosed from NodeIoNum */
    TilingContextFaker& IrInstanceNum(const std::vector<uint32_t>& inputInstanceNum,
                                      const std::vector<uint32_t>& outputInstanceNum);

    TilingContextFaker& NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                    ge::Format storageFormat);

    TilingContextFaker& NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                     ge::Format storageFormat);

    TilingContextFaker& Attr(const std::string& attrName, bool attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, int64_t attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, float attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const AscendString& attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<bool>& attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<int64_t>& attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<float>& attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<AscendString>& attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }
    TilingContextFaker& Attr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attr) {
        OpTilingContextBuilder::MutableOpInfo().Attr(attrName.c_str(), attr);
        return *this;
    }

    TilingContextFaker& InputTensors(const std::vector<Tensor *>& inputTensors);

    TilingContextFaker& OutputTensors(const std::vector<Tensor *>& outputTensors);

    TilingContextFaker& CompileInfo(const void* compileInfo);

    TilingContextFaker& PlatformInfo(const void* platformInfo);

    TilingContextFaker& DeterministicInfo(int32_t* deterministicInfo);

    TilingContextFaker& TilingData(const void* tilingData);

    TilingContextFaker& Workspace(const ContinuousVector* workspace);

    ContextHolder<TilingContext> Build();
};

} // namespace gert
#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CONTEXT_FAKER_H
