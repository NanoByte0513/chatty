#pragma once
#include <vector>
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/shape.h"
#include "framework/dtype.h"
#include "flatbuffers/model_generated.h"

namespace chatty {

class FBS_ScaleInfo_t {
public:
    const char* name() const;
    Shape shape() const;
    DType dtype() const;
    int32_t zero_point() const;

    const void* data() const;
    const int8_t* dataAsCstInt8() const;
    const uint8_t* dataAsCstUInt8() const;
    const int16_t* dataAsCstInt16() const;
    const uint16_t* dataAsCstUInt16() const;
    const int32_t* dataAsCstInt32() const;
    const uint32_t* dataAsCstUInt32() const;
    const int64_t* dataAsCstInt64() const;
    const uint64_t* dataAsCstUInt64() const;
    const float* dataAsCstFloat() const;
    int8_t* dataAsInt8() const;
    uint8_t* dataAsUInt8() const;
    int16_t* dataAsInt16() const;
    uint16_t* dataAsUInt16() const;
    int32_t* dataAsInt32() const;
    uint32_t* dataAsUInt32() const;
    int64_t* dataAsInt64() const;
    uint64_t* dataAsUInt64() const;
    float* dataAsFloat() const;

private:
    const void* p_data_ = nullptr;
};

class FBS_Tensor_t {
public:
    const char* name() const;
    Shape shape() const;
    DType dtype() const;
    FBS_ScaleInfo_t scale() const;

    const void* data() const;
    const int8_t* dataAsCstInt8() const;
    const uint8_t* dataAsCstUInt8() const;
    const int16_t* dataAsCstInt16() const;
    const uint16_t* dataAsCstUInt16() const;
    const int32_t* dataAsCstInt32() const;
    const uint32_t* dataAsCstUInt32() const;
    const int64_t* dataAsCstInt64() const;
    const uint64_t* dataAsCstUInt64() const;
    const float* dataAsCstFloat() const;
    int8_t* dataAsInt8() const;
    uint8_t* dataAsUInt8() const;
    int16_t* dataAsInt16() const;
    uint16_t* dataAsUInt16() const;
    int32_t* dataAsInt32() const;
    uint32_t* dataAsUInt32() const;
    int64_t* dataAsInt64() const;
    uint64_t* dataAsUInt64() const;
    float* dataAsFloat() const;

private:
    const void* p_data_ = nullptr;
};

class iChattyModel {
public:
    virtual Status init() = 0;

private:
    
    
};

class ChattyModel : public iChattyModel {
public:
    ChattyModel();
    ~ChattyModel();
    Status init();
private:
    const void* p_data_;
    const chatty_fbs::Model* p_model_;
};

} // namespace chatty