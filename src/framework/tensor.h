#pragma once
#include "framework/dtype.h"
#include "framework/shape.h"

namespace chatty {
class Tensor {
public:
    Tensor() = default;
    Tensor(const void* data, DType dtype, const Shape& shape);

    DType dtype() const;
    Shape shape() const;

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
    const void* data_ = nullptr;
    DType dtype_ = DType::Float32;
    Shape shape_ = {};
};
} // namespace chatty