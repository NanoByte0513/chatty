#pragma once
#include "framework/dtype.hpp"
#include "framework/shape.hpp"

namespace chatty {
class Tensor {
public:
    Tensor() = default;
    Tensor(const void* data, DType dtype, const Shape& shape);

    const void* getData() const;
    DType getDType() const;

private:
    const void* data_ = nullptr;
    DType dtype_ = DType::Float32;
    Shape shape_ = {};
};
} // namespace chatty