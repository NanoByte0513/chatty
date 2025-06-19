#pragma once
#include <vector>
#include "stdint.h"

namespace chatty {
class Shape {
public:
    Shape() = default;
    explicit Shape(const std::vector<size_t>& dimensions);
    int32_t ndim() const;
    size_t num_elements() const;

    int operator[](int index) const;
    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;

private:
    std::vector<size_t> dims_;
    int32_t ndim_ = 0;
};
}