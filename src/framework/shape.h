#pragma once
#include <vector>
#include "stdint.h"

namespace chatty {
class Shape {
public:
    Shape() = default;
    explicit Shape(const std::vector<int64_t>& dimensions);

    const int& operator[](int index) const;  // const 版本

private:
    std::vector<int64_t> dims_;
    int ndim_ = 0;
};
}