#pragma once
#include <vector>
#include "stdint.h"
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/shape.h"

namespace chatty {
class Executor {
public:
    virtual Status init() = 0;
    virtual Status prefill(const std::vector<int32_t> &token_ids, float* logits) = 0;
    virtual Status decode(const std::vector<int32_t> &token_ids, float* logits) = 0;
    virtual Status forward(const std::vector<int32_t> &token_ids, float* logits) = 0;

private:
    int32_t dev_id_;
};
}