#pragma once
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/dtype.h"

namespace chatty {
struct FFNParam {
    Tensor& up_weight;
    Tensor& gate_weight;
    Tensor& down_weight;
    float epsilon;

    QuantType quant_type;
};

class FFNBlock {
public:
    virtual Status forward(const Tensor& x, Tensor& out) = 0;
    virtual Status loadParams(std::shared_ptr<ChattyModel> p_model) = 0;

protected:
    FFNParam param_;
};
} // namespace chatty