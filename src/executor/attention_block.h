#pragma once
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/dtype.h"

namespace chatty {
struct AttnParam {
    Tensor& q_weight;
    Tensor& k_weight;
    Tensor& v_weight;
    Tensor& o_weight;

    QuantType quant_type;
};

class AttnBlock {
public:
    
    virtual Status forward(const Tensor& x, Tensor& out) = 0;
    virtual Status loadParams(std::shared_ptr<ChattyModel> p_model) = 0;

protected:
    AttnBlock(AttnParam param):param_(param) {};

protected:
    AttnParam param_;
};
} // namespace chatty
