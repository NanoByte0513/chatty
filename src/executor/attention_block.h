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
    AttnBlock();
    AttnBlock(AttnParam param): param_(param) {};
    ~AttnBlock();
    Status forward(const Tensor& x, Tensor& out);

protected:
    AttnParam param_;
};
} // namespace chatty
