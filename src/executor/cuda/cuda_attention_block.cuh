#pragma once
#include "executor/attention_block.h"

namespace chatty {
namespace cuda {
class CudaAttnBlock: public AttnBlock {
    CudaAttnBlock(AttnParam param);
    ~CudaAttnBlock();
    Status forward(const Tensor& x, Tensor& out);
};

} // namespace cuda
} // namespace chatty