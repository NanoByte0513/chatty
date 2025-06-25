#pragma once
#include "executor/attention_block.h"

namespace chatty {
namespace cuda {
class CUDAAttnBlock: public AttnBlock {
    CUDAAttnBlock(AttnParam param);
    ~CUDAAttnBlock();
    Status forward(const Tensor& x, Tensor& out);
};

} // namespace cuda
} // namespace chatty