#include "cuda_attention_block.cuh"

namespace chatty {
namespace cuda {

CUDAAttnBlock::CUDAAttnBlock(AttnParam param): AttnBlock(param) {
    
}

CUDAAttnBlock::~CUDAAttnBlock() {

}

Status CUDAAttnBlock::forward(const Tensor& x, Tensor& out){

}

} // namespace cuda
} // namespace chatty