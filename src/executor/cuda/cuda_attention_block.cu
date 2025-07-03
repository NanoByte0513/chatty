#include "cuda_attention_block.cuh"

namespace chatty {
namespace cuda {

CudaAttnBlock::CudaAttnBlock(AttnParam param):AttnBlock(param) {
    
}

CudaAttnBlock::~CudaAttnBlock() {

}

Status CudaAttnBlock::forward(const Tensor& x, Tensor& out){

}

} // namespace cuda
} // namespace chatty