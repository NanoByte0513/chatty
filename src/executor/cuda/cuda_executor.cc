#include "cuda_executor.h"

namespace chatty {
namespace cuda {

Status CUDAExecutor::init(std::shared_ptr<ChattyModel> p_model) {
    p_model_ = p_model;
}

Status CUDAExecutor::prefill(const std::vector<int32_t> &token_ids, float* logits) {

}

Status CUDAExecutor::decode(const std::vector<int32_t> &token_ids, KVCache kv_cache, float* logits) {

}

Status CUDAExecutor::forward(const std::vector<int32_t> &token_ids, KVCache kv_cache, float* logits) {

}

Status CUDAExecutor::tokenize(const char* prompt, int32_t* token_ids, int32_t& token_num) {

}

Status CUDAExecutor::detokenize(char* prompt, const int32_t* token_ids, int32_t token_num) {

}

} // namespace cuda
} // namespace chatty