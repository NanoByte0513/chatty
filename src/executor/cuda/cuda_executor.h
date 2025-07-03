#pragma once
#include "executor/executor.h"

namespace chatty {
namespace cuda {

class CUDAExecutor: public Executor {
public:
    Status init(std::shared_ptr<ChattyModel> p_model) override;
    Status prefill(const std::vector<int32_t> &token_ids, float* logits) override;
    Status decode(const std::vector<int32_t> &token_ids, KVCache kv_cache, float* logits) override;
    Status forward(const std::vector<int32_t> &token_ids, KVCache kv_cache, float* logits) override;
    Status tokenize(const char* prompt, int32_t* token_ids, int32_t& token_num) override;
    Status detokenize(char* prompt, const int32_t* token_ids, int32_t token_num) override;
};

}
}