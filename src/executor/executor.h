#pragma once
#include <vector>
#include "stdint.h"
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/shape.h"
#include "framework/kvcache.h"
#include "framework/model.h"
#include "executor/attention_block.h"
#include "executor/ffn_block.h"

namespace chatty {
class Executor {
public:
    virtual Status init(std::shared_ptr<ChattyModel> p_model) = 0;
    virtual Status prefill(const std::vector<int32_t> &token_ids, float* logits) = 0;
    virtual Status decode(const std::vector<int32_t> &token_ids, KVCache kv_cache, float* logits) = 0;
    virtual Status forward(const std::vector<int32_t> &token_ids, KVCache kv_cache, float* logits, int32_t batch_size, int32_t seq_len) = 0;
    virtual Status tokenize(const char* prompt, int32_t* token_ids, int32_t& token_num) = 0;
    virtual Status detokenize(char* prompt, const int32_t* token_ids, int32_t token_num) = 0;

    virtual Status embedding(const std::vector<int32_t> &token_ids, Tensor& out) = 0;
    virtual Status normalize(const Tensor& x, Tensor& out) = 0;
    virtual Status topk(int32_t k, const Tensor& logits, std::vector<float>& topk_logits, std::vector<int>& topk_tokens) = 0;

protected:
    int32_t dev_id_;
    std::vector<AttnBlock> attn_blocks_;
    std::vector<FFNBlock> ffn_blocks_;
    std::shared_ptr<ChattyModel> p_model_;
};
}