#pragma once
#include "framework/status.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chatty {
/***
 * 使用Hugging Face的AutoTokenizer生成基准结果，对比此tokenizer的输出
 */
class Tokenizer {
public:
    enum TokenizerType {
        SENTENCEPIECE = 0,
        BPE = 1,
        HUIGGINGFACE = 2
    };
    static Status createTokenizer(Tokenizer* tokenizer, const void* data, size_t data_size);
    static Status createTokenizer(Tokenizer* tokenizer, std::string file_path);

protected:
    virtual Status encode(const std::string& str, std::vector<int32_t>& token_ids) const = 0;
    virtual Status decode() const = 0;
    virtual Status load_vocab() const = 0;
    virtual Status load_merges() const = 0;
    virtual Status load_specials() const = 0;
    virtual bool is_eos(int32_t token_id) const = 0;
    virtual int32_t vocab_size() const = 0;
    std::vector<int32_t> special_tokens_;
    std::vector<int32_t> eos_tokens_;
};

} // namespace chatty