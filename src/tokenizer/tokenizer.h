#pragma once
#include "framework/status.h"
#include <cstddef>
#include <cstdint>

namespace chatty {
class Tokenizer {
public:
    static Status createTokenizer(Tokenizer* tokenizer, const void* data, size_t data_size);

protected:
    virtual Status encode() const = 0;
    virtual Status decode() const = 0;
    virtual bool is_bos(int32_t token_id) const = 0;
    virtual bool is_eos(int32_t token_id) const = 0;
    virtual int32_t vocab_size() const = 0;

};

} // namespace chatty