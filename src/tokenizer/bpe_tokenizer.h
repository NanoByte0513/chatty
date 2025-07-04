#pragma once
#include "tokenizer.h"

namespace chatty {
class BpeTokenizer : public Tokenizer {
public:
    BpeTokenizer();
    ~BpeTokenizer();

private:
    // std::vector<int32_t> 
};

} // namespace chatty