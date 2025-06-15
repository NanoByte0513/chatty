#pragma once
#include "framework/status.h"
#include "framework/tensor.h"

namespace chatty {
class KVCache {
public:
    KVCache();
    ~KVCache();

private:
    Tensor* k_cache_ = nullptr;
    Tensor* v_cache_ = nullptr;
};

} // namespace chatty
