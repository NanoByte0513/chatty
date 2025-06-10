#include "framework/tensor.hpp"

namespace chatty {
void layernorm(Tensor& x, const Tensor& alpha, const Tensor& beta, int axes=-1, float epsilon=1e-5);
void rmsnorm(Tensor& x, const Tensor& alpha, int axes=-1, float epsilon=1e-5);
}