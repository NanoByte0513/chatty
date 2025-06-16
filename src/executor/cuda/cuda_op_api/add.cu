#include "add.cuh"

namespace chatty {
namespace cuda {

Status add(const Tensor& x, const Tensor& y, Tensor& out) {
    DType dtype = x.dtype();
    if(dtype != y.dtype()) {
        // TODO: log
        return StatusCode::CHATTY_STATUS_FAILURE;
    }

    // dim3 grid((x.shape().num_elements() + 255) / 256);
}

Status add_fp16(const __half* x, const __half* y, __half* out,  int m, int n);
Status add_fp32(const float* x, const float* y, float* out, int m, int n);
Status add_int8(const int8_t* x, const int8_t* y, int8_t* out, int m, int n);
Status add_int16(const int16_t* x, const int16_t* y, int16_t* out, int m, int n);
Status add_int32(const int32_t* x, const int32_t* y, int32_t* out, int m, int n);

__global__
void kernel_add_fp16(const __half* x, const __half* y, __half* out, int m, int n);
__global__
void kernel_add_fp32(const float* x, const float* y, float* out, int m, int n);
__global__
void kernel_add_int8(const int8_t* x, const int8_t* y, int8_t* out, int m, int n);
__global__
void kernel_add_int16(const int16_t* x, const int16_t* y, int16_t* out, int m, int n);
__global__
void kernel_add_int32(const int32_t* x, const int32_t* y, int32_t* out, int m, int n);

} // namespace cuda
} // namespace chatty