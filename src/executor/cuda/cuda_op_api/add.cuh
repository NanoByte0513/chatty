#pragma once
#include "cuda_fp16.h"
#include "framework/status.h"
#include "framework/dtype.h"
#include "framework/tensor.h"
#include <cublas_v2.h>

namespace chatty {
namespace cuda {

Status add(cublasHandle_t handle, float alpha=1.0f, const Tensor& x, Tensor& y);

Status add_fp16(cublasHandle_t handle, float alpha=1.0f, const __half* x, const __half* y, int m, int n);
Status add_fp32(cublasHandle_t handle, float alpha=1.0f, const float* x, const float* y, int m, int n);
Status add_int8(cublasHandle_t handle, float alpha=1.0f, const int8_t* x, const int8_t* y, int m, int n);
Status add_int16(cublasHandle_t handle, float alpha=1.0f, const int16_t* x, const int16_t* y, int m, int n);
Status add_int32(cublasHandle_t handle, float alpha=1.0f, const int32_t* x, const int32_t* y, int m, int n);

// __global__
// void kernel_add_fp16(const __half* x, const __half* y, __half* out, int m, int n);
// __global__
// void kernel_add_fp32(const float* x, const float* y, int m, int n);
// __global__
// void kernel_add_int8(const int8_t* x, const int8_t* y, int8_t* out, int m, int n);
// __global__
// void kernel_add_int16(const int16_t* x, const int16_t* y, int16_t* out, int m, int n);
// __global__
// void kernel_add_int32(const int32_t* x, const int32_t* y, int m, int n);

} // namespace cuda
} // namespace chatty