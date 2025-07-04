#pragma once
#include "framework/status.h"
#include "framework/dtype.h"
#include "framework/tensor.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace chatty {
namespace cuda {
Status linear(cublasHandle_t handle, const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out);
Status linear(cublasHandle_t handle, const Tensor& x, const Tensor& weight, Tensor& out);

Status linear_bf16(cublasHandle_t handle, const __nv_bfloat16* x, const __nv_bfloat16* weight, float* out, int32_t m, int32_t k, int32_t n);

/**
 * * Linear operation for fp16 inputs and outputs.
 * * @param x Input tensor of shape (m, k) in fp16.
 * * @param weight Weight tensor of shape (k, n) in fp16.
 * * @param bias Bias tensor of shape (n,) in fp16 or fp32.
 * * @param out Output tensor of shape (m, n) in fp16.
 * * @return chatty::Status.
 */
Status linear_fp16(cublasHandle_t handle, const __half* x, const __half* weight, const __half* bias, float* out, int32_t m, int32_t k, int32_t n);
Status linear_fp16(cublasHandle_t handle, const __half* x, const __half* weight, const float* bias, float* out, int32_t m, int32_t k, int32_t n);

/**
 * * Linear operation for int8 weights and fp16 inputs and outputs.
 * * @param x Input tensor of shape (m, k) in fp16.
 * * @param weight Weight tensor of shape (k, n) in int8.
 * * @param bias Bias tensor of shape (n,) in fp16 or fp32.
 * * @param out Output tensor of shape (m, n) in fp16.
 * * @return chatty::Status.
 */
Status linear_w8fp16(cublasHandle_t handle, const __half* x, const int8_t* weight, const __half* bias, __half* out, int32_t m, int32_t k, int32_t n);
Status linear_w8fp16(cublasHandle_t handle, const __half* x, const int8_t* weight, const float* bias, __half* out, int32_t m, int32_t k, int32_t n);

/**
 * * Linear operation for int16 inputs and outputs with int8 weights.
 * * @param x Input tensor of shape (m, k) in int16.
 * * @param weight Weight tensor of shape (k, n) in int8.
 * * @param bias Bias tensor of shape (n,) in fp16 or fp32.
 * * @param out Output tensor of shape (m, n) in int16.
 * * @return chatty::Status.
 */
Status linear_w8a16(cublasHandle_t handle, const int16_t* x, const int8_t* weight, const __half* bias, int16_t* out, int32_t m, int32_t k, int32_t n);
Status linear_w8a16(cublasHandle_t handle, const int16_t* x, const int8_t* weight, const float* bias, int16_t* out, int32_t m, int32_t k, int32_t n);

/**
 * * Linear operation for int8 inputs and outputs with int8 weights.
 * * @param x Input tensor of shape (m, k) in int8.
 * * @param weight Weight tensor of shape (k, n) in int8.
 * * @param bias Bias tensor of shape (n,) in fp16 or fp32.
 * * @param out Output tensor of shape (m, n) in int8.
 * * @return chatty::Status.
 */
Status linear_w8a8(cublasHandle_t handle, const int8_t* x, const int8_t* weight, const __half* bias, int8_t* out, int32_t m, int32_t k, int32_t n);
Status linear_w8a8(cublasHandle_t handle, const int8_t* x, const int8_t* weight, const float* bias, int8_t* out, int32_t m, int32_t k, int32_t n);

} // namespace cuda
} // namespace chatty
