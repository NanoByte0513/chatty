#include "linear.cuh"
#include "cublas_v2.h"

namespace chatty {
namespace cuda {
Status linear(cublasHandle_t handle, const Tensor& x, const Tensor& weight, const Tensor& bias, Tensor& out) {
    DType input_dtype = x.dtype();
    DType weight_dtype = weight.dtype();
    int m = x.shape()[0];
    int k = x.shape()[1];
    int n = weight.shape()[1];
    switch(weight_dtype) {
    case DType::Float16:
        switch(input_dtype) {
        case DType::Float16:
            return linear_fp16(handle, (const __half*)x.data(), (const __half*)weight.data(), (const __half*)bias.data(), out.dataAsFloat(), m, k, n);
        }

    case DType::BF16:
        switch(input_dtype) {
        case DType::BF16:
            return linear_bf16(handle, (const __nv_bfloat16*)x.data(), (const __nv_bfloat16*)weight.data(), out.dataAsFloat(), m, k, n);
        }

    default:
        return StatusCode::CHATTY_STATUS_FAILURE;
    }
    return StatusCode::CHATTY_STATUS_SUCCESS;
}

Status linear(cublasHandle_t handle, const Tensor& x, const Tensor& weight, Tensor& out) {
    DType input_dtype = x.dtype();
    DType weight_dtype = weight.dtype();
    int m = x.shape()[0];
    int k = x.shape()[1];
    int n = weight.shape()[1];
    switch(weight_dtype) {
    case DType::Float16:
        switch(input_dtype) {
        case DType::Float16:
            return linear_fp16(handle, (const __half*)x.data(), (const __half*)weight.data(), nullptr, out.dataAsFloat(), m, k, n);
        }
    default:
        return StatusCode::CHATTY_STATUS_FAILURE;
    }
    return StatusCode::CHATTY_STATUS_SUCCESS;
}

// Status linear_fp16(cublasHandle_t handle, const __half* x, const __half* weight, const __half* bias, float* out, int m, int k, int n) {
//     // C = alpha * AB + beta * C
//     cublasStatus_t status = cublasGemmEx(
//         cublasHandle_t handle,          // cuBLAS 上下文句柄
//         cublasOperation_t transA,       // A 是否转置
//         cublasOperation_t transB,       // B 是否转置
//         int m, int n, int k,            // 矩阵维度（C = m×n, A = m×k, B = k×n）
//         const void *alpha,              // 缩放因子（float 或 __half 指针）
//         const void *A, cudaDataType Atype, int lda,  // A 矩阵及参数
//         const void *B, cudaDataType Btype, int ldb,  // B 矩阵及参数
//         const void *beta,               // C 的缩放因子
//         void *C, cudaDataType Ctype, int ldc, // C 矩阵及参数
//         cudaDataType ComputeType,       // 计算精度（如 CUDA_R_32F）
//         cublasGemmAlgo_t algo           // 算法选择（启用 Tensor Core）
//     );
// }

Status linear_bf16(cublasHandle_t handle, const __nv_bfloat16* x, const __nv_bfloat16* weight, float* out, int m, int k, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // A、B是否需要转置
        m, n, k,
        &alpha,
        weight, CUDA_R_16BF, n,      // A类型为BF16
        x, CUDA_R_16BF, k,      // B类型为BF16
        &beta,
        out, CUDA_R_32F, n,       // C类型为FP32（累加防精度损失）
        CUDA_R_32F,               // 计算精度为FP32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP // 启用Tensor Core
    );
    return StatusCode::CHATTY_STATUS_SUCCESS;
}

} // namespace cuda
} // namespace chatty
