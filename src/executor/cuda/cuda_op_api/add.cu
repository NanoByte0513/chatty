#include "add.cuh"

namespace chatty {
namespace cuda {

Status add(cublasHandle_t handle, float alpha=1.0f, const Tensor& x, Tensor& y) {
    DType dtype = x.dtype();
    Shape shape = x.shape();
    if(dtype != y.dtype() || shape != y.shape()) {
        // TODO: log
        return StatusCode::CHATTY_STATUS_FAILURE;
    }
    switch(dtype) {
    case DType::Float16:
        return add_fp16(
            handle, alpha, 
            static_cast<const __half*>(x.data()), 
            static_cast<const __half*>(y.data()),
            shape.num_elements()
        );
    }
    case DType::Float32:
        return add_fp32(
            handle, alpha,
            static_cast<const float*>(x.data()), 
            static_cast<const float*>(y.data()),
        );
}

Status add_fp16(cublasHandle_t handle, float alpha=1.0f, const __half* x, __half* y, int m, int n) {
    // cublasStatus_t status = cublasGemmEx(
    //     handle,          // cuBLAS 上下文句柄
    //     CUBLAS_OP_N,       // A 矩阵转置标志（N/T/C）
    //     CUBLAS_OP_N,       // B 矩阵转置标志（N/T/C）
    //     m, n, k,            // 矩阵维度：C = [m×n], A = [m×k], B = [k×n]
    //     &alpha,              // 标量系数 α
    //     x,                  // A 矩阵指针
    //     CUDA_R_16F,           // A 数据类型（如 CUDA_R_16F）
    //     m,                        // A 的主维度（列主序下为行数）
    //     y,                  // B 矩阵指针
    //     CUDA_R_16F,           // B 数据类型
    //     k,                        // B 的主维度
    //     const void *beta,               // 标量系数 β
    //     void *C,                        // C 矩阵指针（输出）
    //     cudaDataType_t Ctype,           // C 数据类型
    //     int ldc,                        // C 的主维度
    //     cudaDataType_t CUDA_R_32F,     // 内部计算精度（如 CUDA_R_32F）
    //     CUBLAS_GEMM_DEFAULT             // 算法选择
    // );

}

Status add_fp32(cublasHandle_t handle, float alpha=1.0f, const float* x, float* y, int m, int n) {
    cublasSaxpy(
        handle,   // cuBLAS句柄
        m * n,        // 元素总数
        &alpha,   // α的指针
        x,      // 输入x
        1,        // 连续访问（步长1）
        y,      // 输入/输出y
        1         // 连续访问
    );
}

Status add_int8(cublasHandle_t handle, float alpha=1.0f, const int8_t* x, const int8_t* y, int m, int n);

Status add_int16(cublasHandle_t handle, float alpha=1.0f, const int16_t* x, const int16_t* y, int m, int n);

Status add_int32(cublasHandle_t handle, float alpha=1.0f, const int32_t* x, const int32_t* y, int m, int n);

} // namespace cuda
} // namespace chatty