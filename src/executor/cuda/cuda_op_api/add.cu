// #include "add.cuh"

// namespace chatty {
// namespace cuda {

// Status add(cublasHandle_t handle, float alpha, const Tensor& x, float beta, const Tensor& y, Tensor& out) {
//     DType dtype = x.dtype();
//     Shape shape = x.shape();
//     if(dtype != y.dtype() || shape != y.shape()) {
//         // TODO: log
//         return StatusCode::CHATTY_STATUS_FAILURE;
//     }
//     switch(dtype) {
//     case DType::Float16:
//         return add_fp16(
//             handle,
//             alpha, static_cast<const __half*>(x.data()), 
//             beta, static_cast<const __half*>(y.data()),
//             const_cast<__half*>(static_cast<const __half*>(out.data())),
//             shape[0], shape[1]
//         );
//     case DType::Float32:
//         return add_fp32(
//             handle, 
//             alpha, x.dataAsCstFloat(), 
//             beta, y.dataAsCstFloat(),
//             out.dataAsFloat(),
//             shape[0], shape[1]
//         );
//     default:
//         return StatusCode::CHATTY_STATUS_FAILURE;
//     }
    
// }

// Status add_fp16(cublasHandle_t handle, float alpha, const __half* x, float beta, const __half* y, __half* out, int m, int n) {
//     __half alpha_h = __float2half(alpha);
//     __half beta_h = __float2half(beta);
//     cublasStatus_t status = cublasHgemm(handle,
//                                 CUBLAS_OP_N, CUBLAS_OP_N,
//                                 m, n, 1,       // k=1
//                                 &alpha_h,          // 直接使用__half*类型
//                                 x, lda,
//                                 y, ldb,
//                                 &beta_h,
//                                 out, ldc
//                             );

// }

// Status add_fp32(cublasHandle_t handle, float alpha, const float* x, float beta, const float* y, float* out, int m, int n) {
//     cublasStatus_t status = cublasSaxpy(
//                                 handle,   // cuBLAS句柄
//                                 m * n,        // 元素总数
//                                 &alpha,   // α的指针
//                                 x,      // 输入x
//                                 1,        // 连续访问（步长1）
//                                 y,      // 输入/输出y
//                                 1         // 连续访问
//                             );
//     if(status == CUBLAS_STATUS_SUCCESS) {
//         return StatusCode::CHATTY_STATUS_SUCCESS;
//     } else {
//         return StatusCode::CHATTY_STATUS_FAILURE;
//     }
// }

// Status add_int8(cublasHandle_t handle, float alpha=1.0f, const int8_t* x, const int8_t* y, int m, int n);

// Status add_int16(cublasHandle_t handle, float alpha=1.0f, const int16_t* x, const int16_t* y, int m, int n);

// Status add_int32(cublasHandle_t handle, float alpha=1.0f, const int32_t* x, const int32_t* y, int m, int n);

// } // namespace cuda
// } // namespace chatty