#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(func) \
    do { \
        cudaError_t status = (func); \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(func) \
    do { \
        cublasStatus_t status = (func); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    // 1. 初始化矩阵参数（可修改）
    const int M = 1024; // 矩阵A的行数
    const int N = 1024; // 矩阵B的列数
    const int K = 1024; // 矩阵A的列数/矩阵B的行数
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 2. 创建主机端数据
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N); // 用于验证的CPU结果

    // 初始化随机数据
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 3. 分配设备端内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 4. 初始化cuBLAS句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 5. 数据传输到GPU
    CHECK_CUBLAS(cublasSetMatrix(M, K, sizeof(float), h_A.data(), M, d_A, M));
    CHECK_CUBLAS(cublasSetMatrix(K, N, sizeof(float), h_B.data(), K, d_B, K));

    // 6. 执行矩阵乘法（核心操作）
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUBLAS(
        cublasSgemm(handle,
                    CUBLAS_OP_N,  // A不转置（因cuBLAS默认列优先，此处实际需转置）
                    CUBLAS_OP_N,  // B不转置
                    M,            // 结果矩阵行数
                    N,            // 结果矩阵列数
                    K,            // 累加维度
                    &alpha,
                    d_A, M,       // A的leading dimension = M
                    d_B, K,       // B的leading dimension = K
                    &beta,
                    d_C, M)       // C的leading dimension = M
    );
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // 7. 取回结果
    CHECK_CUBLAS(cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C.data(), M));

    // 8. 计算耗时
    float duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f;
    float gflops = (2.0f * M * N * K) / (duration_ms * 1e6); // 计算GFLOPS

    // 9. 验证结果（简单验证前10个元素）
    for (int i = 0; i < 10; ++i) {
        int row = rand() % M, col = rand() % N;
        float cpu_val = 0.0f;
        for (int k = 0; k < K; ++k) {
            cpu_val += h_A[row * K + k] * h_B[k * N + col];
        }
        float diff = fabs(h_C[row * N + col] - cpu_val);
        if (diff > 1e-5) {
            std::cerr << "Validation failed at (" << row << "," << col << "): "
                      << "GPU=" << h_C[row * N + col] << ", CPU=" << cpu_val << std::endl;
        }
    }

    // 10. 输出性能数据
    std::cout << "Matrix: [" << M << "x" << K << "] * [" << K << "x" << N << "]\n"
              << "Time: " << duration_ms << " ms\n"
              << "Performance: " << gflops << " GFLOPS" << std::endl;

    // 11. 清理资源
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}