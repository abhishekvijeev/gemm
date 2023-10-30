// Host stubs for each CUDA kernel

#include <iostream>

#include <kernels.cuh>
#include <runner.h>

#include <helper.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/host_tensor.h>

void run_sgemm_naive(
    float *A,
    float *B,
    float *C, 
    int M,
    int N,
    int K, 
    float alpha,
    float beta
)
{
    // Max threads in a thread block = 1024
    dim3 block_dim(32, 32);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    
    kernel1_naive<<<grid_dim, block_dim>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void run_sgemm_shmem(
    float *A,
    float *B,
    float *C, 
    int M,
    int N,
    int K, 
    float alpha,
    float beta,
    size_t shmem_per_sm
)
{
    // Max threads in a thread block = 1024
    dim3 block_dim(32, 32);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);
    
    kernel2_shmem<<<grid_dim, block_dim>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}