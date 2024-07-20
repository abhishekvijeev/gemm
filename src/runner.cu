// Host stubs for each CUDA kernel

#include <cmath>
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
    int DIM, 
    float alpha,
    float beta
)
{
    // Max threads in a thread block = 1024
    dim3 block_dim(32, 32);
    dim3 grid_dim((DIM + block_dim.x - 1) / block_dim.x, (DIM + block_dim.y - 1) / block_dim.y);
    
    kernel1_naive<<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void run_sgemm_shmem(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta,
    size_t max_shmem_per_block
)
{
    // The SGEMM tiled matrix multiplication algorithm requires each
    // thread within a thread block to load one element from input
    // matrix A and one element from input matrix B to shared memory.
    // Here, we assume that A and B are square matrices of the same
    // size. Therefore, the amount of shared memory used per thread
    // is equal to 2 * sizeof(element)
    size_t shmem_per_thread = 2 * sizeof(A);

    // We assume the simple case where the thread block dimensions
    // and shared memory tile dimensions are identical. Therefore,
    // each shared memory tile is a square matrix of dimensions
    // (block_dim.y, block_dim.x)
    int max_threads_per_block = std::min(max_shmem_per_block / shmem_per_thread, 1024UL);
    size_t shmem_per_block = max_threads_per_block * shmem_per_thread;
    dim3 block_dim(sqrt(max_threads_per_block), sqrt(max_threads_per_block));
    dim3 grid_dim((DIM + block_dim.x - 1) / block_dim.x, (DIM + block_dim.y - 1) / block_dim.y);

    kernel2_shmem<<<grid_dim, block_dim, shmem_per_block>>>(A, B, C, DIM, alpha, beta, block_dim.y, block_dim.x);
    CUDA_CHECK(cudaGetLastError());
}

void run_gemm_thread_coarsen_row(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
)
{
    dim3 block_dim(32, 32);
    dim3 grid_dim((DIM + block_dim.x - 1) / block_dim.x, (DIM + block_dim.y - 1) / block_dim.y);
    kernel3_thread_coarsen_row<<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}