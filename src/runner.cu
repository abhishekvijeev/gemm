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
    dim3 grid_dim(
        (DIM + block_dim.x - 1) / block_dim.x,
        (DIM + block_dim.y - 1) / block_dim.y
    );
    
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
    const int TILE_WIDTH = 32;
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (DIM + block_dim.x - 1) / block_dim.x,
        (DIM + block_dim.y - 1) / block_dim.y
    );

    kernel2_shmem<TILE_WIDTH>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta, block_dim.y, block_dim.x);
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
    const int tileM = 64;
    const int tileN = 64;
    const int tileK = 8;

    const int TILE_WIDTH = 32;
    const int COARSEN_FACTOR = 2;
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (DIM + (block_dim.x * COARSEN_FACTOR) - 1) / (block_dim.x * COARSEN_FACTOR),
        (DIM + block_dim.y - 1) / block_dim.y
    );
    kernel3_thread_coarsen_row <TILE_WIDTH, COARSEN_FACTOR>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}