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

void run_gemm_thread_coarsen_v1(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
)
{
    const int TILE_WIDTH = 32;
    const int COARSEN_FACTOR = 2;

    assert(TILE_WIDTH * COARSEN_FACTOR <= DIM);

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        (DIM + (block_dim.x * COARSEN_FACTOR) - 1) / (block_dim.x * COARSEN_FACTOR),
        (DIM + block_dim.y - 1) / block_dim.y
    );

    kernel3_thread_coarsen_v1 <TILE_WIDTH, COARSEN_FACTOR>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void run_gemm_thread_coarsen_v2(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
)
{
    // We fix the number of threads per block and vary the number
    // of elements computed per thread according to tile size, which
    // is a tunable parameter (maximum allowed tile width along any
    // dimension = 128)
    const int THREAD_BLOCK_DIM_X = 2;
    const int THREAD_BLOCK_DIM_Y = 2;
    dim3 block_dim(THREAD_BLOCK_DIM_X, THREAD_BLOCK_DIM_Y);
    // dim3 grid_dim(
    //     (DIM + THREAD_BLOCK_DIM_X - 1) / THREAD_BLOCK_DIM_X,
    //     (DIM + THREAD_BLOCK_DIM_Y - 1) / THREAD_BLOCK_DIM_Y
    // );
    dim3 grid_dim(2,2);

    printf("DIM: %d\n", DIM);
    printf("blockDim.x: %d, blockDim.y: %d\n", block_dim.x, block_dim.y);
    printf("gridDim.x: %d, gridDim.y: %d\n", grid_dim.x, grid_dim.y);

    // Each thread block is responsible for a TILE_M x TILE_N tile
    // of C - each such tile is computed as a sum of products between
    // A's tiles (of size TILE_M x TILE_K) and B's tiles (of size
    // TILE_K x TILE_N)
    const int BLOCK_TILE_M = 4;
    const int BLOCK_TILE_N = 4;
    const int BLOCK_TILE_K = 4;

    // Number of C's elements computed by each thread in the
    // horizontal and vertical directions respectively
    const int THREAD_TILE_M = (BLOCK_TILE_M + THREAD_BLOCK_DIM_X - 1) / THREAD_BLOCK_DIM_X;
    const int THREAD_TILE_N = (BLOCK_TILE_N + THREAD_BLOCK_DIM_Y - 1) / THREAD_BLOCK_DIM_Y;

    kernel4_thread_coarsen_v2
        <BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}