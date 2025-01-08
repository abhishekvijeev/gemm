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
    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;

    const int THREADBLOCK_TILE_M = 64;
    const int THREADBLOCK_TILE_N = 64;
    const int THREADBLOCK_TILE_K = 64;

    const int THREADBLOCK_DIM_X = THREADBLOCK_TILE_N / THREAD_TILE_N;
    const int THREADBLOCK_DIM_Y = THREADBLOCK_TILE_M / THREAD_TILE_M;
    const int GRID_DIM_X = DIM / THREADBLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / THREADBLOCK_TILE_M;

    dim3 block_dim(THREADBLOCK_DIM_Y, THREADBLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    // printf("grid_dim.x: %d\n", grid_dim.x);
    // printf("grid_dim.y: %d\n", grid_dim.y);
    // printf("block_dim.x: %d\n", block_dim.x);
    // printf("block_dim.y: %d\n", block_dim.y);

    kernel4_thread_coarsen_v2
        <THREADBLOCK_TILE_M, THREADBLOCK_TILE_N, THREADBLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void run_gemm_vectorize(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
)
{
    // const int THREAD_TILE_M = 2;
    // const int THREAD_TILE_N = 2;

    // const int THREADBLOCK_TILE_M = 4;
    // const int THREADBLOCK_TILE_N = 4;
    // const int THREADBLOCK_TILE_K = 4;

    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;

    const int THREADBLOCK_TILE_M = 64;
    const int THREADBLOCK_TILE_N = 64;
    const int THREADBLOCK_TILE_K = 64;

    const int THREADBLOCK_DIM_X = THREADBLOCK_TILE_N / THREAD_TILE_N;
    const int THREADBLOCK_DIM_Y = THREADBLOCK_TILE_M / THREAD_TILE_M;
    const int GRID_DIM_X = DIM / THREADBLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / THREADBLOCK_TILE_M;

    dim3 block_dim(THREADBLOCK_DIM_Y, THREADBLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    // printf("grid_dim.x: %d\n", grid_dim.x);
    // printf("grid_dim.y: %d\n", grid_dim.y);
    // printf("block_dim.x: %d\n", block_dim.x);
    // printf("block_dim.y: %d\n", block_dim.y);

    kernel5_vectorize
        <THREADBLOCK_TILE_M, THREADBLOCK_TILE_N, THREADBLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void run_gemm_cute(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
)
{
    const int THREADBLOCK_TILE_M = 4;
    const int THREADBLOCK_TILE_N = 4;
    const int THREADBLOCK_TILE_K = 4;

    const int THREADBLOCK_DIM_X = 4;
    const int THREADBLOCK_DIM_Y = 4;

    const int GRID_DIM_X = DIM / THREADBLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / THREADBLOCK_TILE_M;

    printf("GRID_DIM_Y: %d\n", GRID_DIM_Y);
    printf("GRID_DIM_X: %d\n", GRID_DIM_X);

    dim3 block_dim(THREADBLOCK_DIM_Y, THREADBLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    kernel6_cute
        <THREADBLOCK_TILE_M, THREADBLOCK_TILE_N, THREADBLOCK_TILE_K>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}

void run_gemm_warptile(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
)
{
    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;

    const int THREADBLOCK_TILE_M = 64;
    const int THREADBLOCK_TILE_N = 64;
    const int THREADBLOCK_TILE_K = 64;

    const int THREADBLOCK_DIM_X = THREADBLOCK_TILE_N / THREAD_TILE_N;
    const int THREADBLOCK_DIM_Y = THREADBLOCK_TILE_M / THREAD_TILE_M;
    const int GRID_DIM_X = DIM / THREADBLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / THREADBLOCK_TILE_M;

    dim3 block_dim(THREADBLOCK_DIM_Y, THREADBLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    // printf("grid_dim.x: %d\n", grid_dim.x);
    // printf("grid_dim.y: %d\n", grid_dim.y);
    // printf("block_dim.x: %d\n", block_dim.x);
    // printf("block_dim.y: %d\n", block_dim.y);

    kernel7_warptile
        <THREADBLOCK_TILE_M, THREADBLOCK_TILE_N, THREADBLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}