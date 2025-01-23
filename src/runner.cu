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

    const int BLOCK_TILE_M = 64;
    const int BLOCK_TILE_N = 64;
    const int BLOCK_TILE_K = 64;

    const int BLOCK_DIM_X = BLOCK_TILE_N / THREAD_TILE_N;
    const int BLOCK_DIM_Y = BLOCK_TILE_M / THREAD_TILE_M;
    const int GRID_DIM_X = DIM / BLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / BLOCK_TILE_M;

    dim3 block_dim(BLOCK_DIM_Y, BLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    // printf("grid_dim.x: %d\n", grid_dim.x);
    // printf("grid_dim.y: %d\n", grid_dim.y);
    // printf("block_dim.x: %d\n", block_dim.x);
    // printf("block_dim.y: %d\n", block_dim.y);

    kernel4_thread_coarsen_v2
        <BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
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

    // const int BLOCK_TILE_M = 4;
    // const int BLOCK_TILE_N = 4;
    // const int BLOCK_TILE_K = 4;

    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;

    const int BLOCK_TILE_M = 64;
    const int BLOCK_TILE_N = 64;
    const int BLOCK_TILE_K = 64;

    const int BLOCK_DIM_X = BLOCK_TILE_N / THREAD_TILE_N;
    const int BLOCK_DIM_Y = BLOCK_TILE_M / THREAD_TILE_M;
    const int GRID_DIM_X = DIM / BLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / BLOCK_TILE_M;

    dim3 block_dim(BLOCK_DIM_Y, BLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    // printf("grid_dim.x: %d\n", grid_dim.x);
    // printf("grid_dim.y: %d\n", grid_dim.y);
    // printf("block_dim.x: %d\n", block_dim.x);
    // printf("block_dim.y: %d\n", block_dim.y);

    kernel5_vectorize
        <BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N>
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
    const int BLOCK_TILE_M = 4;
    const int BLOCK_TILE_N = 4;
    const int BLOCK_TILE_K = 4;

    const int BLOCK_DIM_X = 4;
    const int BLOCK_DIM_Y = 4;

    const int GRID_DIM_X = DIM / BLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / BLOCK_TILE_M;

    printf("GRID_DIM_Y: %d\n", GRID_DIM_Y);
    printf("GRID_DIM_X: %d\n", GRID_DIM_X);

    dim3 block_dim(BLOCK_DIM_Y, BLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    kernel6_cute
        <BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K>
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
    // A threadblock computes a BLOCK_TILE_M * BLOCK_TILE_N tile of C
    const int BLOCK_TILE_M = 128;
    const int BLOCK_TILE_N = 128;
    const int BLOCK_TILE_K = 16;

    // A warp computes a WARP_TILE_M * WARP_TILE_N tile of C
    const int WARP_TILE_M = 64;
    const int WARP_TILE_N = 32;

    // A thread computes a THREAD_TILE_M * THREAD_TILE_N tile of C
    const int THREAD_TILE_M = 8;
    const int THREAD_TILE_N = 8;

    // Number of threads along each warp dimension
    const int WARP_DIM_X = WARP_TILE_N / THREAD_TILE_N;
    const int WARP_DIM_Y = WARP_TILE_M / THREAD_TILE_M;
    static_assert((WARP_DIM_X * WARP_DIM_Y) == 32);

    // Number of warps along each threadblock dimension
    const int NUM_WARPS_X = BLOCK_TILE_N / WARP_TILE_N;
    const int NUM_WARPS_Y = BLOCK_TILE_M / WARP_TILE_M;

    // Threadblock dimensions
    const int BLOCK_DIM_X = WARP_DIM_X * NUM_WARPS_X;
    const int BLOCK_DIM_Y = WARP_DIM_Y * NUM_WARPS_Y;

    // Grid dimensions
    const int GRID_DIM_X = DIM / BLOCK_TILE_N;
    const int GRID_DIM_Y = DIM / BLOCK_TILE_M;

    dim3 block_dim(BLOCK_DIM_Y, BLOCK_DIM_X);
    dim3 grid_dim(GRID_DIM_Y, GRID_DIM_X);

    // printf("grid_dim.x: %d\n", grid_dim.x);
    // printf("grid_dim.y: %d\n", grid_dim.y);
    // printf("block_dim.x: %d\n", block_dim.x);
    // printf("block_dim.y: %d\n", block_dim.y);

    kernel7_warptile
        <BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, WARP_TILE_M, WARP_TILE_N, THREAD_TILE_M, THREAD_TILE_N>
        <<<grid_dim, block_dim>>>(A, B, C, DIM, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
}