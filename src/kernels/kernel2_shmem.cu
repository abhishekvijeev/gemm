#include <kernels.cuh>

/*
 * Shared Memory Tiling
 *
 * Key Idea - Partition the data into subsets called tiles so that
 * each tile fits into shared memory
 *
 * Caveats:
 * a) It must be feasible for kernel computation on these tiles
 * to be performed independently of each other
 * b) Not all data structures are amenable to tile-based
 * partitioning
 *
 * Tiling - A program transformation technique that localizes the
 * memory locations accessed among threads and the timing of their
 * accesses. It divides the long access sequences of each thread
 * into phases and uses barrier synchronization to keep the timing
 * of accesses to each section at close intervals. This technique
 * controls the amount of on-chip memory required by localizing the
 * accesses both in time and in space.
 * 
 * Technique - Threads collaboratively load subsets of the input
 * matrices (A and B) into shared memory before they individually use
 * these matrices in their dot product computations. The dot product
 * calculations performed by each thread are now divided into phases
 * In general, if an input matrix is of the dimension Width and the
 * tile size is referred to as TILE_WIDTH, the dot product would be
 * performed in Width/TILE_WIDTH phases. The creation of these phases,
 * termed 'strip mining', is key to reducing global memory accesses.
 *
 * A kernel's shared memory usage affects occupancy (ratio of active
 * warps/threads to the maximum number of warps/threads that can be
 * scheduled on an SM). If it exceeds the maximum amount of shared
 * memory that's available per SM, occupancy decreases.
 * 
 */

#define TILE_WIDTH 16

__global__ void kernel2_shmem(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int row = bidy * TILE_WIDTH + tidy;
    int col = bidx * TILE_WIDTH + tidx;
    int num_phases = M / TILE_WIDTH;
    float val = 0;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    for (int phase = 0; phase < num_phases; phase++) {
        tileA[tidy][tidx] = A[row * N + phase * TILE_WIDTH + tidx];
        tileB[tidy][tidx] = B[(phase * TILE_WIDTH + tidy) * K + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            val += tileA[tidy][k] * tileB[k][tidx];
        }
        __syncthreads();
    }
    C[row * N + col] = val;
}