#include <kernels.cuh>

// Shared Memory Tiling
//
// Key Idea - Partition the data into subsets called tiles so that
// each tile fits into shared memory
//
// Caveats:
//  a) It must be feasible for kernel computation on these tiles
// to be performed independently of each other
//  b) Not all data structures are amenable to tile-based
// partitioning
//
// Tiling - A program transformation technique that localizes the
// memory locations accessed among threads and the timing of their
// accesses. It divides the long access sequences of each thread
// into phases and uses barrier synchronization to keep the timing
// of accesses to each section at close intervals. This technique
// controls the amount of on-chip memory required by localizing the
// accesses both in time and in space.
//
// Technique - Threads collaboratively load subsets of the input
// matrices (A and B) into shared memory before they individually use
// these matrices in their dot product computations. The dot product
// calculations performed by each thread are now divided into phases
// In general, if an input matrix is of the dimension Width and the
// tile size is referred to as TILE_WIDTH, the dot product would be
// performed in Width/TILE_WIDTH phases. The creation of these phases
// is key to the reduction of accesses to global memory.
__global__ void kernel2_shmem(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    extern __shared__ float tile[];
}