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
 * The amount of shared memory allocated within a kernel reflects
 * the amount used by each of the kernel's thread blocks. This depends
 * on the algorithm being implemented by the kernel. For example, the
 * tiled matrix multiplication algorithm requires each thread to load
 * one element of input matrix A and one element of input matrix B to
 * shared memory. If A and B have a data type of float, each element
 * occupies 4 bytes of memory, which implies that each thread loads a
 * total of 8 bytes to shared memory. However, each SM has a limited
 * amount of shared memory, that depends on the underlying GPU
 * architecture. For example, A100 GPUs can be configured to use 164KB
 * of shared memory per SM. Taking the example of the tiled matrix
 * multiplication algorithm that requires 8 bytes of shared memory
 * per thread, we see that the maximum amount of shared memory used
 * by a thread block (assuming a maximum of 1024 threads per block)
 * does not exceed 8 * 1024 = 8192 bytes. Therefore, the SM's
 * occupancy (ratio of active
 * warps/threads to the maximum number of warps/threads that can be
 * scheduled on an SM) is not limited by shared memory usage in this case.
 * On the other hand, if the GPU architecture only supported 4KB of
 * shared memory per SM, allocating thread blocks with 1024 threads
 * per block would result in each block using 8KB of shared memory,
 * which exceeds the amount available per SM, leading to lower
 * occupancy. In this case, thread blocks must not contain more than
 * 4096 / 8 = 512 threads. In general, if a GPU architecture supports
 * 'b' bytes of shared memory per SM and a kernel requires 'm' bytes
 * of shared memory per thread, the kernel must be launched with a
 * a maximum of b / m threads per thread block. Therefore, it is
 * desirable for a kernel to be able
 * to use different amounts of shared memory depending on the amount
 * available in hardware. This is enabled with by using the 'extern'
 * keyword in front of the shared memory declaration and omitting
 * the size of the array. However, the array is one-dimensional
 * 
 * The kernel below only considers the case where thread
 * block and shared memory tile dimensions are identical i.e., each
 * thread loads one element from matrix A and one element from matrix
 * B to shared memory. Allocating thread blocks with more threads than
 * tile elements is wasteful because some the additional threads will
 * be idle. On the other hand, allocating thread blocks with fewer
 * threads than tile elements is also wasteful because some tile
 * elements will remain unused.
 */

#define TILE_WIDTH 32

__global__ void kernel2_shmem(float *A, float *B, float *C, int DIM, float alpha, float beta, size_t tile_dim_y, size_t tile_dim_x)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int row = bidy * TILE_WIDTH + tidy;
    int col = bidx * TILE_WIDTH + tidx;
    int num_phases = DIM / TILE_WIDTH;
    float sum = 0;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    for (int phase = 0; phase < num_phases; phase++) {
        tileA[tidy][tidx] = A[row * DIM + phase * TILE_WIDTH + tidx];
        tileB[tidy][tidx] = B[(phase * TILE_WIDTH + tidy) * DIM + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[tidy][k] * tileB[k][tidx];
        }
        __syncthreads();
    }
    C[row * DIM + col] = sum;
}