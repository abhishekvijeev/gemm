#include <kernels.cuh>

/*
 * Threads with consecutive thread IDs within a thread block belong to a warp,
 * whose threads execute the same instruction in lock-step (SIMD).
 * (where threadID = tid.y * blockDim. + tid.x)
 * As of this writing, Nvidia GPUs group 32 threads into a warp.
 *
 * Points to remember about global memory accesses:
 *
 *      (a) Modern GPUs support loading 32, 64 or 128 bytes from global memory
 *          in a single bus transaction. Therefore, if each thread within a
 *          warp loads a 32-bit (4-byte) float value from global memory, such
 *          that all loads are to consecutive global memory addresses, the warp
 *          scheduler can coalesce this 128-byte load (32 threads * 4 bytes per
 *          thread) into a single transaction. Otherwise, the GPU will execute
 *          as many 32-byte transactions as necessary to fetch all floats,
 *          leading to wasted memory bandwidth.
 *
 *      (b) If all threads within a warp issue a load to the same global memory
 *          address again, only a single memory bus transaction is performed
 *          because the value retrieved from memory is broadcast to all threads
 *          within the warp.
 *
 * This naive kernel uses both the above mechanisms to avoid multiple memory
 * bus transactions. Since the kernel computes each element of C using an inner
 * product, during each iteration of the inner loop, all threads within a warp
 * load the same element of matrix A, which results in a single bus transaction
 * whose value is broadcast to all other threads. Similarly, during each
 * iteration of the inner loop, all threads within a warp access consecutive
 * elements (belonging to the same row) of matrix B, due to which they are all
 * coalesced into a single bus transaction.
 *
 * However, each iteration of the inner loop performs two global memory
 * accesses and two floating point operations. One global memory access fetches
 * an element of A, and the other fetches an element of B. One floating-point
 * operation multiplies the elements fetched from A and B, and the other
 * accumulates the product into C_acc. Thus, the loop has a
 * compute-to-global-memory-access ratio only 1.0, which leads to severe
 * under-utilization of the peak execution speed of modern GPUs.
 */
__global__ void kernel1_naive(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    // This thread computes C[row][col] = alpha * (A[row] dot B[col]) + beta * C[row][col]
    // where A[row] is a row vector and B[col] is a column vector
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float acc = 0.0;

    if (row < N && col < M) {
        for (int i = 0; i < K; i++) {
            // A[row][i] * B[i][col]
            acc += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}
