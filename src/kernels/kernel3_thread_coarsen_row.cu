#include <kernels.cuh>

#include <stdio.h>

/*
 * In kernels 1 and 2, work was parallelized across threads at the
 * finest granularity i.e., each thread was responsible for computing
 * the result for a single element of matrix C. The advantage of
 * parallelizing work at the finest granularity is transparent
 * scalability, which exposes the maximum amount of parallelism to
 * hardware. If the hardware has enough resources to perform all the
 * work in parallel, the application has exposed enough parallelism to
 * fully utilize the hardware. If there aren't enough resources,
 * hardware can simply serialize this work.
 * 
 * However, parallelizing work at the finest granularity often has
 * costs associated with it, which stem from many sources including
 * synchronization overheads, redundant memory loads across thread
 * blocks, etc. If hardware needs to serialize work due to insufficient
 * resources, these costs will have been paid unnecessarily, leading to
 * performance degradation. In such cases, it is beneficial for the
 * programmer to explicitly serialize work and reduce the costs of
 * parallelism. This is typically done by assigning each thread
 * multiple units of work, an approach commonly referred to as "thread
 * coarsening".
 * 
 * In kernel 2, each output tile is processed by a different thread
 * block and because shared memory contents cannot be shared across
 * thread blocks, each block must load its own copy of the input
 * matrices. Interestingly, the Hopper architecture introduces support
 * for thread block clusters, which allow thread blocks within a
 * cluster toaccess each other's shared memory segments. Let's defer
 * discussion of thread block clusters for later and instead, direct
 * our attention to thread coarsening. Although having different thread
 * blocks load the same input tiles is redundant, we pay this price in
 * exchange for the ability to process two output tiles in parallel
 * with different thread blocks. However, if these thread blocks end up
 * being serialized by hardware, we pay the price for nothing. In such
 * situations, it is better to have a single thread block process two
 * output tiles, whereby each thread in the block processes two output
 * elements. This way, the "coarsened" thread block load the input
 * tiles once and reuse them for two output tiles.
 * 
 * We verify that this indeed happens using nsight compute for kernel 2 - mio stalls are very high
 * 
 */

// template <const int TILE_WIDTH, const int COARSEN_FACTOR>
// __global__ void kernel3_thread_coarsen_row(float *A, float *B, float *C, int DIM, float alpha, float beta)
// {
//     int tidx = threadIdx.x, tidy = threadIdx.y;
//     int bidx = blockIdx.x, bidy = blockIdx.y;
//     int row = bidy * TILE_WIDTH + tidy;
//     int colStart = bidx * TILE_WIDTH * COARSEN_FACTOR + tidx;
//     int num_phases = DIM / TILE_WIDTH;
//     float sum[COARSEN_FACTOR] = {0.0f};

//     __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

//     for (int phase = 0; phase < num_phases; phase++) {
//         tileA[tidy][tidx] = A[row * DIM + phase * TILE_WIDTH + tidx];
//         if (bidx == 0 && bidy == 0 && phase == 1) {
//             printf("phase %d: tileA[%d][%d] = A[%d][%d] = %f\n", phase, tidy, tidx, row, phase * TILE_WIDTH + tidx, tileA[tidy][tidx]);
//         }
//         if (tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0)
//             printf("\n");

//         for (int c = 0; c < COARSEN_FACTOR; c++) {
//             int col = colStart + c * TILE_WIDTH;
//             tileB[tidy][tidx] = B[(phase * TILE_WIDTH + tidy) * DIM + col];
//             __syncthreads();

//             if (bidx == 0 && bidy == 0 && phase == 1 && c == 0) {
//                 printf("phase %d: c = %d, tileB[%d][%d] = B[%d][%d] = B[%d] = %f\n", phase, c, tidy, tidx, phase * TILE_WIDTH + tidy, col, (phase * TILE_WIDTH + tidy) * DIM + col, tileB[tidy][tidx]);
//             }
//             if (tidx == 0 && tidy == 0)
//                 printf("\n");

//             for (int k = 0; k < TILE_WIDTH; k++) {
//                 sum[c] += tileA[tidy][k] * tileB[k][tidx];
//                 if (bidx == 0 && bidy == 0 && phase == 1 && c == 0) {
//                     // printf("row = %d, col = %d: ", row, col);
//                     // printf("A[%d][%d], B[%d][%d]\n", tidy, k, k, tidx);
//                     printf("row %d col %d = %f * %f\n", row, col, tileA[tidy][k], tileB[k][tidx]);
//                 }
//             }
//             __syncthreads();
//             // if (tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0) {
//             //     printf("\n");
//             // }
//             // if (tidx == 0 && tidy == 0 && bidx == 1 && bidy == 0) {
//             //     printf("phase = %d, c = %d, sum[c] = %f\n", phase, c, sum[c]);
//             // }
//         }
//     }
//     for (int c = 0; c < COARSEN_FACTOR; c++) {
//         int col = colStart + c * TILE_WIDTH;
//         C[row * DIM + col] = sum[c];
//     }
// }