#include <kernels.cuh>

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
 */
__global__ void kernel3_thread_coarsen_row(float *A, float *B, float *C, int DIM, float alpha, float beta)
{

}