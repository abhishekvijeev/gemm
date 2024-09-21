#include <debug.cuh>

#include <kernels/kernel1_naive.cuh>
#include <kernels/kernel2_shmem.cuh>
#include <kernels/kernel3_thread_coarsen_v1.cuh>
#include <kernels/kernel4_thread_coarsen_v2.cuh>
#include <kernels/kernel5_vectorize.cuh>
#include <kernels/kernel6_cute.cuh>