#include <runner.h>

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/host_tensor.h>

// Host stubs for each CUDA kernel
#include <iostream>
void run_sgemm_naive(int M, int N, int K, float alpha, float beta)
{
    std::cout <<"do nothing\n";
}