#include <sgemm.h>

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/host_tensor.h>

void sgemm(int M, int N, int K, float alpha, float beta)
{
    cutlass::HostTensor<float, cutlass::layout::RowMajor> A(cutlass::MatrixCoord(M, K));
    float x;

    // A.copy_out_device_to_host(&x);
    // printf("%lu\n", A.size());
    // printf("%lu\n", A.capacity());
    // printf("%f\n", A.host_data()[0]);
    // printf("%f\n", x);
}