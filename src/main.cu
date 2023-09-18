#include <cutlass/gemm/device/gemm.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <cstdio>
#include <iostream>

#include <parse_cmdline.h>
#include <sgemm_runner/runner.h>

int main(int argc, char **argv)
{
    parse_cmdline_options(argc, argv);
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));

    uint64_t seed = 2080;

    // Gaussian random distribution
    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stddev = 5.0_hf;

    // Specify the number of bits right of the binary decimal that are permitted
    // to be non-zero. A value of "0" here truncates random values to integers
    int bits_less_than_one = 2;

    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(),
        seed * 2019,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
        C_cutlass.device_view(),
        seed * 1993,
        mean,
        stddev,
        bits_less_than_one
    );

    A.sync_host();
    std::cout << A.host_view() << std::endl;

    // run_sgemm_naive(M, N, K, ALPHA, BETA);

    // Compute the reference result using the host-side GEMM reference implementation.
    // cutlass::reference::host::Gemm<
    //     cutlass::half_t,                           // ElementA
    //     cutlass::layout::ColumnMajor,              // LayoutA
    //     cutlass::half_t,                           // ElementB
    //     cutlass::layout::ColumnMajor,              // LayoutB
    //     cutlass::half_t,                           // ElementOutput
    //     cutlass::layout::ColumnMajor,              // LayoutOutput
    //     cutlass::half_t,
    //     cutlass::half_t
    // > gemm_ref;

    // gemm_ref(
    //     {M, N, K},                      // problem size (type: cutlass::gemm::GemmCoord)
    //     alpha,                          // alpha        (type: cutlass::half_t)
    //     A.host_ref(),                   // A            (type: TensorRef<half_t, ColumnMajor>)
    //     B.host_ref(),                   // B            (type: TensorRef<half_t, ColumnMajor>)
    //     beta,                           // beta         (type: cutlass::half_t)
    //     C_reference.host_ref()          // C            (type: TensorRef<half_t, ColumnMajor>)
    // );

    // // Compare reference to computed results.
    // if (!cutlass::reference::host::TensorEquals(
    //     C_reference.host_view(), 
    //     C_cutlass.host_view())) {

    //     char const *filename = "errors_01_cutlass_utilities.csv";

    //     std::cerr << "Error - CUTLASS GEMM kernel differs from reference. Wrote computed and reference results to '" << filename << "'" << std::endl;

    //     //
    //     // On error, print C_cutlass and C_reference to std::cerr.
    //     //
    //     // Note, these are matrices of half-precision elements stored in host memory as
    //     // arrays of type cutlass::half_t.
    //     //

    //     std::ofstream file(filename);

    //     // Result of CUTLASS GEMM kernel
    //     file << "\n\nCUTLASS =\n" << C_cutlass.host_view() << std::endl;

    //     // Result of reference computation
    //     file << "\n\nReference =\n" << C_reference.host_view() << std::endl;

    //     // Return error code.
    //     return cudaErrorUnknown;
    // }
}