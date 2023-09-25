#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>
#include <helper.h>

#include <cstdio>
#include <iostream>

#include <parse_cmdline.h>
#include <util.h>
#include <sgemm_runner/runner.h>

int main(int argc, char **argv)
{
    parse_cmdline_options(argc, argv);
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(m, k));
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(k, n));
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> C_expt(cutlass::MatrixCoord(m, n));
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(m, n));
    GpuTimer timer;
    uint64_t seed;
    uint64_t flops;
    double expt_time_s, ref_time_s;
    cublasHandle_t handle;

    flops = 2 * ((uint64_t)m * n * k) + ((uint64_t)m * n);
    // Gaussian random distribution
    float mean = 0.0_hf;
    float stddev = 5.0_hf;

    // Specify the number of bits right of the binary decimal that are permitted
    // to be non-zero. A value of "0" here truncates random values to integers
    int bits_less_than_one = 1;

    seed = time(NULL);
    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    seed = time(NULL);
    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    seed = time(NULL);
    cutlass::reference::device::TensorFillRandomGaussian(
        C_expt.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::device_memory::copy_device_to_device(
        C_reference.device_data(), 
        C_expt.device_data(), 
        C_expt.capacity());
    C_reference.sync_host();

    A.sync_host();
    B.sync_host();
    C_expt.sync_host();

    // Discard first iteration
    run_sgemm_naive(A.device_data(), B.device_data(), C_expt.device_data(), m, n, k, ALPHA, BETA);
    C_expt.sync_host();

    timer.start();
    for (int i = 0; i < ITERATIONS; i++) {
        run_sgemm_naive(A.device_data(), B.device_data(), C_expt.device_data(), m, n, k, ALPHA, BETA);
    }
    timer.stop();
    expt_time_s = timer.elapsed_millis() / 1000;
    printf("Expt: %.2f GFlops\n", (ITERATIONS * flops * 1e-9) / expt_time_s);

    CUBLAS_CHECK(cublasCreate_v2(&handle));
    CUBLAS_CHECK(cublasSgemm_v2(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m, n, k,
                    &ALPHA,
                    A.device_data(), k,
                    B.device_data(), n,
                    &BETA,
                    C_reference.device_data(), n));

    timer.start();
    for (int i = 0; i < ITERATIONS; i++) {
        CUBLAS_CHECK(cublasSgemm_v2(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m, n, k,
                    &ALPHA,
                    A.device_data(), k,
                    B.device_data(), n,
                    &BETA,
                    C_reference.device_data(), n));
    }
    timer.stop();
    ref_time_s = timer.elapsed_millis() / 1000.0;
    C_reference.sync_host();
    printf("Ref: %.2f GFlops\n", (ITERATIONS * flops * 1e-9) / ref_time_s);

    // Compare reference to computed results.
    if (!cutlass::reference::host::TensorEquals(
        C_reference.host_view(), 
        C_expt.host_view())) {
        std::cout << "ERROR: Results are incorrect!" << std::endl;
        // std::cout << "Experiment results:"  << std::endl << std::endl;
        // std::cout << C_expt.host_view() << std::endl << std::endl;
        // std::cout << "Reference results:"  << std::endl << std::endl;
        // std::cout << C_reference.host_view() << std::endl << std::endl;       
    }

    cublasDestroy_v2(handle);
}