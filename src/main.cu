#include <cublas_v2.h>
#include <cuda_runtime.h>

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

#include <cmdline.h>
#include <runner.h>
#include <util.h>

void run_kernel(
    int kernel_num,
    cudaDeviceProp *prop,
    float *A,
    float *B,
    float *C,
    int dim,
    float alpha,
    float beta
)
{
    size_t max_shmem_per_block = prop->sharedMemPerBlock;
    switch (kernel_num) {
        case 1:
            run_sgemm_naive(A, B, C, dim, alpha, beta);
            break;

        case 2:
            run_sgemm_shmem(A, B, C, dim, alpha, beta, max_shmem_per_block);
            break;

        case 3:
            run_gemm_thread_coarsen_v1(A, B, C, dim, alpha, beta);
            break;
        
        case 4:
            run_gemm_thread_coarsen_v2(A, B, C, dim, alpha, beta);
            break;
        
        case 5:
            run_gemm_vectorize(A, B, C, dim, alpha, beta);
            break;

        case 6:
            run_gemm_cute(A, B, C, dim, alpha, beta);
            break;
        
        case 7:
            run_gemm_warptile(A, B, C, dim, alpha, beta);
            break;

        default:
            printf("Invalid kernel number - [1-7] allowed\n");
            break;
    }
}

int main(int argc, const char **argv)
{
    CommandLineArgs args;
    args.parse(argc, argv);

    int DIM = args.dim;
    int KERNEL = args.kernel;
    float ALPHA = args.alpha;
    float BETA = args.beta;
    int ITERATIONS = args.iterations;

    cutlass::HostTensor<float, cutlass::layout::RowMajor> A(cutlass::MatrixCoord(DIM, DIM));
    cutlass::HostTensor<float, cutlass::layout::RowMajor> B(cutlass::MatrixCoord(DIM, DIM));
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C_expt(cutlass::MatrixCoord(DIM, DIM));
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C_reference(cutlass::MatrixCoord(DIM, DIM));
    GpuTimer timer;
    uint64_t seed;
    uint64_t flops;
    double expt_time_s, ref_time_s;
    cublasHandle_t handle;
    int device;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    flops = 2 * ((uint64_t)DIM * DIM * DIM) + ((uint64_t)DIM * DIM);
    // Gaussian random distribution
    float mean = 2;
    // float stddev = 5.0_hf;
    float stddev = 1;

    // Specify the number of bits right of the binary decimal that are permitted
    // to be non-zero. A value of "0" here truncates random values to integers
    int bits_less_than_one = 0;

    seed = 1;
    // seed = time(NULL);
    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    // seed = time(NULL);
    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    // seed = time(NULL);
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
    run_kernel(KERNEL, &prop, A.device_data(), B.device_data(), C_expt.device_data(), DIM, ALPHA, BETA);
    C_expt.sync_host();

    timer.start();
    for (int i = 0; i < ITERATIONS; i++) {
        run_kernel(KERNEL, &prop, A.device_data(), B.device_data(), C_expt.device_data(), DIM, ALPHA, BETA);
    }
    timer.stop();
    expt_time_s = timer.elapsed_millis() / 1000;
    printf("Expt: %.2f GFlops/s\n", (ITERATIONS * flops * 1e-9) / expt_time_s);

    CUBLAS_CHECK(cublasCreate_v2(&handle));
    CUBLAS_CHECK(cublasSgemm_v2(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    DIM, DIM, DIM,
                    &ALPHA,
                    A.device_data(), DIM,
                    B.device_data(), DIM,
                    &BETA,
                    C_reference.device_data(), DIM));

    timer.start();
    for (int i = 0; i < ITERATIONS; i++) {
        CUBLAS_CHECK(cublasSgemm_v2(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    DIM, DIM, DIM,
                    &ALPHA,
                    A.device_data(), DIM,
                    B.device_data(), DIM,
                    &BETA,
                    C_reference.device_data(), DIM));
    }
    timer.stop();
    ref_time_s = timer.elapsed_millis() / 1000.0;
    C_reference.sync_host();
    printf("Ref: %.2f GFlops/s\n", (ITERATIONS * flops * 1e-9) / ref_time_s);

    // std::cout << "A:"  << std::endl << std::endl;
    // std::cout << A.host_view() << std::endl << std::endl;

    // std::cout << "B:"  << std::endl << std::endl;
    // std::cout << B.host_view() << std::endl << std::endl;

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