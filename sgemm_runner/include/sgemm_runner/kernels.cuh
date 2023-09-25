#include <cutlass/numeric_types.h>

__global__ void kernel1_naive(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);