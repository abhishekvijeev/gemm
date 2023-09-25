#include <kernels.cuh>

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
