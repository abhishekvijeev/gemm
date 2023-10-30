__global__ void kernel1_naive(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);

__global__ void kernel2_shmem(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);