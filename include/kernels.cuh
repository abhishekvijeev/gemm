__global__ void kernel1_naive(float *A, float *B, float *C, int DIM, float alpha, float beta);

__global__ void kernel2_shmem(float *A, float *B, float *C, int DIM, float alpha, float beta, size_t tile_dim_y, size_t tile_dim_x);