void run_sgemm_naive(
    float *A,
    float *B,
    float *C,
    int DIM,
    float alpha,
    float beta
);

void run_sgemm_shmem(
    float *A,
    float *B,
    float *C,
    int DIM,
    float alpha,
    float beta,
    size_t max_shmem_per_block
);

void run_gemm_thread_coarsen_v1(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
);

void run_gemm_thread_coarsen_v2(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
);

void run_gemm_vectorize(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
);

void run_gemm_cute(
    float *A,
    float *B,
    float *C, 
    int DIM,
    float alpha,
    float beta
);