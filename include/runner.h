void run_sgemm_naive(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
);

void run_sgemm_shmem(
    float *A,
    float *B,
    float *C,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    size_t shmem_per_sm
);