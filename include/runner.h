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
    float beta
);