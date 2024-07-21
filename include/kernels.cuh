#include <cstdio>

__global__ void kernel1_naive(float *A, float *B, float *C, int DIM, float alpha, float beta);

__global__ void kernel2_shmem(float *A, float *B, float *C, int DIM, float alpha, float beta, size_t tile_dim_y, size_t tile_dim_x);

template <const int TILE_WIDTH, const int COARSEN_FACTOR>
__global__ void kernel3_thread_coarsen_row(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int row = bidy * TILE_WIDTH + tidy;
    int colStart = bidx * TILE_WIDTH * COARSEN_FACTOR + tidx;
    int num_phases = DIM / TILE_WIDTH;
    float sum[COARSEN_FACTOR] = {0.0f};

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    for (int phase = 0; phase < num_phases; phase++) {
        tileA[tidy][tidx] = A[row * DIM + phase * TILE_WIDTH + tidx];
        if (bidx == 0 && bidy == 0 && phase == 1) {
            printf("phase %d: tileA[%d][%d] = A[%d][%d] = %f\n", phase, tidy, tidx, row, phase * TILE_WIDTH + tidx, tileA[tidy][tidx]);
        }
        if (tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0)
            printf("\n");

        for (int c = 0; c < COARSEN_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;
            tileB[tidy][tidx] = B[(phase * TILE_WIDTH + tidy) * DIM + col];
            __syncthreads();

            if (bidx == 0 && bidy == 0 && phase == 1 && c == 0) {
                printf("phase %d: c = %d, tileB[%d][%d] = B[%d][%d] = B[%d] = %f\n", phase, c, tidy, tidx, phase * TILE_WIDTH + tidy, col, (phase * TILE_WIDTH + tidy) * DIM + col, tileB[tidy][tidx]);
            }
            if (tidx == 0 && tidy == 0)
                printf("\n");

            for (int k = 0; k < TILE_WIDTH; k++) {
                sum[c] += tileA[tidy][k] * tileB[k][tidx];
                if (bidx == 0 && bidy == 0 && phase == 1 && c == 0) {
                    // printf("row = %d, col = %d: ", row, col);
                    // printf("A[%d][%d], B[%d][%d]\n", tidy, k, k, tidx);
                    printf("row %d col %d = %f * %f\n", row, col, tileA[tidy][k], tileB[k][tidx]);
                }
            }
            __syncthreads();
            // if (tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0) {
            //     printf("\n");
            // }
            // if (tidx == 0 && tidy == 0 && bidx == 1 && bidy == 0) {
            //     printf("phase = %d, c = %d, sum[c] = %f\n", phase, c, sum[c]);
            // }
        }
    }
    for (int c = 0; c < COARSEN_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        C[row * DIM + col] = sum[c];
    }
}