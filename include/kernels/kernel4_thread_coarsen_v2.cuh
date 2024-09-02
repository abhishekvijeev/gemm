/*
 * Problems with Kernel 3:
 * 
 * 1) Assumes that tile size and thread block size are the same
 * 
 * 2) The elements of C computed by a single thread don't belong to
 *    the same tile - this precludes the possibility of vectorized
 *    loads, which are crucial to achieving hight throughput.
 * 
 * Kernel 4 mitigates the above two problems.
 * 
 * a) Each threadblock computes the result for a BLOCK_TILE_M x BLOCK_TILE_N
 * tile of C.
 * 
 * b) Each thread computes the result for a THREAD_TILE_M x THREAD_TILE_N
 * tile of C.
 */

template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K, const int THREAD_TILE_M, const int THREAD_TILE_N>
__global__ void kernel4_thread_coarsen_v2(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;

    // equivalent to (bidy * blockDim.y * THREAD_TILE_M  + tidy)
    int cRowStart = bidy * BLOCK_TILE_M  + tidy;

    // equivalent to (bidx * blockDim.x * THREAD_TILE_N  + tidx)
    int cColStart = bidx * BLOCK_TILE_N + tidx;

    // if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 0) {
    //     printf("rowStart: %d\n", rowStart);
    //     printf("colStart: %d\n", colStart);
    // }

    int num_phases = DIM / BLOCK_TILE_K;

    __shared__ float ATile[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float BTile[BLOCK_TILE_K][BLOCK_TILE_N];

    float sum[THREAD_TILE_M][THREAD_TILE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            sum[i][j] = 0;
        }
    }

    for (int phase = 0; phase < num_phases; phase++) {
        // if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 0) {
        //     printf("phase %d\n", phase);
        // }
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {

                if (bidx == 0 && tidx == 1 && bidy == 0 && tidy == 0) {
                    printf("ATile[%d][%d] = A[%d][%d] = %.0f\n", (tidy * THREAD_TILE_M) + i, (tidx * THREAD_TILE_N) + j, cRowStart * THREAD_TILE_N + i, phase * BLOCK_TILE_K + tidx * THREAD_TILE_M + j, A[(cRowStart * THREAD_TILE_N + i) * DIM + (phase * BLOCK_TILE_K + tidx * THREAD_TILE_M + j)]);

                //     printf("BTile[%d][%d] = B[%d][%d] = %.0f\n", (tidy * THREAD_TILE_M) + i, (tidx * THREAD_TILE_N) + j, (phase * BLOCK_TILE_K + tidy * THREAD_TILE_N + i), (cColStart * THREAD_TILE_M + j), B[(phase * BLOCK_TILE_K + tidy * THREAD_TILE_N + i) * DIM + (cColStart * THREAD_TILE_M + j)];
                }

                ATile[(tidy * THREAD_TILE_M) + i][(tidx * THREAD_TILE_N) + j] = 
                    A[(cRowStart * THREAD_TILE_M + i) * DIM + (phase * BLOCK_TILE_K + tidx * THREAD_TILE_N + j)];

                BTile[(tidy * THREAD_TILE_M) + i][(tidx * THREAD_TILE_N) + j] = 
                    B[(phase * BLOCK_TILE_K + tidy * THREAD_TILE_N + i) * DIM + (cColStart * THREAD_TILE_M + j)];
            }
        }
        __syncthreads();
        // if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 0) {
        //     printf("\n");;
        // }

        if (bidx == 0 && bidy == 0 && tidx == 1 && tidy == 0) {
            printf("ATile:\n\n");
            for (int i = 0; i < BLOCK_TILE_M; i++) {
                for (int j = 0; j < BLOCK_TILE_K; j++) {
                    printf("%.0f ", ATile[i][j]);
                }
                printf("\n");
            }
            printf("\n\n");

            printf("BTile:\n\n");
            for (int i = 0; i < BLOCK_TILE_K; i++) {
                for (int j = 0; j < BLOCK_TILE_N; j++) {
                    printf("%.0f ", BTile[i][j]);
                }
                printf("\n");
            }
            printf("\n\n");
        }

        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                sum[i][j] +=
                    ATile[(tidy * THREAD_TILE_M) + i][(tidx * THREAD_TILE_N) + j] * 
                    BTile[(tidy * THREAD_TILE_M) + i][(tidx * THREAD_TILE_N) + j];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j++) {
            C[(cRowStart + i) * DIM + (cColStart + j)] = sum[i][j];
        }
    }

}