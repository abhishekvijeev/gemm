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

#define DEBUG_BIDX 0
#define DEBUG_BIDY 0
#define DEBUG_TIDX 1
#define DEBUG_TIDY 0

__device__ bool debug_thread()
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;

    return (bidx == DEBUG_BIDX && bidy == DEBUG_BIDY && tidx == DEBUG_TIDX && tidy == DEBUG_TIDY);
}

template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K, const int THREAD_TILE_M, const int THREAD_TILE_N>
__global__ void kernel4_thread_coarsen_v2(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;

    // equivalent to (bidy * blockDim.y * THREAD_TILE_M)
    int cRowStart = bidy * BLOCK_TILE_M;

    // equivalent to (bidx * blockDim.x * THREAD_TILE_N)
    int cColStart = bidx * BLOCK_TILE_N;

    // if (debug_thread()) {
    //     printf("rowStart: %d\n", cRowStart);
    //     printf("colStart: %d\n", cColStart);
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
        // if (debug_thread()) {
        //     printf("phase %d\n", phase);
        // }
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {

                int ATileRow = (tidy * THREAD_TILE_M) + i;
                int ATileCol = (tidx * THREAD_TILE_N) + j;
                int ARow = (cRowStart + tidy * THREAD_TILE_M + i);
                int ACol = (phase * BLOCK_TILE_K + tidx * THREAD_TILE_N + j);

                int BTileRow = (tidy * THREAD_TILE_M) + i;
                int BTileCol = (tidx * THREAD_TILE_N) + j;
                int BRow = (phase * BLOCK_TILE_K + tidy * THREAD_TILE_N + i);
                int BCol = (cColStart + tidx * THREAD_TILE_M + j);

                if (tidx * THREAD_TILE_N < BLOCK_TILE_K)
                    ATile[ATileRow][ATileCol] = A[ARow * DIM + ACol];
                if (tidy * THREAD_TILE_M < BLOCK_TILE_K)
                    BTile[BTileRow][BTileCol] = B[BRow * DIM + BCol];

                // if (debug_thread()) {
                //     printf("ATile[%d][%d] = A[%d][%d] = %.0f\n", ATileRow, ATileCol, ARow, ACol, ATile[ATileRow][ATileCol]);
                //     printf("BTile[%d][%d] = B[%d][%d] = %.0f\n", BTileRow, BTileCol, BRow, BCol, BTile[BTileRow][BTileCol]);
                // }
            }
        }
        __syncthreads();
        // if (debug_thread()) {
        //     printf("\n");;
        // }

        // if (debug_thread()) {
        //     printf("ATile:\n\n");
        //     for (int i = 0; i < BLOCK_TILE_M; i++) {
        //         for (int j = 0; j < BLOCK_TILE_K; j++) {
        //             printf("%.0f ", ATile[i][j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n\n");

        //     printf("BTile:\n\n");
        //     for (int i = 0; i < BLOCK_TILE_K; i++) {
        //         for (int j = 0; j < BLOCK_TILE_N; j++) {
        //             printf("%.0f ", BTile[i][j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n\n");
        // }

        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                #pragma unroll
                for (int k = 0; k < BLOCK_TILE_K; k++) {
                    sum[i][j] +=
                        ATile[(tidy * THREAD_TILE_M) + i][k] * 
                        BTile[k][(tidx * THREAD_TILE_N) + j];
                    // if (debug_thread()) {
                    //     printf("ATile[%d][%d] * BTile[%d][%d] = %.0f * %.0f\n", (tidy * THREAD_TILE_M) + i, k, k,
                    //         (tidx * THREAD_TILE_N) + j, ATile[(tidy * THREAD_TILE_M) + i][k], BTile[k][(tidx * THREAD_TILE_N) + j]);
                    // }
                }
            }
        }
        // if (debug_thread()) {
        //     printf("\n");
        // }
        __syncthreads();
    }

    // if (debug_thread()) {
    //     printf("Result Computed:\n\n");
    //     for (int i = 0; i < THREAD_TILE_M; i++) {
    //         for (int j = 0; j < THREAD_TILE_N; j++) {
    //             printf("%.0f ", sum[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");
    // }

    // printf("%d, %d\n", cRowStart, cColStart);
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int cRow = (cRowStart + tidy * THREAD_TILE_M + i);
            int cCol = (cColStart + tidx * THREAD_TILE_N + j);
            C[cRow * DIM + cCol] = sum[i][j];
            // if (debug_thread()) {
            //     printf("C[%d][%d] = %.0f\n", cRow, cCol, sum[i][j]);
            // }
        }
    }

}