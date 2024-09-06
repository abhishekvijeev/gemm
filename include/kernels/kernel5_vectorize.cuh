template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K, const int THREAD_TILE_M, const int THREAD_TILE_N>
__global__ void kernel5_vectorize(float *A, float *B, float *C, int DIM, float alpha, float beta)
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
            int ATileRow = (tidy * THREAD_TILE_M) + i;
            int ATileCol = (tidx * THREAD_TILE_N);
            int ARow = (cRowStart + tidy * THREAD_TILE_M + i);
            int ACol = (phase * BLOCK_TILE_K + tidx * THREAD_TILE_N);


            int BTileRow = (tidy * THREAD_TILE_M) + i;
            int BTileCol = (tidx * THREAD_TILE_N);
            int BRow = (phase * BLOCK_TILE_K + tidy * THREAD_TILE_N + i);
            int BCol = (cColStart + tidx * THREAD_TILE_M);

            float4 AVec = *reinterpret_cast<float4 *>(&A[ARow * DIM + ACol]);
            // if (debug_thread()) {
            //     printf("AVec.x: %f\n", AVec.x);
            //     printf("AVec.y: %f\n", AVec.y);
            //     printf("AVec.z: %f\n", AVec.z);
            //     printf("AVec.w: %f\n", AVec.w);   
            // }
            ATile[ATileRow][ATileCol + 0] = AVec.x;
            ATile[ATileRow][ATileCol + 1] = AVec.y;
            ATile[ATileRow][ATileCol + 2] = AVec.z;
            ATile[ATileRow][ATileCol + 3] = AVec.w;

            float4 BVec = *reinterpret_cast<float4 *>(&B[BRow * DIM + BCol]);
            BTile[BTileRow][BTileCol + 0] = BVec.x;
            BTile[BTileRow][BTileCol + 1] = BVec.y;
            BTile[BTileRow][BTileCol + 2] = BVec.z;
            BTile[BTileRow][BTileCol + 3] = BVec.w;

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
        int cRow = (cRowStart + tidy * THREAD_TILE_M + i);
        int cCol = (cColStart + tidx * THREAD_TILE_N);
        float4 CVec;
        CVec.x = sum[i][0];
        CVec.y = sum[i][1];
        CVec.z = sum[i][2];
        CVec.w = sum[i][3];
        *reinterpret_cast<float4 *>(&C[cRow * DIM + cCol]) = CVec;
    }

}