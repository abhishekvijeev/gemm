template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K, const int WARP_TILE_M, const int WARP_TILE_N, const int THREAD_TILE_M, const int THREAD_TILE_N>
__global__ void kernel7_warptile(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;

    // equivalent to (bidy * blockDim.y * THREAD_TILE_M)
    int cRowStart = bidy * BLOCK_TILE_M;

    // equivalent to (bidx * blockDim.x * THREAD_TILE_N)
    int cColStart = bidx * BLOCK_TILE_N;

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
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j += 4) {
                int ATileRow = (tidy * THREAD_TILE_M) + i;
                int ATileCol = (tidx * THREAD_TILE_N) + j;
                int ARow = (cRowStart + tidy * THREAD_TILE_M + i);
                int ACol = (phase * BLOCK_TILE_K + tidx * THREAD_TILE_N + j);

                int BTileRow = (tidy * THREAD_TILE_M) + i;
                int BTileCol = (tidx * THREAD_TILE_N) + j;
                int BRow = (phase * BLOCK_TILE_K + tidy * THREAD_TILE_N + i);
                int BCol = (cColStart + tidx * THREAD_TILE_M + j);

                if (tidx * THREAD_TILE_N < BLOCK_TILE_K) {
                    float4 AVec = *reinterpret_cast<float4 *>(&A[ARow * DIM + ACol]);
                    ATile[ATileRow][ATileCol + 0] = AVec.x;
                    ATile[ATileRow][ATileCol + 1] = AVec.y;
                    ATile[ATileRow][ATileCol + 2] = AVec.z;
                    ATile[ATileRow][ATileCol + 3] = AVec.w;
                }
                if (tidy * THREAD_TILE_M < BLOCK_TILE_K) {
                    float4 BVec = *reinterpret_cast<float4 *>(&B[BRow * DIM + BCol]);
                    BTile[BTileRow][BTileCol + 0] = BVec.x;
                    BTile[BTileRow][BTileCol + 1] = BVec.y;
                    BTile[BTileRow][BTileCol + 2] = BVec.z;
                    BTile[BTileRow][BTileCol + 3] = BVec.w;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; k++) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    sum[i][j] +=
                        ATile[(tidy * THREAD_TILE_M) + i][k] * 
                        BTile[k][(tidx * THREAD_TILE_N) + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j += 4) {
            int cRow = (cRowStart + tidy * THREAD_TILE_M + i);
            int cCol = (cColStart + tidx * THREAD_TILE_N + j);
            float4 CVec;
            CVec.x = sum[i][j];
            CVec.y = sum[i][j + 1];
            CVec.z = sum[i][j + 2];
            CVec.w = sum[i][j + 3];
            *reinterpret_cast<float4 *>(&C[cRow * DIM + cCol]) = CVec;
        }
    }

}
