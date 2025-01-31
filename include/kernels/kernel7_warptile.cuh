template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K, const int WARP_TILE_M, const int WARP_TILE_N, const int THREAD_TILE_M, const int THREAD_TILE_N>
__global__ void kernel7_warptile(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;

    /*  
        Thread block tile:

                  <----------------- BLOCK_TILE_N --------------->

          ↑       +-----------+-----------+-----------+----------+
          |       |WARP_IDX=0 |WARP_IDX=1 |WARP_IDX=2 |WARP_IDX=3|
          |       |WARP_X=0   |WARP_X=1   |WARP_X=2   |WARP_X=3  |
          |       |WARP_Y=0   |WARP_Y=0   |WARP_Y=0   |WARP_Y=0  |
    BLOCK_TILE_M  +-----------+-----------+-----------+----------+
          |       |WARP_IDX=4 |WARP_IDX=5 |WARP_IDX=6 |WARP_IDX=7|
          |       |WARP_X=0   |WARP_X=1   |WARP_X=2   |WARP_X=3  |
          |       |WARP_Y=1   |WARP_Y=1   |WARP_Y=1   |WARP_Y=1  |
          ↓       +-----------+-----------+-----------+----------+
    */
    constexpr int NUM_WARPS_X = BLOCK_TILE_N / WARP_TILE_N;
    constexpr int NUM_WARPS_Y = BLOCK_TILE_M / WARP_TILE_M;

    const int THREAD_IDX = tidy * blockDim.x + tidx;
    const int WARP_IDX = THREAD_IDX / 32;
    const int WARP_X = WARP_IDX % NUM_WARPS_X;
    const int WARP_Y = WARP_IDX / NUM_WARPS_X;

    /*
        Warp tile:

                            WARP_TILE_N
                      <-- matrix elements -->
                
           ↑          +---------------------+           ↑
           |          |                     |           |
           |          |                     |           |
           |          |                     |           |
      WARP_TILE_M     |                     |       WARP_DIM_Y
    matrix elements   |                     |        threads
           |          |                     |           |
           |          |                     |           |
           |          |                     |           |
           ↓          +---------------------+           ↓

                      <----- WARP_DIM_X ---->
                              threads
    */
    constexpr int WARP_DIM_X = WARP_TILE_N / THREAD_TILE_N;
    constexpr int WARP_DIM_Y = WARP_TILE_M / THREAD_TILE_M;
    const int THREAD_X_IN_WARP = (THREAD_IDX % 32) % WARP_DIM_X;
    const int THREAD_Y_IN_WARP = (THREAD_IDX % 32) / WARP_DIM_X;

    // Each thread could be responsible for the computation of multiple
    // output tiles depending on the warp tile and thread tile sizes
    // 
    // For example, let WARP_TILE_M = 64, WARP_TILE_N = 32,
    // THREAD_TILE_M = 8 and THREAD_TILE_N = 8
    //      => WARP_DIM_X = 32 / 8 = 4
    //      => WARP_DIM_Y = 64 / 8 = 8
    //
    // Along the 'M' dimension, we have 64 matrix elements and 8
    // threads, each of which is responsible for 8 elemets. Therefore, 
    // the number of tiles that an individual thread is responsible
    // along the 'M' dimension is: 64 / (8 * 8) = 1
    // 
    // Similarly, along the 'N' dimension, we have 32 matrix elements
    // and 4 threads, each of which is responsible for 8 elements.
    // Therefore, the number of tiles that an individual thread is
    // responsible for along the 'N' dimension is: 32 / (4 * 8) = 1
    constexpr int NUM_THREAD_TILES_PER_WARP_TILE_M = WARP_TILE_M / (WARP_DIM_Y * THREAD_TILE_M);
    constexpr int NUM_THREAD_TILES_PER_WARP_TILE_N = WARP_TILE_N / (WARP_DIM_X * THREAD_TILE_N);

    float a_reg[NUM_THREAD_TILES_PER_WARP_TILE_M][THREAD_TILE_M] = {0};
    float b_reg[NUM_THREAD_TILES_PER_WARP_TILE_N][THREAD_TILE_N] = {0};
    float c_reg[NUM_THREAD_TILES_PER_WARP_TILE_M][NUM_THREAD_TILES_PER_WARP_TILE_N][THREAD_TILE_M][THREAD_TILE_N] = {0};

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
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int cRow = (cRowStart + tidy * THREAD_TILE_M + i);
            int cCol = (cColStart + tidx * THREAD_TILE_N + j);
            C[cRow * DIM + cCol] = sum[i][j];
        }
    }

}
