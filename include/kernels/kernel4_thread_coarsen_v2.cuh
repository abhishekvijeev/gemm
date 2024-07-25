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
 * Each threadblock computes the result for a BLOCK_TILE_M x BLOCK_TILE_N
 * tile of C.
 * 
 * Each thread computes the result for a THREAD_TILE_M x THREAD_TILE_N
 * tile of C.
 */
template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K, const int THREAD_TILE_M, const int THREAD_TILE_N>
__global__ void kernel4_thread_coarsen_v2(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int cRowStart = bidy * blockDim.y * THREAD_TILE_M  + tidy;
    int cColStart = bidx * blockDim.x * THREAD_TILE_N + tidx;

    // if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 0) {
    //     printf("rowStart: %d\n", rowStart);
    //     printf("colStart: %d\n", colStart);
    // }

    int num_phases = DIM / BLOCK_TILE_K;

    __shared__ float blockTileA[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float blockTileB[BLOCK_TILE_K][BLOCK_TILE_N];

    float sum[THREAD_TILE_M][THREAD_TILE_N];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            sum[i][j] = 0;
        }
    }

    for (int phase = 0; phase < num_phases; phase++) {
        if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 1) {
            printf("phase %d\n", phase);
        }
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {

                if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 1) {
                    printf("blockTileA[%d][%d] = A[%d][%d]\n", (tidy * THREAD_TILE_M) + i, (tidx * THREAD_TILE_N) + j, cRowStart + i, phase * BLOCK_TILE_K + tidx + j);
                }

                // blockTileA[tidy + i][tidx + j] = A[rowStart + i][];
            }
        }
        __syncthreads();
        if (bidx == 0 && tidx == 0 && bidy == 1 && tidy == 0) {
            printf("\n");;
        }
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            C[(cRowStart + i) * DIM + cColStart + j] = sum[i][j];
        }
    }
}