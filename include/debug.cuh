#define DEBUG_BIDX 0
#define DEBUG_BIDY 0
#define DEBUG_TIDX 0
#define DEBUG_TIDY 0

__device__ bool debug_thread()
{
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bidx = blockIdx.x, bidy = blockIdx.y;

    return (bidx == DEBUG_BIDX && bidy == DEBUG_BIDY && tidx == DEBUG_TIDX && tidy == DEBUG_TIDY);
}