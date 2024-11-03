#include <cute/tensor.hpp>

template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K>
__global__ void kernel6_cute(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    using namespace cute;

    __shared__ float shared_a[BLOCK_TILE_M * BLOCK_TILE_K];
    __shared__ float shared_b[BLOCK_TILE_K * BLOCK_TILE_N];

    // Create global memory tensors
    Tensor gmem_a = make_tensor(make_gmem_ptr(A), make_shape(DIM, DIM), make_stride(DIM, 1));
    Tensor gmem_b = make_tensor(make_gmem_ptr(B), make_shape(DIM, DIM), make_stride(DIM, 1));
    Tensor gmem_c = make_tensor(make_gmem_ptr(C), make_shape(DIM, DIM), make_stride(DIM, 1));

    // Define tile sizes
    auto tile_shape_a = make_shape(BLOCK_TILE_M, BLOCK_TILE_K);
    auto tile_shape_b = make_shape(BLOCK_TILE_K, BLOCK_TILE_N);
    auto tile_shape_c = make_shape(BLOCK_TILE_M, BLOCK_TILE_N);

    // Partition global memory into tiles
    auto gA = local_tile(gmem_a, tile_shape_a, make_coord(blockIdx.y, _));
    auto gB = local_tile(gmem_b, tile_shape_b, make_coord(_, blockIdx.x));
    auto gC = local_tile(gmem_c, tile_shape_c, make_coord(blockIdx.y, blockIdx.x));
    auto num_tiles = size<2>(gA);

    // Create shared memory tensors
    Tensor sA = make_tensor(make_smem_ptr(shared_a), tile_shape_a, make_stride(BLOCK_TILE_K, 1));
    Tensor sB = make_tensor(make_smem_ptr(shared_b), tile_shape_b, make_stride(BLOCK_TILE_N, 1));

    // Partition the tile's elements across all threads in the threadblock
    auto tA = make_layout(make_shape(1, 1));
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);

    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 1) {
        printf("Block x = %d, y = %d\n\n", blockIdx.x, blockIdx.y);
        print_tensor(gC); printf("\n\n");
    }

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++)
    {
        // if (thread0()) {
        //     printf("Tile A%d Before Copy:\n", tile_idx);
        //     print_tensor(sA); printf("\n");
        // }
        // __syncthreads();

        copy(tAgA(_,_,tile_idx), tAsA);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // if (thread0()) {
        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0) {
        //     printf("Tile A%d:\n", tile_idx);
        //     print_tensor(sA); printf("\n");
        // }
        // __syncthreads();
    }

    

    // (4,4):(8,1)
    // _ (4,4,2):(8,1,4)
    // if (thread0()) {
    //     print_tensor(gA); printf("\n\n\n");
    //     print_tensor(gB); printf("\n\n\n");
    //     printf("%d\n", num_tiles);
    //     // print(size(gA)); printf("\n");
    // }

    // if (thread0()) {
    // if (blockIdx.x == 0 && blockIdx.y == 1) {
    //     if (threadIdx.x == 0 && threadIdx.y == 0) {
    //         print_tensor(gmem_a); printf("\n");
    //     }
    //     // print_tensor(gmem_a(_, 2)); printf("\n");
    //     auto a_tiler = make_shape(2, 2);
    //     auto tiled_a = zipped_divide(gmem_a, a_tiler);

    //     // Equivalent to:
    //     // my_partition = local_tile(gmem_a, a_tiler, make_coord(blockIdx.x, blockIdx.y));
    //     auto my_partition = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y)); 
    //     print_tensor(my_partition); printf("\n");

    //     auto my_partition = local_tile(gmem_a, a_tiler, make_coord(blockIdx.x, blockIdx.y));
    //     if (threadIdx.x == 0 && threadIdx.y == 0) {
    //         print_tensor(my_partition); printf("\n");
    //     }
    // }

    // Tensor sA = make_tensor(make_smem_ptr(shared_a), make_shape(BLOCK_TILE_M, BLOCK_TILE_K), make_stride(BLOCK_TILE_K, 1));
    // Tensor sB = make_tensor(make_smem_ptr(shared_b), make_shape(BLOCK_TILE_K, BLOCK_TILE_N), make_stride(BLOCK_TILE_N, 1));

    // if (thread0()) {
    //     print_tensor(gA); printf("\n");
    //     print_tensor(sA); printf("\n");
    // }
    // __syncthreads();

    // /*
    //  * The kernel now has:
    //  *
    //  * a) tiles of global memory by applying the tiler to the full tensors
    //  * b) tiles of shared memory
    //  * 
    //  * Next, we want to copy one tile of global memory to our tile of shared memory.
    //  * If we partition the two tiles of data across the threads in the CTA, then each thread can copy its own subtensor of data.
    //  */

    // auto tA = make_layout(make_shape(Int<2>{},Int<4>{}));
    // auto tB = make_layout(make_shape(Int<2>{},Int<4>{}));

    // Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    // Tensor tAsA = local_partition(sA, tA, threadIdx.x);

    // Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    // Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    // copy(tAgA(_,_), tAsA);
    // copy(tBgB(_,_), tBsB);

    // cp_async_fence();
    // cp_async_wait<0>();
    // __syncthreads(); 

    // if (thread0()) {
    //     print_tensor(sA); printf("\n");
    // }
    // __syncthreads();
}