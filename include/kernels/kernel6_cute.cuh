#include <cute/tensor.hpp>

template <const int BLOCK_TILE_M, const int BLOCK_TILE_N, const int BLOCK_TILE_K>
__global__ void kernel6_cute(float *A, float *B, float *C, int DIM, float alpha, float beta)
{
    using namespace cute;

    Tensor gmem_a = make_tensor(make_gmem_ptr(A), make_shape(DIM, DIM), make_stride(DIM, 1));
    Tensor gmem_b = make_tensor(make_gmem_ptr(B), make_shape(DIM, DIM), make_stride(DIM, 1));
    Tensor gmem_c = make_tensor(make_gmem_ptr(C), make_shape(DIM, DIM), make_stride(DIM, 1));

    auto cta_tiler_a = make_shape(BLOCK_TILE_M, BLOCK_TILE_K);
    auto cta_tiler_b = make_shape(BLOCK_TILE_K, BLOCK_TILE_N);
    auto cta_tiler_c = make_shape(BLOCK_TILE_M, BLOCK_TILE_N);

    auto local_tile_a = local_tile(gmem_a, cta_tiler_a, make_coord(blockIdx.x, blockIdx.y));
    auto local_tile_b = local_tile(gmem_b, cta_tiler_b, make_coord(blockIdx.x, blockIdx.y));
    auto local_tile_c = local_tile(gmem_c, cta_tiler_c, make_coord(blockIdx.x, blockIdx.y));

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

    __shared__ float shared_a[BLOCK_TILE_M * BLOCK_TILE_K];
    __shared__ float shared_b[BLOCK_TILE_K * BLOCK_TILE_N];

    Tensor smem_a = make_tensor(make_smem_ptr(shared_a), make_shape(BLOCK_TILE_M, BLOCK_TILE_K), make_stride(BLOCK_TILE_K, 1));
    Tensor smem_b = make_tensor(make_smem_ptr(shared_b), make_shape(BLOCK_TILE_K, BLOCK_TILE_N), make_stride(BLOCK_TILE_N, 1));
}