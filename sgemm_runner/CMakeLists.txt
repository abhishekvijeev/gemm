set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

add_library(sgemm_runner SHARED src/runner.cu)
set_target_properties(sgemm_runner PROPERTIES CUDA_ARCHITECTURES "80")
target_include_directories(sgemm_runner PUBLIC
    ./include/sgemm_runner
    ../cutlass/include
    ../cutlass/tools/util/include
    ../cutlass/tools/library/include
    ../cutlass/examples/common
)

# -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_80,code=[sm_80,compute_80]

enable_language(CUDA)
