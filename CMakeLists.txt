cmake_minimum_required(VERSION 3.10)

project(SGEMM)

find_package(Boost COMPONENTS program_options REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CUBLAS_LIBRARIES -lcublas)

enable_language(CUDA)

add_executable(main
    src/main.cu
    src/parse_cmdline.cu
    src/runner.cu
)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80")
target_include_directories(main PUBLIC
    ./include
    ./cutlass/include
    ./cutlass/tools/util/include
    ./cutlass/tools/library/include
    ./cutlass/examples/common
    ${Boost_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(main PUBLIC ${CUBLAS_LIBRARIES})
target_link_libraries(main PUBLIC ${Boost_LIBRARIES})
# target_compile_options(main PUBLIC -pg -G)