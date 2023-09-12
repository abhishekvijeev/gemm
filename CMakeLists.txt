cmake_minimum_required(VERSION 3.10)

project(SGEMM)

add_subdirectory(sgemm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

enable_language(CUDA)

add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80")
target_include_directories(main PUBLIC
    $PROJECT_SOURCE_DIR/sgemm/include
    $PROJECT_SOURCE_DIR/cutlass/include
    $PROJECT_SOURCE_DIR/cutlass/tools/util/include
    $PROJECT_SOURCE_DIR/cutlass/tools/library/include
    $PROJECT_SOURCE_DIR/cutlass/examples/common
)
target_link_libraries(main PUBLIC sgemm)
target_compile_options(main PUBLIC -pg -G)