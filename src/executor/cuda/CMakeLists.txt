find_package(CUDA)
set(CMAKE_CUDA_VERBOSE_BUILD ON)

# Find src files
file(GLOB_RECURSE CUDA_SRC "*.cu*")
file(GLOB_RECURSE CXX_SRC "*.cc")

# Add libs
cuda_add_library(chatty_cuda ${CUDA_SRC} ${CXX_SRC})
CUDA_ADD_CUBLAS_TO_TARGET(chatty_cuda)
