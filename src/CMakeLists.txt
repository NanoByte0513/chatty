cmake_minimum_required(VERSION 3.10)  # 需≥3.10
project(my_cuda_app LANGUAGES CUDA CXX)  # 启用CUDA和C++支持[1,2,7](@ref)

set(CHATTY_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CHATTY_CUDA_ENABLED)
    message("CHATTY_CUDA_ENABLED")
endif()

if(CHATTY_PRT_MSG)
    message("CHATTY_PRT_MSG")
endif()

# 可选：设置CUDA标准版本（如C++17）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CHATTY_ROOT_DIR}/bin)

# Add exe
add_executable(cuda_exe ${CHATTY_ROOT_DIR}/src/main.cu)

# 可选：链接CUDA运行时库（通常自动链接）
target_link_libraries(cuda_exe PRIVATE cudart cublas)