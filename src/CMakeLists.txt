cmake_minimum_required(VERSION 3.10)  # 需≥3.10
project(my_cuda_app LANGUAGES CUDA CXX)  # 启用CUDA和C++支持[1,2,7](@ref)

set(CHATTY_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CHATTY_CUDA_ENABLED)
    message("CHATTY_CUDA_ENABLED")
    target_include_directories(my_cuda_app PRIVATE ${CHATTY_ROOT_DIR}/include)
    add_subdirectory(executor/cuda/)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:chatty_cuda>")
endif()

if(CHATTY_x86_ENABLED)
    message("CHATTY_x86_ENABLED")
endif()

# 可选：设置CUDA标准版本（如C++17）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CHATTY_ROOT_DIR}/bin)

# Add src files
file(GLOB_RECURSE SRC_LIST ${CHATTY_ROOT_DIR}/src/framework/*.c*)
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
file(GLOB SRC_LIST ${CHATTY_ROOT_DIR}/src/executor/*.c*)
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})

# Add libs
if(CHATTY_STATIC_LIB)
    add_library(chatty STATIC ${TOTAL_SRC_LIST} ${TARGET_OBJECTS})
else()
    add_library(chatty SHARED ${TOTAL_SRC_LIST} ${TARGET_OBJECTS})
endif()

target_link_libraries(cuda_exe PRIVATE cudart cublas)
