cmake_minimum_required(VERSION 3.10)
project(chatty LANGUAGES CUDA CXX)

set(CHATTY_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CHATTY_CUDA_ENABLED)
    message("CHATTY_CUDA_ENABLED")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_subdirectory(executor/cuda/)
    set(TARGET_OBJECTS ${TARGET_OBJECTS} "$<TARGET_OBJECTS:chatty_cuda>")
elseif(CHATTY_x86_ENABLED)
    message("CHATTY_x86_ENABLED")
endif()

# include dir
include_directories(${CHATTY_ROOT_DIR}/include)
include_directories(${CHATTY_ROOT_DIR}/src)

# Set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CHATTY_ROOT_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CHATTY_ROOT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CHATTY_ROOT_DIR}/bin)

# Add src files
file(GLOB_RECURSE SRC_LIST ${CHATTY_ROOT_DIR}/src/framework/*.c*)
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
file(GLOB SRC_LIST ${CHATTY_ROOT_DIR}/src/executor/*.c*)
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})

# Add libs
message("SRC_LIST: ${TOTAL_SRC_LIST}")
if(CHATTY_STATIC_LIB)
    add_library(chatty STATIC ${TOTAL_SRC_LIST} ${TARGET_OBJECTS})
else()
    add_library(chatty SHARED ${TOTAL_SRC_LIST} ${TARGET_OBJECTS})
    # Hide all symbols in shared lib, explicitly mark the symbols to be exported in the code
    target_compile_options(chatty PRIVATE 
        -fvisibility=hidden -fvisibility-inlines-hidden
    )
endif()

# target_link_libraries(cuda_exe PRIVATE cudart cublas)
