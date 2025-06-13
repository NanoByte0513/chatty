cmake_minimum_required(VERSION 3.10)
project(test_cuda)

set(CHATTY_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CHATTY_ROOT_DIR}/bin)

# Add exe
add_executable(test_cuda ${CHATTY_ROOT_DIR}/test/test_cuda/test_cuda.cc)

target_link_directories(test_cuda PRIVATE ${CHATTY_ROOT_DIR}/bin)
target_link_libraries(test_cuda PRIVATE chatty)
