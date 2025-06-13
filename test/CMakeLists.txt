cmake_minimum_required(VERSION 3.10)

if(TEST_CUDA_ENABLED)
    add_subdirectory(test_cuda)
endif()

if(TEST_x86_ENABLED)
    add_subdirectory(test_x86)
endif()
