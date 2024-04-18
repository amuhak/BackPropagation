cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(BackPropagation CUDA)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_MODE ON)
    message(STATUS "Debug mode")
else ()
    set(DEBUG_MODE OFF)
    message(STATUS "Release mode")
endif ()

if (DEBUG_MODE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -G -lineinfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -g")
    add_compile_definitions(DEBUG)
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
endif ()

set(CMAKE_CUDA_STANDARD 17)

add_executable(BackPropagation
        main.cu
        Matrix.cpp
        Matrix.h
        RandomT.h
        ThreadPool.h
        matmulBenchmark.h
        tests/matmul_unit_test.cpp
)
set_target_properties(BackPropagation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if (DEBUG_MODE)
    find_package(GSL REQUIRED)
    target_link_libraries(BackPropagation GSL::gsl GSL::gslcblas)
endif ()
