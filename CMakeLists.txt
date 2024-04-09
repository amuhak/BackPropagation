cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "nvcc")

project(BackPropagation CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 ") # -g -G -lineinfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
set(CMAKE_CUDA_STANDARD 17)

add_executable(BackPropagation
        main.cu
        Matrix.cpp
        Matrix.h
        RandomT.h
        ThreadPool.h
        matmulBenchmark.h
)

set_target_properties(BackPropagation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
