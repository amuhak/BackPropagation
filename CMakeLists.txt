cmake_minimum_required(VERSION 3.22)
project(BackPropagation CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(BackPropagation
        main.cu
        Matrix.cpp
        Matrix.h
)

set_target_properties(BackPropagation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
