cmake_minimum_required(VERSION 3.22)
option(GPU "Enables gpu support" "OFF")
option(TESTING "Switch to testing builds" "OFF")

# set(TESTING "ON")

message(STATUS "GPU is ${GPU}")
message(STATUS "Testing is ${TESTING}")

if (TESTING STREQUAL "ON")
    message(STATUS "Testing enabled")
    add_compile_definitions(TESTING)
endif ()

if (GPU STREQUAL "ON")
    find_program(CMAKE_CUDA_COMPILER nvcc)
    if (NOT CMAKE_CUDA_COMPILER)
        message(fatal_error "could not find nvcc. please set CMAKE_CUDA_COMPILER to the path of nvcc.
        for example: -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
    endif ()
    enable_language(CUDA)
    message(STATUS "CUDA enabled")
else ()
    message(STATUS "CUDA disabled")
endif ()

if (GPU STREQUAL "ON")
    project(BackPropagation CUDA)
else ()
    project(BackPropagation)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_MODE ON)
    message(STATUS "Debug mode")
else ()
    set(DEBUG_MODE OFF)
    message(STATUS "Release mode")
endif ()

if (DEBUG_MODE)
    if (GPU STREQUAL "ON")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -rdynamic -Wall")
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES ".*[Ii][Nn][Tt][Ee][Ll].*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra -Wconversion -Wsign-conversion -Woverloaded-virtual -pedantic -g -fno-inline -rdynamic")
    endif ()
    add_compile_definitions(DEBUG)
else ()
    if (GPU STREQUAL "ON")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES ".*[Ii][Nn][Tt][Ee][Ll].*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -march=native -flto -fuse-linker-plugin -fprefetch-loop-arrays -fopenmp -fopenmp-simd")
    endif ()
endif ()
if (GPU STREQUAL "ON")
    set(CMAKE_CUDA_STANDARD 17)
else ()
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (GPU STREQUAL "ON")
    add_executable(BackPropagation
            main.cu
            Matrix.h
            RandomT.h
            ThreadPool.h
            Matrix.cuh
            tests/matmul_unit_test.cu
            tests/matmul_unit_test.cuh
            CSV.h
            backPropagation.cuh
    )
else ()
    add_executable(BackPropagation
            main.cpp
            Matrix.h
            RandomT.h
            ThreadPool.h
            tests/matmul_unit_test.cpp
            tests/matmul_unit_test.h
            CSV.h
            backPropagation.h
    )
endif ()

if (GPU STREQUAL "ON")
    set_target_properties(BackPropagation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

if (DEBUG_MODE)
    find_package(GSL REQUIRED)
    target_link_libraries(BackPropagation GSL::gsl GSL::gslcblas)
endif ()
if (GPU STREQUAL "ON")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif ()
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})

enable_testing()

add_test(NAME matmul_unit_test COMMAND BackPropagation)