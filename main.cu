#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.cuh"
#include "Matrix.h"
#include <cstdio>

#define STR(x) #x

int main1();

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main() {
    warm_up_gpu<<<1, 1024>>>();
    cudaDeviceSynchronize();
    std::cout << "Warm up done\n";
#define TYPE double
    main1();
#define TYPE float
    main1();
#define TYPE int
    main1();
    return 0;
}

int main1() {
    int x = 1024;
    int y = 100;
    auto *aData = new TYPE[x * y];
    auto *bData = new TYPE[x * y];
    RandomT<TYPE> rand;
    for (int i = 0; i < x * y; i++) {
        aData[i] = rand.generate();
        bData[i] = rand.generate();
    }
    Matrix<TYPE> a(x, y);
    a.set(aData);
    Matrix<TYPE> b(y, x);
    b.set(bData);
    /*auto start = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();*/
    auto start = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply_parallel(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTimeParallel = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // ans.prTYPE();
    Matrix_cu<TYPE> a_cu(x, y);
    a_cu.set(aData);
    Matrix_cu<TYPE> b_cu(y, x);
    b_cu.set(bData);
    start = std::chrono::high_resolution_clock::now();
    auto ans_cu = matrix_multiply(a_cu, b_cu);
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // ans_cu.prTYPE();
    std::cout << "size: " << x << "x" << y << std::endl;
    std::cout << std::boolalpha << "ans_cu == ans: " << (ans_cu == ans) << std::endl;
    // std::cout << "CPU Time: " << cpuTime << "ms\n";
    std::cout << "CPU Parallel Time: " << cpuTimeParallel << "ms\n";
    std::cout << "GPU Time: " << gpuTime << "ms\n" << std::endl;
}