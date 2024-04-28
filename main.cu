#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.cuh"
#include "Matrix.h"
#include <cstdio>

#define TYPE int

int main() {
    int n = 1024;
    auto *aData = new TYPE[n * n];
    auto *bData = new TYPE[n * n];
    RandomT<TYPE> rand;
    for (int i = 0; i < n * n; i++) {
        aData[i] = rand.generate();
        bData[i] = rand.generate();
    }
    Matrix<TYPE> a(n, n);
    a.set(aData);
    Matrix<TYPE> b(n, n);
    b.set(bData);
    /*auto start = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();*/
    auto start = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply_parallel(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTimeParallel = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CPU\n";
    // ans.prTYPE();
    Matrix_cu<TYPE> a_cu(n, n);
    a_cu.set(aData);
    Matrix_cu<TYPE> b_cu(n, n);
    b_cu.set(bData);
    std::cout << "CUDA\n";
    start = std::chrono::high_resolution_clock::now();
    auto ans_cu = matrix_multiply(a_cu, b_cu);
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // ans_cu.prTYPE();
    std::cout << "size: " << n << "x" << n << std::endl;
    std::cout << std::boolalpha << "ans_cu == ans: " << (ans_cu == ans) << std::endl;
    // std::cout << "CPU Time: " << cpuTime << "ms\n";
    std::cout << "CPU Parallel Time: " << cpuTimeParallel << "ms\n";
    std::cout << "GPU Time: " << gpuTime << "ms\n";
}