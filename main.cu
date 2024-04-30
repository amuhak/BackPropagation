#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.cuh"
#include "Matrix.h"
#include <cstdio>

#define STR(x) #x

template<typename T>
void main1();

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
    Matrix_cu<float> a(2,2);
    auto *data = new float[4]{1, 2, 3, 4};
    a.set(data);
    std::cout << a[0][0] << " " << a[0][3] << std::endl;
    return 0;
}
void test(){

    std::cout << "TYPE: double\n";
    main1<double>();
    std::cout << "TYPE: float\n";
    main1<float>();
    std::cout << "TYPE: int\n";
    main1<int>();
}

template<typename T>
void main1() {
    int x = 1 << 10;
    int y = 1 << 10;
    auto *aData = new T[x * y];
    auto *bData = new T[x * y];
    RandomT<T> rand;
    for (int i = 0; i < x * y; i++) {
        aData[i] = rand.generate();
        bData[i] = rand.generate();
    }
    Matrix<T> a(x, y);
    a.set(aData);
    Matrix<T> b(y, x);
    b.set(bData);

    auto start = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply_parallel(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTimeParallel = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    Matrix_cu<T> a_cu(x, y);
    a_cu.set(aData);
    Matrix_cu<T> b_cu(y, x);
    b_cu.set(bData);

    start = std::chrono::high_resolution_clock::now();
    auto ans_cu = matrix_multiply(a_cu, b_cu);
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "size: " << x << "x" << y << std::endl;
    std::cout << std::boolalpha << "ans_cu == ans: " << (ans_cu == ans) << std::endl;
    std::cout << "CPU Parallel Time: " << cpuTimeParallel << "ms\n";
    std::cout << "GPU Time: " << gpuTime << "ms\n" << std::endl;

    delete[] aData;
    delete[] bData;
}
