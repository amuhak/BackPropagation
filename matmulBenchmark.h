//
// Created by amuly on 4/8/2024.
//

#ifndef BACKPROPAGATION_MATMULBENCHMARK_H
#define BACKPROPAGATION_MATMULBENCHMARK_H

#include "Matrix.h"

template<typename T, typename F>
void matmulBenchmark(Matrix<T> &A, Matrix<T> &B, F &func) {
    long double time = 0;
    const int n = 1;
    for (int i = 0; i < n; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto C = func(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
        // std::cout << "Elapsed time: " << elapsed.count() << "s\n";
    }
    std::cout << "Average time: " << time / n << "s\n";
}

#endif //BACKPROPAGATION_MATMULBENCHMARK_H
