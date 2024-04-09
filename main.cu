#include <iostream>
#include <chrono>
#include "ThreadPool.h"
#include "matmulBenchmark.h"


int main(int argc, char *argv[]) {
    const int n = 1000;
    Matrix<int> a(n, n);
    Matrix<int> b(n, n);
    a.fillRandom();
    b.fillRandom();
    // matmulBenchmark(a, b, matrix_multiply<int,int>);
    auto c = matrix_multiply(a, b);
    std::cout << "not parallel" << std::endl;
    matmulBenchmark(a, b, matrix_multiply<int, int>);
    std::cout << "parallel" << std::endl;
    matmulBenchmark(a, b, matrix_multiply_parallel<int, int>);
    auto d = matrix_multiply_parallel(a, b);
    std::cout << "c == d: " << (c == d) << std::endl;
    return 0;
}