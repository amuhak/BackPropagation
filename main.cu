#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "ThreadPool.h"
#include "SafeQueue.h"
#include "matmulBenchmark.h"


std::random_device rd;
std::mt19937 mt(rd());
std::uniform_int_distribution<int> dist(-1000, 1000);
auto rnd = std::bind(dist, mt);


// Simple function that adds multiplies two numbers and prints the result
void multiply(const int a, const int b) {
    const int res = a * b;
    std::cout << a << " * " << b << " = " << res << std::endl;
}

// Same as before but now we have an output parameter
void multiply_output(int &out, const int a, const int b) {
    out = a * b;
    std::cout << a << " * " << b << " = " << out << std::endl;
}

// Same as before but now we have an output parameter
int multiply_return(const int a, const int b) {
    const int res = a * b;
    std::cout << a << " * " << b << " = " << res << std::endl;
    return res;
}


int main(int argc, char *argv[]) {

    Matrix<float> a(3, 3);
    a[0][0] = 2;
    a[0][1] = 3;
    a[0][2] = 4;
    a[1][0] = 5;
    a[1][1] = 6;
    a[1][2] = 7;
    a[2][0] = 8;
    a[2][1] = 9;
    a[2][2] = 10;
    a.print();
    Matrix<float> b(3, 3);
    b[0][0] = 11;
    b[0][1] = 12;
    b[0][2] = 13;
    b[1][0] = 14;
    b[1][1] = 15;
    b[1][2] = 16;
    b[2][0] = 17;
    b[2][1] = 18;
    b[2][2] = 19;
    b.print();
    auto C = matrix_multiply_parallel(a, b);
    C.print();
    return 0;
    ThreadPool pool(30);
    pool.init();

    // Submit work to the pool
    for (int i = 1; i < 3; ++i) {
        for (int j = 1; j < 10; ++j) {
            pool.submit(multiply, i, j);
        }
    }

    // Submit function with output parameter passed by ref
    int output_ref;
    auto future1 = pool.submit(multiply_output, std::ref(output_ref), 5, 6);

    // Wait for multiplication output to finish
    future1.get();
    std::cout << "Last operation result is equals to " << output_ref << std::endl;

    // Submit function with return parameter
    auto future2 = pool.submit(multiply_return, 5, 3);

    // Wait for multiplication output to finish
    int res = future2.get();
    std::cout << "Last operation result is equals to " << res << std::endl;

    pool.shutdown();
    std::cout << "Pool is shut down" << std::endl;
    return 0;
}