#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "../RandomT.h"
#include "../Matrix.cuh"
#include "../Matrix.h"
#include "matmul_unit_test.cuh"

#ifdef DEBUG

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_cblas.h>

#endif

bool eq(double *a, double *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        auto max = std::max(std::abs(a[i]), std::abs(b[i]));
        if ((std::abs(a[i]) - std::abs(b[i]) > max * relative_difference_factor)) {
            std::cout << a[i] - b[i] << std::endl << max << std::endl;
            std::cout << "Mismatch at index: " << i << " Expected: " << a[i] << " Got: " << b[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

bool matmul_unit_test(size_t n) {
    RandomT<double> r;

    auto *a = new double[n * n];
    auto *b = new double[n * n];
    auto *c = new double[n * n]{};

    for (size_t i = 0; i < n * n; i++) {
        a[i] = r.generate();
        b[i] = r.generate();
    }

    Matrix<double> const m1(n, n, a);
    Matrix<double> const m2(n, n, b);

    auto start_time_parallel = std::chrono::high_resolution_clock::now();
    auto ans_parallel = matrix_multiply_parallel(m1, m2);
    auto end_time_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> const elapsed_time_parallel = end_time_parallel - start_time_parallel;

    Matrix_cu<double> c1(n, n);
    c1.set(a);
    Matrix_cu<double> c2(n, n);
    c2.set(b);
    auto start_time_cu = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply(c1, c2);
    auto end_time_cu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> const elapsed_time_cu = end_time_cu - start_time_cu;

#ifdef DEBUG
    gsl_matrix_view const A = gsl_matrix_view_array(a, n, n);
    gsl_matrix_view const B = gsl_matrix_view_array(b, n, n);
    gsl_matrix_view C = gsl_matrix_view_array(c, n, n);
    auto start_time_gsl = std::chrono::high_resolution_clock::now();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &A.matrix, &B.matrix,
                   0.0, &C.matrix);
    auto end_time_gsl = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> const elapsed_time_gsl = end_time_gsl - start_time_gsl;
#endif

    bool correct;
    std::cout << std::boolalpha;
    std::cout << "Matrix multiplication test" << std::endl;
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Matrix multiplication CUDA == Matrix multiplication parallel  "
              << (correct = (ans == ans_parallel))
              << std::endl;

#if DEBUG
    bool const temp = eq(c, ans_parallel.data, n * n);
    std::cout << "Matrix multiplication parallel == GSL matrix multiplication   " << temp << std::endl;
    correct &= temp;
    std::cout << "GSL time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time_gsl).count()
              << "ms" << std::endl;
#endif
    std::cout << "Parallel time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time_parallel).count()
              << "ms" << std::endl;
    std::cout << "CUDA time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time_cu).count()
              << "ms" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    return correct;
}