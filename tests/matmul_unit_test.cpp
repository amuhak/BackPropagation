#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sys/types.h>
#include <chrono>
#include <iomanip>
#include "../RandomT.h"
#include "../Matrix.h"
#include "matmul_unit_test.h"

#ifdef DEBUG

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_cblas.h>

#endif

bool eq(double *a, double *b, int len) {
    const auto relative_difference_factor = 0.000001;
    for (int i = 0; i < len; i++) {
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

bool matmul_unit_test(int no) {
    std::cout << std::setprecision(12);
    RandomT<double> r;
    const ulong n = no;

    auto *a = new double[n * n];
    auto *b = new double[n * n];
    auto *c = new double[n * n]{};

    for (uint i = 0; i < n * n; i++) {
        a[i] = r.generate();
        b[i] = r.generate();
    }

    Matrix<double> const m1(n, n, a);
    Matrix<double> const m2(n, n, b);

    auto start = std::chrono::high_resolution_clock::now();
    auto ans = matrix_multiply(m1, m2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    auto start_parallel = std::chrono::high_resolution_clock::now();
    auto ans_parallel = matrix_multiply_parallel(m1, m2);
    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_parallel = end_parallel - start_parallel;

#ifdef DEBUG
    gsl_matrix_view const A = gsl_matrix_view_array(a, n, n);
    gsl_matrix_view const B = gsl_matrix_view_array(b, n, n);
    gsl_matrix_view C = gsl_matrix_view_array(c, n, n);
    auto start_gsl = std::chrono::high_resolution_clock::now();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &A.matrix, &B.matrix,
                   0.0, &C.matrix);
    auto end_gsl = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_gsl = end_gsl - start_gsl;
#endif

    bool correct;
    std::cout << std::boolalpha;
    std::cout << "Matrix multiplication test" << std::endl;
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Matrix multiplication single threaded == Matrix multiplication parallel  "
              << (correct = (ans == ans_parallel))
              << std::endl;

#if DEBUG
    bool const temp = eq(c, ans.data, n * n);
    std::cout << "Matrix multiplication single threaded == GSL matrix multiplication       " << temp << std::endl;
    correct &= temp;
    std::cout << "GSL time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds_gsl).count()
              << "ms" << std::endl;
#endif
    std::cout << "Single threaded time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count()
              << "ms" << std::endl;
    std::cout << "Multithreaded time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds_parallel).count() << "ms"
              << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    return correct;
}