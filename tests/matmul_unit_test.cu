#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sys/types.h>

#include "../RandomT.h"
#include "../Matrix.cuh"
#include "matmul_unit_test.cuh"

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
    auto ans_parallel = matrix_multiply_parallel(m1, m2);

    Matrix_cu<double> c1 (n,n);
    c1.set(a);
    Matrix_cu<double> c2 (n,n);
    c2.set(b);

    auto ans = matrix_multiply(c1, c2);

#ifdef DEBUG
    gsl_matrix_view const A = gsl_matrix_view_array(a, n, n);
    gsl_matrix_view const B = gsl_matrix_view_array(b, n, n);
    gsl_matrix_view C = gsl_matrix_view_array(c, n, n);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &A.matrix, &B.matrix,
                   0.0, &C.matrix);
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
#endif

    delete[] a;
    delete[] b;
    delete[] c;
    return correct;
}