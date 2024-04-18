#ifdef DEBUG

#include <iostream>
#include <chrono>
#include <cstdio>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

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

bool matmul_unit_test() {
    RandomT<double> r;
    const int n = 1000;
    double *a = new double[n * n];
    double *b = new double[n * n];
    double *c = new double[n * n]{};
    for (int i = 0; i < n * n; i++) {
        a[i] = r.generate();
        b[i] = r.generate();
    }
    Matrix<double> m1(n, n, a);
    Matrix<double> m2(n, n, b);
    auto m3 = matrix_multiply_parallel(m1, m2);
    matmulBenchmark(m1, m2, matrix_multiply_parallel < double, double > );
    // m3.print();
    gsl_matrix_view A = gsl_matrix_view_array(a, n, n);
    gsl_matrix_view B = gsl_matrix_view_array(b, n, n);
    gsl_matrix_view C = gsl_matrix_view_array(c, n, n);
    auto start = std::chrono::high_resolution_clock::now();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &A.matrix, &B.matrix,
                   0.0, &C.matrix);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << "s\n";
    std::cout << std::boolalpha;
    return eq(m3.data, C.matrix.data, m3.length);
}
#endif