// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_H
#define BACKPROPAGATION_MATRIX_H

#include <stdexcept>
#include <iostream>
#include <gsl/gsl_matrix.h>
#include "RandomT.h"

const int CONCURRENCY_LIMIT = (int) std::thread::hardware_concurrency();

template<typename T>
class Matrix {
public:
    int rows{};
    int cols{};
    int length;
    T *data;

    Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument(
                    "Invalid matrix size, rows and cols must be greater than 0. Rows: " + std::to_string(rows) +
                    ", Cols: " + std::to_string(cols));
        }
        this->rows = rows;
        this->cols = cols;
        length = rows * cols;
        this->data = new T[length];
    }

    Matrix(int rows, int cols, T *data) {
        this->rows = rows;
        this->cols = cols;
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument(
                    "Invalid matrix size, rows and cols must be greater than 0. Rows: " + std::to_string(rows) +
                    ", Cols: " + std::to_string(cols));
        }
        length = rows * cols;
        this->data = new T[length];
        // Copy data, don't copy the pointer
        for (int i = 0; i < length; i++) {
            this->data[i] = data[i];
        }
    }

    void fill(T value) {
        for (int i = 0; i < length; i++) {
            data[i] = value;
        }
    }

    void set(T *input) {
        for (int i = 0; i < length; i++) {
            this->data[i] = input[i];
        }
    }

    void fill0() {
        fill(0);
    }

    auto operator[](long index) {
        return (data + (index * cols));
    }

    bool operator==(const Matrix<T> &other) const {
        if (rows != other.rows || cols != other.cols) {
            return false;
        }
        for (int i = 0; i < length; i++) {
            if (data[i] != other.data[i]) {
                std::cout << "Mismatch at index: " << i << " Expected: " << other.data[i] << " Got: " << data[i]
                          << std::endl;
                return false;
            }
        }
        return true;
    }

    #ifdef DEBUG_MODE
    bool operator==(const gsl_matrix &other) const {
        for (int i = 0; i < length; i++) {
            if (data[i] != other.data[i]) {
                std::cout << "Mismatch at index: " << i << " Expected: " << other.data[i] << " Got: " << data[i]
                          << std::endl;
                return false;
            }
        }
        return true;
    }
    #endif

    void fillRandom() {
        RandomT<T> rand;
        for (int i = 0; i < length; i++) {
            data[i] = rand.generate();
        }
    }

    void print() {
        std::cout << "Matrix: " << rows << "x" << cols << "\n";
        for (int i = 0; i < rows; i++) {
            std::cout << "[";
            for (int j = 0; j < cols; j++) {
                std::cout << data[i * cols + j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << "]";
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    ~Matrix() {
        delete[] this->data;
    }
};

template<typename T, typename U>
auto matrix_multiply(const Matrix<T> &a, const Matrix<U> &b) {
    if (a.cols != b.rows || a.rows != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match." +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " and " +
                                    std::to_string(b.rows) + "x" + std::to_string(b.cols) + " respectively.");
    }
    Matrix<T> result(a.rows, b.cols);
    matrix_multiply_solve_for_range_internal(&a, &b, &result, 0, result.length - 1);
    return result;
}

template<typename T, typename U, typename V>
auto
matrix_multiply_solve_for_range_internal(const Matrix<T> *a,
                                         const Matrix<U> *b,
                                         Matrix<V> *result,
                                         long long start,
                                         long long end) {
    long long xStart = start / b->cols;
    long long yStart = start % b->cols;
    long long xEnd = (end) / b->cols;
    long long yEnd = (end) % b->cols;
    for (long long i = xStart; i <= xEnd; i++) {
        const long yEndReal = (i == xEnd ? yEnd : (b->cols - 1));
        const long yStartReal = (i == xStart ? yStart : 0);
        for (long long j = yStartReal; j <= yEndReal; j++) {
            result->data[i * result->cols + j] = matrix_multiply_internal(a, b, i, j);
        }
    }
}

template<typename T, typename U>
auto matrix_multiply_parallel(const Matrix<T> &a, const Matrix<U> &b) {
    if (a.cols != b.rows || a.rows != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match." +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " and " +
                                    std::to_string(b.rows) + "x" + std::to_string(b.cols) + " respectively.");
    }
    Matrix<T> result(a.rows, b.cols);
    result.fill(69);
    const long long n = result.length - 1;
    long long noOfSolutionsPerThread = n / CONCURRENCY_LIMIT;
    noOfSolutionsPerThread++;
    // noOfSolutionsPerThread = 1;
    // noOfSolutionsPerThread = std::max(noOfSolutionsPerThread, 1LL);
    ThreadPool pool;
    pool.Start();
    auto *aPtr = &a;
    auto *bPtr = &b;
    auto *resultPtr = &result;
    for (long long i = 0; i <= n; i += noOfSolutionsPerThread) {
        pool.QueueJob(
                [noOfSolutionsPerThread, i, aPtr, bPtr, resultPtr, n] {
                    matrix_multiply_solve_for_range_internal<T, U, T>(aPtr,
                                                                      bPtr,
                                                                      resultPtr,
                                                                      i,
                                                                      std::min(i + noOfSolutionsPerThread - 1, n));
                }
        );
    }
    pool.Stop();
    return result;
}

template<typename T, typename U>
auto matrix_multiply_internal(const Matrix<T> *a, const Matrix<U> *b, long x, long y) {
    decltype(T{} * U{}) ans = 0; // Auto doesn't work here for some reason
    for (int i = 0; i < a->cols; i++) {
        ans += a->data[x * a->cols + i] * b->data[i * b->cols + y];
    }
    return ans;
}

#endif //BACKPROPAGATION_MATRIX_H