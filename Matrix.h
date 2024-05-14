// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_H
#define BACKPROPAGATION_MATRIX_H

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>

#ifdef DEBUG_MODE
#include <gsl/gsl_matrix.h>
#endif

#include "RandomT.h"
#include "ThreadPool.h"

const int CONCURRENCY_LIMIT = (int) std::thread::hardware_concurrency();

template<typename T>
class Matrix {
public:
    int rows{}; //NOLINT
    int cols{}; //NOLINT
    int length; //NOLINT
    T *data;    //NOLINT

    /**
     * Constructor for the Matrix class
     * @param rows number of rows
     * @param cols number of columns
     */
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

    /**
     * Constructor for the Matrix class. Copies the data from the pointer. Does not copy the pointer.
     * You are free to delete the pointer after this.
     * @param rows number of rows
     * @param cols number of columns
     * @param data pointer to the data
     */
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

    /**
     * Copy constructor
     * @param other matrix to copy from
     */
    Matrix(const Matrix<T> &other) {
        std::cout << "Copy constructor called" << std::endl;
        this->rows = other.rows;
        this->cols = other.cols;
        length = other.length;
        this->data = new T[length];
        set(other.data);
    }

    Matrix(Matrix &&other) noexcept {
        std::cout << "Move constructor called\n";
        this->rows = other.rows;
        this->cols = other.cols;
        length = other.length;
        this->data = other.data;
        other.data = nullptr;
    }

    /**
     * Fills the matrix with the given value
     * @param value value to fill the matrix with
     */
    void fill(T value) {
        for (int i = 0; i < length; i++) {
            data[i] = value;
        }
    }

    /**
     * Sets the matrix with the given data. Copies the data from the pointer. Does not copy the pointer.
     * You are free to delete the pointer after this.
     * @param input pointer to the data
     */
    void set(T *input) {
        for (int i = 0; i < length; i++) {
            this->data[i] = input[i];
        }
    }

    /**
     * Fills the matrix with 0s
     */
    void fill0() {
        fill(0);
    }

    /**
     *
     * @param index row to access
     * @return a pointer to the row
     */
    auto operator[](long index) const {
        return (data + (index * cols));
    }

    /**
     * @param other matrix to compare with
     * @return true if the matrices are equal, false otherwise
     */
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

    /**
     * Copy assignment operator
     * @param other matrix to copy from
     * @return reference to the new matrix
     */
    Matrix &operator=(const Matrix<T> &other) {
        std::cout << "Copy assignment operator called" << std::endl;
        if (this == &other) {
            return *this;
        }
        set(other.data);
        return *this;
    }

    /**
     * Move assignment operator
     * @param other matrix to move from
     * @return reference to the new matrix
     */
    Matrix &operator=(Matrix<T> &&other) noexcept {
        std::cout << "Move assignment operator called\n";
        if (this == &other) {
            return *this;
        }
        delete[] this->data;
        std::swap(data, other.data);
        return *this;
    }

#ifdef DEBUG_MODE
    /**
     * @param gsl_matrix to compare with
     * @return true if the matrices are equal, false otherwise
     */
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

    /**
     * Fills the matrix with random values. Range is from INT_MIN to INT_MAX
     */
    void fillRandom() {
        RandomT<T> rand;
        for (int i = 0; i < length; i++) {
            data[i] = rand.generate();
        }
    }

    /**
     * Prints the matrix
     */
    void print() {
        std::cout << "Matrix: " << rows << "x" << cols << "\n";
        for (int i = 0; i < rows; i++) {
            std::cout << "[";
            for (int j = 0; j < cols; j++) {
                std::cout << data[i * cols + j];
                if (j < cols - 1) {
                    std::cout << ", ";
                }
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

/**
 * Multiplies two matrices
 * @tparam T Type of the first matrix
 * @tparam U Type of the second matrix
 * @param a Input matrix 1
 * @param b Input matrix 2
 * @return The result of the matrix multiplication. The return type is:  \code Matrix &lt;decltype(T{} * U{})&gt; \endcode
 */
template<typename T, typename U>
Matrix<decltype(T{} * U{})> matrix_multiply(const Matrix<T> &a, const Matrix<U> &b) {
    if (a.cols != b.rows || a.rows != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match." +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " and " +
                                    std::to_string(b.rows) + "x" + std::to_string(b.cols) + " respectively.");
    }
    Matrix<decltype(T{} * U{})> result(a.rows, b.cols);
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

/**
 * Multiplies two matrices in parallel on the CPU
 * @tparam T Type of the first matrix
 * @tparam U Type of the second matrix
 * @param a Input matrix 1
 * @param b Input matrix 2
 * @return The result of the matrix multiplication. The return type is:  \code Matrix &lt;decltype(T{} * U{})&gt; \endcode
 */
template<typename T, typename U>
Matrix<decltype(T{} * U{})> matrix_multiply_parallel(const Matrix<T> &a, const Matrix<U> &b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions do not match." +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " and " +
                                    std::to_string(b.rows) + "x" + std::to_string(b.cols) + " respectively.");
    }
    Matrix<decltype(T{} * U{})> result(a.rows, b.cols);
    result.fill(69);
    const long long n = result.length - 1;
    long long noOfSolutionsPerThread = n / CONCURRENCY_LIMIT;
    noOfSolutionsPerThread++;
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
decltype(T{} * U{}) matrix_multiply_internal(const Matrix<T> *a, const Matrix<U> *b, long x, long y) {
    decltype(T{} * U{}) ans = 0; // Auto doesn't work here for some reason
    for (int i = 0; i < a->cols; i++) {
        ans += a->data[x * a->cols + i] * b->data[i * b->cols + y];
    }
    return ans;
}

#endif //BACKPROPAGATION_MATRIX_H