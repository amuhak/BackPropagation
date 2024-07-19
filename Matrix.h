// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_H
#define BACKPROPAGATION_MATRIX_H

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>
#include <cstddef>
#include <numeric>
#include "RandomT.h"
#include "ThreadPool.h"

#ifdef DEBUG

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_cblas.h>

#endif

constexpr size_t THRESHOLD_TO_USE_PARALLEL = 185;            // You can tune this value to get the best performance
constexpr auto relative_difference_factor = 0.0001;
const size_t CONCURRENCY_LIMIT = std::max((size_t) 1, (size_t) std::thread::hardware_concurrency());

template<typename T>
class Matrix {
public:
    size_t rows{}; //NOLINT
    size_t cols{}; //NOLINT
    size_t length{}; //NOLINT
    T *data = nullptr;    //NOLINT

    /**
     * Initializes the matrix with the given size
     * @param other_rows Number of rows
     * @param other_cols Number of columns
     */
    void init(size_t other_rows, size_t other_cols) {
        this->rows = other_rows;
        this->cols = other_cols;
        this->length = other_rows * other_cols;
        if (length == 0) {
            data = nullptr;
        } else {
            this->data = new T[length];
        }
    }

    /**
     * Default constructor for the Matrix class
     */
    Matrix() {
        init(0, 0);
    }

    /**
     * Constructor for the Matrix class
     * @param rows number of rows
     * @param cols number of columns
     */
    Matrix(size_t rows, size_t cols) {
        init(rows, cols);
    }

    /**
     * Constructor for the Matrix class. Copies the data from the pointer. Does not copy the pointer.
     * You are free to delete the pointer after this.
     * @param rows number of rows
     * @param cols number of columns
     * @param data pointer to the data
     */
    Matrix(size_t rows, size_t cols, T *data) {
        init(rows, cols);
        set(data);
    }

    /**
     * Copy constructor
     * @param other matrix to copy from
     */
    Matrix(const Matrix<T> &other) {
        init(other.rows, other.cols);
        set(other.data);
    }

    /**
     * Constructor for casting.
     * @tparam U Type of other matrix
     * @param other Other matrix
     */
    template<class U>
    explicit Matrix(const Matrix<U> &other) {
        init(other.rows, other.cols);
        for (size_t i = 0; i < length; i++) {
            this->data[i] = (T) other.data[i];
        }
    }

    /**
     * Move constructor
     * @param other matrix to move from
     */
    Matrix(Matrix &&other) noexcept {
        using std::swap;
        swap(rows, other.rows);
        swap(cols, other.cols);
        swap(length, other.length);
        swap(data, other.data);
    }

    T sum() {
        return std::accumulate(data, data + length, static_cast<T>(0));
    }

    /**
     * Fills the matrix with the given value
     * @param value value to fill the matrix with
     */
    void fill(T value) {
        std::fill(data, data + length, value);
    }

    /**
     * Fills the matrix with 0s
     */
    void fill0() {
        fill(static_cast<T>(0));
    }

    /**
     * Sets the matrix with the given data. Copies the data from the pointer. Does not copy the pointer.
     * You are free to delete the pointer after this.
     * @param input pointer to the data
     */
    void set(T *input) {
        std::copy(input, input + length, this->data);
    }


    /**
     * @param index row to access
     * @return a pointer to the row
     */
    auto operator[](const size_t index) const {
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

        using std::abs;

        for (size_t i = 0; i < length; i++) {
            auto diff = abs(data[i]) - abs(other.data[i]);
            auto max = std::max(abs(data[i]), abs(other.data[i]));
            if (diff > max * relative_difference_factor) {
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
        if (this == &other) {
            return *this;
        }

        if (length != other.length) {
            delete[] data;
            data = new T[other.length];
        }

        rows = other.rows;
        cols = other.cols;
        length = other.length;
        set(other.data);
        return *this;
    }

    /**
     * Move assignment operator
     * @param other matrix to move from
     * @return reference to the new matrix
     */
    Matrix &operator=(Matrix<T> &&other) noexcept {
        using std::swap;
        if (this == &other) {
            return *this;
        }
        swap(rows, other.rows);
        swap(cols, other.cols);
        swap(length, other.length);
        swap(data, other.data);
        return *this;
    }

    Matrix<T> operator/(T scalar) {
        Matrix<T> result(*this);
        T *temp = result.data;
        for (size_t i = 0; i < length; i++) {
            temp[i] /= scalar;
        }
        return result;
    }

    Matrix<T> operator*(T scalar) {
        Matrix<T> result(*this);
        T *temp = result.data;
        for (size_t i = 0; i < length; i++) {
            temp[i] *= scalar;
        }
        return result;
    }

    Matrix<T> operator+(T scalar) {
        Matrix<T> result(*this);
        T *temp = result.data;
        for (size_t i = 0; i < length; i++) {
            temp[i] += scalar;
        }
        return result;
    }

    Matrix<T> operator-(T scalar) {
        Matrix<T> result(*this);
        T *temp = result.data;
        for (size_t i = 0; i < length; i++) {
            temp[i] -= scalar;
        }
        return result;
    }

    Matrix<T> operator+(Matrix<T> &other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < length; i++) {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        if (rows == other.rows && other.cols == 1) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                auto add = other.data[i];
                auto *ptr = (*this)[i];
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = ptr[j] + add;
                }
            }
            return result;
        }

        throw std::invalid_argument("Matrix dimensions do not match. " +
                                    std::to_string(rows) + "x" + std::to_string(cols) + " and " +
                                    std::to_string(other.rows) + "x" + std::to_string(other.cols) + " respectively.");
    }

    Matrix<T> operator-(Matrix<T> &other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < length; i++) {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }
        if (rows == other.rows && other.cols == 1) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                auto sub = other.data[i];
                auto *ptr = (*this)[i];
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = ptr[j] - sub;
                }
            }
            return result;
        }
        throw std::invalid_argument("Matrix dimensions do not match. " +
                                    std::to_string(rows) + "x" + std::to_string(cols) + " and " +
                                    std::to_string(other.rows) + "x" + std::to_string(other.cols) + " respectively.");
    }

    // Element wise multiplication
    Matrix<T> operator*(Matrix<T> &other) {
        if (rows == other.rows && cols == other.cols) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < length; i++) {
                result.data[i] = data[i] * other.data[i];
            }
            return result;
        }
        if (rows == other.rows && other.cols == 1) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                auto multi = other.data[i];
                auto *ptr = (*this)[i];
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = ptr[j] * multi;
                }
            }
            return result;
        }
        throw std::invalid_argument("Matrix dimensions do not match. " +
                                    std::to_string(rows) + "x" + std::to_string(cols) + " and " +
                                    std::to_string(other.rows) + "x" + std::to_string(other.cols) + " respectively.");
    }

#ifdef DEBUG

    /**
     * @param gsl_matrix to compare with
     * @return true if the matrices are equal, false otherwise
     */
    bool operator==(const gsl_matrix &other) const {
        for (size_t i = 0; i < length; i++) {
            if (data[i] != other.data[i]) {
                std::cout << "Mismatch at index: " << i << " Expected: " << other.data[i] << " Got: " << data[i]
                          << std::endl;
                return false;
            }
        }
        return true;
    }

    gsl_matrix_view to_gsl_matrix() {
        return gsl_matrix_view_array(data, rows, cols);
    }

#endif

    /**
     * Fills the matrix with random values. Range is from INT_MIN to INT_MAX
     */
    void fillRandom() {
        RandomT<T> rand;
        for (size_t i = 0; i < length; i++) {
            data[i] = rand.generate();
        }
    }

    void fillRandom(T min, T max) {
        RandomT<T> rand;
        for (size_t i = 0; i < length; i++) {
            data[i] = rand.generate(min, max);
        }
    }

    Matrix<T> t() {
        return transpose();
    }

    Matrix<T> transpose() {
        Matrix<T> ans = Matrix<T>(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                ans[j][i] = (*this)[i][j];
            }
        }
        return ans;
    }

    /**
     * Prints the matrix
     */
    void print() const {
        std::cout << "Matrix: " << rows << "x" << cols << "\n";
        for (size_t i = 0; i < rows; i++) {
            std::cout << "[";
            for (size_t j = 0; j < cols; j++) {
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
        this->data = nullptr;
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
template<typename T>
Matrix<T> matrix_multiply(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions do not match." +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " and " +
                                    std::to_string(b.rows) + "x" + std::to_string(b.cols) + " respectively.");
    }
    Matrix<T> result(a.rows, b.cols);
    T *cache = new T[b.rows];
    T *aData = a.data;
    T *bData = b.data;
    for (size_t i = 0; i < b.cols; ++i) {
        for (size_t j = 0; j < b.rows; ++j) {
            cache[j] = bData[j * b.cols + i];
        }
        for (size_t j = 0; j < a.rows; ++j) {
            T ans{};
            for (size_t k = 0; k < a.cols; ++k) {
                ans += aData[j * a.cols + k] * cache[k];
            }
            result.data[j * result.cols + i] = ans;
        }
    }
    delete[] cache;
    return result;
}

template<typename T>
void
matrix_multiply_solve_for_column_internal(const Matrix<T> *a, const Matrix<T> *b, const Matrix<T> *result, size_t i) {
    T *cache = new T[b->rows];
    T *aData = a->data;
    T *bData = b->data;
    T *resultData = result->data;
    for (size_t j = 0; j < b->rows; ++j) {
        cache[j] = bData[j * b->cols + i];
    }
    for (size_t j = 0; j < a->rows; ++j) {
        T ans{};
        for (size_t k = 0; k < a->cols; ++k) {
            ans += aData[j * a->cols + k] * cache[k];
        }
        resultData[j * result->cols + i] = ans;
    }
    delete[] cache;
}

template<typename T>
void matrix_multiply_thread_worker(const Matrix<T> *a, const Matrix<T> *b, const Matrix<T> *result,
                                   size_t start, size_t end, size_t noOfThreads) {
    for (size_t i = start; i < end; i += noOfThreads) {
        matrix_multiply_solve_for_column_internal(a, b, result, i);
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
template<typename T>
Matrix<T> matrix_multiply_parallel(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions do not match." +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " and " +
                                    std::to_string(b.rows) + "x" + std::to_string(b.cols) + " respectively.");
    }
    Matrix<T> result(a.rows, b.cols);
    auto *aPtr = &a;
    auto *bPtr = &b;
    auto *resultPtr = &result;
    auto bCols = b.cols;
    ThreadPool pool;
    pool.Start();
    for (size_t i = 0; i < CONCURRENCY_LIMIT; i++) {
        pool.QueueJob(
                [aPtr, bPtr, resultPtr, i, bCols] {
                    matrix_multiply_thread_worker<T>(aPtr, bPtr, resultPtr, i, bCols, CONCURRENCY_LIMIT);
                }
        );
    }
    pool.Stop();
    return result;
}

template<typename T>
T matrix_multiply_internal(const Matrix<T> *a, const Matrix<T> *b, long x, long y) {
    T ans = 0; // Auto doesn't work here for some reason
    for (size_t i = 0; i < a->cols; i++) {
        ans += a->data[x * a->cols + i] * b->data[i * b->cols + y];
    }
    return ans;
}

template<typename T>
Matrix<T> matmult(const Matrix<T> &a, const Matrix<T> &b) {

    uint64_t const threshold = THRESHOLD_TO_USE_PARALLEL * THRESHOLD_TO_USE_PARALLEL * THRESHOLD_TO_USE_PARALLEL;
    uint64_t const num_multiplications = a.rows * a.cols * b.cols;

    if (num_multiplications > threshold) {
        return matrix_multiply_parallel(a, b);
    }
    return matrix_multiply(a, b);
}

#ifdef DEBUG

template<typename T>
Matrix<T> matmultF(Matrix<T> &a, Matrix<T> &b) {
    Matrix<T> result(a.rows, b.cols);
    auto c = a.to_gsl_matrix();
    auto d = b.to_gsl_matrix();
    auto e = result.to_gsl_matrix();
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &c.matrix, &d.matrix,
                   0.0, &e.matrix);
    return result;
}

#endif
#endif //BACKPROPAGATION_MATRIX_H