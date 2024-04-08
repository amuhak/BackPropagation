// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_H
#define BACKPROPAGATION_MATRIX_H

#include "RandomDouble.h"

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
        // Copy data
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

    void fillRandom() {
        RandomDouble rand;
        for (int i = 0; i < length; i++) {
            data[i] = (T) rand.generate();
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
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            result.data[i * result.cols + j] = matrix_multiply_internal(a, b, i, j);
        }
    }
    return result;
}

template<typename T, typename U>
auto matrix_multiply_internal(const Matrix<T> &a, const Matrix<U> &b, long x, long y) {
    auto ans = 0;
    for (int i = 0; i < a.cols; i++) {
        ans += a.data[x * a.cols + i] * b.data[i * b.cols + y];
    }
    return ans;
}

#endif //BACKPROPAGATION_MATRIX_H
