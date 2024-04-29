// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_CUH
#define BACKPROPAGATION_MATRIX_CUH

#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gsl/gsl_matrix.h>

#include "RandomT.h"
#include "Matrix.h"

#define EPSILON 0.00001
#define SHIFT_SIZE 1024

template<typename T>
class Matrix_cu {
public:
    int *rows;
    int rowsCPU;
    int *cols;
    int colsCPU;
    int *length;
    int lengthCPU;
    T *data;

    Matrix_cu(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument(
                    "Invalid matrix size, rows and cols must be greater than 0. Rows: " + std::to_string(rows) +
                    ", Cols: " + std::to_string(cols));
        }
        cudaMalloc(&this->rows, sizeof(int));
        cudaMalloc(&this->cols, sizeof(int));
        cudaMalloc(&this->length, sizeof(int));
        cudaMemcpy(this->rows, &rows, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->cols, &cols, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->length, &rows, sizeof(int), cudaMemcpyHostToDevice);
        this->rowsCPU = rows;
        this->colsCPU = cols;
        this->lengthCPU = rows * cols;
        cudaMalloc(&data, lengthCPU * sizeof(T));
    }


    void fill(T value) {
        T *temp = new T[lengthCPU];
        for (int i = 0; i < lengthCPU; i++) {
            temp[i] = value;
        }
        cudaMemcpy(this->data, temp, lengthCPU * sizeof(T), cudaMemcpyHostToDevice);
        delete[] temp;
    }

    void set(T *input) {
        cudaMemcpy(this->data, input, lengthCPU * sizeof(T), cudaMemcpyHostToDevice);
    }

    void fill0() {
        fill(0);
    }

    auto operator[](long index) {
        return (data + (index * *cols));
    }

    bool operator==(const Matrix_cu<T> &other) const {
        if (rows != other.rows || cols != other.cols) {
            return false;
        }
        for (int i = 0; i < *length; i++) {
            auto acceptableError = data[i] * EPSILON;
            if (data[i] - other.data[i] > acceptableError) {
                std::cout << "Mismatch at index: " << i << " Expected: " << other.data[i] << " Got: " << data[i]
                          << "Diff: " << data[i] - other.data[i] << std::endl;
                return false;
            }
        }
        return true;
    }

    bool operator==(const Matrix<T> &other) const {
        if (rowsCPU != other.rows || colsCPU != other.cols) {
            return false;
        }
        T *dataCPU = new T[lengthCPU];
        cudaMemcpy(dataCPU, data, lengthCPU * sizeof(T), cudaMemcpyDeviceToHost);
        for (int i = 0; i < lengthCPU; i++) {
            auto acceptableError = std::abs(dataCPU[i] * EPSILON);
            if (std::abs(dataCPU[i] - other.data[i]) > acceptableError) {
                std::cout << "Mismatch at index: " << i << " Expected: " << other.data[i] << " Got: " << dataCPU[i]
                          << " Diff: " << dataCPU[i] - other.data[i] << std::endl;
                return false;
            }
        }
        delete[] dataCPU;
        return true;
    }

#ifdef DEBUG_MODE
    bool operator==(const gsl_matrix &other) const {
        for (int i = 0; i < *length; i++) {
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
        for (int i = 0; i < *length; i++) {
            data[i] = rand.generate();
        }
    }

    void print() {
        std::cout << "Matrix: " << rowsCPU << "x" << colsCPU << "\n";
        T *dataCPU = new T[lengthCPU];
        cudaMemcpy(dataCPU, data, lengthCPU * sizeof(T), cudaMemcpyDeviceToHost);
        for (int i = 0; i < rowsCPU; i++) {
            std::cout << "[";
            for (int j = 0; j < colsCPU; j++) {
                std::cout << dataCPU[i * colsCPU + j];
                if (j < colsCPU - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            std::cout << "\n";
        }
        std::cout << std::endl;
        delete[] dataCPU;
    }

    ~Matrix_cu() {
        cudaFree(data);
    }
};

template<typename T, typename U>
auto matrix_multiply_internal_TEST(const Matrix<T> *a, const Matrix<U> *b, long x, long y) {
    decltype(T{} * U{}) ans = 0; // Auto doesn't work here for some reason
    for (int i = 0; i < a->cols; i++) {
        ans += a->data[x * a->cols + i] * b->data[i * b->cols + y];
    }
    return ans;
}

template<typename T, typename U, typename V>
__global__
void
matrix_multiply_internal_cu(T *a, uint aCols, U *b, uint bCols, V *c, uint cCols, uint shiftDown = 0,
                            uint shiftRight = 0) {
    decltype(T{} * U{}) ans = 0; // Auto doesn't work here for some reason
    shiftDown *= SHIFT_SIZE;
    shiftRight *= SHIFT_SIZE;
    auto x = blockIdx.x + shiftDown;
    auto y = threadIdx.x + shiftRight;
    for (long i = 0; i < aCols; i++) {
        ans += a[x * aCols + i] * b[i * bCols + y];
    }
    c[x * cCols + y] = ans;
}

template<typename T, typename U>
auto matrix_multiply(const Matrix_cu<T> &a, const Matrix_cu<U> &b) {
    if (a.colsCPU != b.rowsCPU) {
        std::cout << "Matrix dimensions do not match." +
                     std::to_string(a.rowsCPU) + "x" + std::to_string(a.colsCPU) + " and " +
                     std::to_string(b.rowsCPU) + "x" + std::to_string(b.colsCPU) + " respectively.";
        exit(-69);
    }
    Matrix_cu<T> result(a.rowsCPU, b.colsCPU);
    result.fill0();
    const int shiftDown = a.rowsCPU / (SHIFT_SIZE + 1);
    const int shiftRight = b.colsCPU / (SHIFT_SIZE + 1);
    for (int i = 0; i <= shiftDown; i++) {
        for (int j = 0; j <= shiftRight; j++) {
            int blockSize = min(SHIFT_SIZE, a.rowsCPU - i * SHIFT_SIZE);
            int noOfThreads = min(SHIFT_SIZE, b.colsCPU - j * SHIFT_SIZE);
            noOfThreads = max(1, noOfThreads);
            matrix_multiply_internal_cu<<<blockSize, noOfThreads>>>(
                    a.data, a.colsCPU,
                    b.data, b.colsCPU,
                    result.data, result.colsCPU,
                    i, j);
            cudaDeviceSynchronize();
        }
    }
    return result;
}

#endif //BACKPROPAGATION_MATRIX_CUH