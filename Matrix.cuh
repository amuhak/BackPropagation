// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_CUH
#define BACKPROPAGATION_MATRIX_CUH

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "Matrix.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_types.h"
#include "driver_types.h"

constexpr long double EPSILON = 0.00001;

template<typename T>
__global__  void div_kernel(T *out, T *data, T div, size_t length, size_t shift);

template<typename T>
__global__  void mult_kernel(T *out, T *data, T div, size_t length, size_t shift);

template<typename T>
__global__  void add_kernel(T *out, T *data, T div, size_t length, size_t shift);

template<typename T>
__global__  void sub_kernel(T *out, T *data, T div, size_t length, size_t shift);

inline size_t NO_OF_THREADS;
inline size_t NO_OF_BLOCKS;

template<typename T>
class Matrix_cu {
public:
    size_t *rows = nullptr;            //NOLINT
    size_t rowsCPU{};                  //NOLINT
    size_t *cols = nullptr;            //NOLINT
    size_t colsCPU{};                  //NOLINT
    size_t *length = nullptr;          //NOLINT
    size_t lengthCPU{};                //NOLINT
    T *data = nullptr;                 //NOLINT

    /**
     * Initializes the matrix with the given size
     * @param other_rows Number of rows
     * @param other_cols Number of columns
     */
    void init(size_t rows_, size_t cols_) {
        cudaMalloc(&this->rows, sizeof(size_t));
        cudaMalloc(&this->cols, sizeof(size_t));
        cudaMalloc(&this->length, sizeof(size_t));
        cudaMemcpy(this->rows, &rows_, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(this->cols, &cols_, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(this->length, &rows_, sizeof(size_t), cudaMemcpyHostToDevice);
        this->rowsCPU = rows_;
        this->colsCPU = cols_;
        this->lengthCPU = rows_ * cols_;
        if (lengthCPU == 0) {
            data = nullptr;
        } else {
            cudaMalloc(&data, lengthCPU * sizeof(T));
        }
        cudaGetDevice(nullptr);
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, 0);
        NO_OF_THREADS = deviceProp.maxThreadsPerBlock;
        NO_OF_BLOCKS = deviceProp.maxThreadsDim[0];
    }

    /**
     * Default constructor for the Matrix class
     */
    Matrix_cu() {
        init(0, 0);
    }

    /**
     * Constructor for the Matrix class
     * @param rows number of rows
     * @param cols number of columns
     */
    Matrix_cu(size_t rows, size_t cols) {
        init(rows, cols);
    }

    /**
     * Constructor for the Matrix class. Copies the data from the pointer. Does not copy the pointer.
     * You are free to delete the pointer after this.
     * @param rows number of rows
     * @param cols number of columns
     * @param data pointer to the data
     */
    Matrix_cu(size_t rows, size_t cols, T *data) {
        init(rows, cols);
        set(data);
    }

    /**
     * Copy constructor
     * @param other matrix to copy from
     */
    Matrix_cu(const Matrix<T> &other) {
        init(other.rows, other.cols);
        set(other.data);
    }

    /**
     * Copy constructor
     * @param other matrix to copy from
     */
    Matrix_cu(const Matrix_cu<T> &other) {
        init(other.rowsCPU, other.colsCPU);
        set_cu(other.data);
    }

    /**
     * Constructor for casting.
     * @tparam U Type of other matrix
     * @param other Other matrix
     */
    template<class U>
    explicit Matrix_cu(const Matrix<U> &other) {
        init(other.rows, other.cols);
        Matrix<T> temp(other);
        set(temp.data);
    }

    /**
     * Constructor for casting.
     * @tparam U Type of other matrix
     * @param other Other matrix
     */
    template<class U>
    explicit Matrix_cu(Matrix_cu<U> &other) {
        init(other.rowsCPU, other.colsCPU);
        Matrix<T> temp(other.toCPU());
        set(temp.data);
    }

    /**
     * Move constructor
     * @param other matrix to move from
     */
    Matrix_cu(Matrix_cu &&other) {
        using std::swap;
        swap(rows, other.rows);
        swap(cols, other.cols);
        swap(length, other.length);
        swap(data, other.data);
        swap(rowsCPU, other.rowsCPU);
        swap(colsCPU, other.colsCPU);
        swap(lengthCPU, other.lengthCPU);
    }

    /**
     * Move constructor
     * @param other matrix to move from
     */
    Matrix_cu(Matrix<T> &&other) {
        using std::swap;
        init(other.rows, other.cols);
        set(other.data);
    }

    T sum() {
        T *dataCPU = new T[lengthCPU];
        cudaMemcpy(dataCPU, data, lengthCPU * sizeof(T), cudaMemcpyDeviceToHost);
        T sum = 0;
        sum = std::accumulate(dataCPU, dataCPU + lengthCPU, sum);
        delete[] dataCPU;
        return sum;
    }

    /**
     * Fill the matrix with a value
     * @param value
     */
    void fill(T value) {
        T *dataCPU = new T[lengthCPU];
        std::fill(dataCPU, dataCPU + lengthCPU, value);
        cudaMemcpy(data, dataCPU, lengthCPU * sizeof(T), cudaMemcpyHostToDevice);
        delete[] dataCPU;
    }

    /**
     * Fills the matrix with 0s
     */
    void fill0() {
        cudaMemset(data, 0, lengthCPU * sizeof(T));
    }

    /**
     * Sets the matrix with the given data. Copies the data from the pointer. Does not copy the pointer.
     * You are free to delete the pointer after this.
     * @param input pointer to the data
     */
    void set(T *input) {
        cudaMemcpy(this->data, input, lengthCPU * sizeof(T), cudaMemcpyHostToDevice);
    }

    void set(size_t row, size_t col, T val) {
        cudaMemcpy(data + (row * colsCPU + col), &val, sizeof(T), cudaMemcpyHostToDevice);
    }

    void set_cu(T *input) {
        cudaMemcpy(data, input, lengthCPU * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    T get(size_t row, size_t col) {
        T val;
        cudaMemcpy(&val, data + (row * colsCPU + col), sizeof(T), cudaMemcpyDeviceToHost);
        return val;
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
        for (size_t i = 0; i < lengthCPU; i++) {
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

    /**
     * Copy assignment operator
     * @param other matrix to copy from
     * @return reference to the new matrix
     */
    Matrix_cu &operator=(const Matrix_cu<T> &other) {
        if (this == &other) {
            return *this;
        }

        if (lengthCPU != other.lengthCPU) {
            cudaFree(data);
            cudaMalloc(&data, other.lengthCPU * sizeof(T));
        }

        cudaMemcpy(rows, other.rows, sizeof(size_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(cols, other.cols, sizeof(size_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(length, other.length, sizeof(size_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(data, other.data, other.lengthCPU * sizeof(T), cudaMemcpyDeviceToDevice);
        rowsCPU = other.rowsCPU;
        colsCPU = other.colsCPU;
        lengthCPU = other.lengthCPU;
        return *this;
    }

    /**
     * Move assignment operator
     * @param other matrix to move from
     * @return reference to the new matrix
     */
    Matrix_cu &operator=(Matrix_cu<T> &&other) noexcept {
        using std::swap;
        swap(rows, other.rows);
        swap(cols, other.cols);
        swap(length, other.length);
        swap(data, other.data);
        swap(rowsCPU, other.rowsCPU);
        swap(colsCPU, other.colsCPU);
        swap(lengthCPU, other.lengthCPU);
        return *this;
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

    Matrix_cu<T> operator/(T scalar) {
        Matrix_cu<T> result(rowsCPU, colsCPU);
        size_t noOfThreads = std::min(NO_OF_THREADS, lengthCPU);
        size_t noOfBlocks = std::min(NO_OF_BLOCKS, (lengthCPU + noOfThreads - 1) / noOfThreads);
        div_kernel<<<noOfBlocks, noOfThreads>>>(result.data, data, scalar, lengthCPU, noOfBlocks * noOfThreads);
        return result;
    }

    Matrix_cu<T> operator*(T scalar) {
        Matrix_cu<T> result(rowsCPU, colsCPU);
        size_t noOfThreads = std::min(NO_OF_THREADS, lengthCPU);
        size_t noOfBlocks = std::min(NO_OF_BLOCKS, (lengthCPU + noOfThreads - 1) / noOfThreads);
        mult_kernel<<<noOfBlocks, noOfThreads>>>(result.data, data, scalar, lengthCPU, noOfBlocks * noOfThreads);
        return result;
    }

    Matrix_cu<T> operator+(T scalar) {
        Matrix_cu<T> result(rowsCPU, colsCPU);
        size_t noOfThreads = std::min(NO_OF_THREADS, lengthCPU);
        size_t noOfBlocks = std::min(NO_OF_BLOCKS, (lengthCPU + noOfThreads - 1) / noOfThreads);
        add_kernel<<<noOfBlocks, noOfThreads>>>(result.data, data, scalar, lengthCPU, noOfBlocks * noOfThreads);
        return result;
    }

    Matrix_cu<T> operator-(T scalar) {
        Matrix_cu<T> result(rowsCPU, colsCPU);
        size_t noOfThreads = std::min(NO_OF_THREADS, lengthCPU);
        size_t noOfBlocks = std::min(NO_OF_BLOCKS, (lengthCPU + noOfThreads - 1) / noOfThreads);
        sub_kernel<<<noOfBlocks, noOfThreads>>>(result.data, data, scalar, lengthCPU, noOfBlocks * noOfThreads);
        return result;
    }

    Matrix<T> toCPU() {
        T *dataCPU = new T[lengthCPU];
        cudaMemcpy(dataCPU, data, lengthCPU * sizeof(T), cudaMemcpyDeviceToHost);
        Matrix<T> result(rowsCPU, colsCPU, dataCPU);
        delete[] dataCPU;
        return result;
    }

    Matrix_cu<T> operator+(Matrix<T> &other) {
        Matrix<T> a = toCPU();
        return Matrix_cu<T>(a + other);
    }

    Matrix_cu<T> operator+(Matrix_cu<T> &other) {
        Matrix<T> a = toCPU();
        Matrix<T> b = other.toCPU();
        return Matrix_cu<T>(a + b);
    }

    Matrix_cu<T> operator-(Matrix<T> &other) {
        Matrix<T> a = toCPU();
        return Matrix_cu<T>(a - other);
    }

    Matrix_cu<T> operator-(Matrix_cu<T> &other) {
        Matrix<T> a = toCPU();
        Matrix<T> b = other.toCPU();
        return Matrix_cu<T>(a - b);
    }

    Matrix_cu<T> operator*(Matrix<T> &other) {
        Matrix<T> temp = this->toCPU();
        return (temp) * (other);
    }

    Matrix_cu<T> operator*(Matrix_cu<T> &other) {
        Matrix<T> temp = this->toCPU();
        Matrix<T> temp1 = other.toCPU();
        return temp * temp1;
    }

    void fillRandom() {
        Matrix<T> temp(rowsCPU, colsCPU);
        temp.fillRandom();
        set(temp.data);
    }

    void fillRandom(T min, T max) {
        Matrix<T> temp(rowsCPU, colsCPU);
        temp.fillRandom(min, max);
        set(temp.data);
    }

    Matrix_cu<T> transpose() {
        Matrix<T> temp = toCPU().transpose();
        return Matrix_cu<T>(temp);
    }

    Matrix_cu<T> t() {
        return transpose();
    }

    void print() {
        toCPU().print();
    }

    ~Matrix_cu() {
        cudaFree(data);
        cudaFree(rows);
        cudaFree(cols);
        cudaFree(length);
    }
};

template<typename T>
__global__  void div_kernel(T *out, T *data, T div, size_t length, size_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < length; i += shift) {
        out[i] = data[i] / div;
    }
}

template<typename T>
__global__  void mult_kernel(T *out, T *data, T div, size_t length, size_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < length; i += shift) {
        out[i] = data[i] * div;
    }
}

template<typename T>
__global__  void add_kernel(T *out, T *data, T div, size_t length, size_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < length; i += shift) {
        out[i] = data[i] + div;
    }
}

template<typename T>
__global__  void sub_kernel(T *out, T *data, T div, size_t length, size_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < length; i += shift) {
        out[i] = data[i] - div;
    }
}

template<typename T, typename U, typename V>
__global__
void
matrix_multiply_internal_cu(T *__restrict__ a, uint32_t aCols,
                            U *__restrict__ b, uint32_t bCols,
                            V *__restrict__ c,
                            uint32_t shiftDown = 0, uint32_t shiftRight = 0) {
    V ans{};      // auto may not give you the right type here
    const uint32_t x = blockIdx.x + shiftDown;
    const uint32_t y = threadIdx.x + shiftRight;

    for (uint32_t i = 0; i < aCols; i++) {
        ans += a[x * aCols + i] * b[i * bCols + y];
    }

    c[x * bCols + y] = ans;       // bCols == cCols
}

template<typename T, typename U>
auto matrix_multiply(const Matrix_cu<T> &a, const Matrix_cu<U> &b) {
    if (a.colsCPU != b.rowsCPU) {
        std::cout << "Matrix dimensions do not match. " << a.rowsCPU << " x " << a.colsCPU << " and " << b.rowsCPU
                  << " x " << b.colsCPU << " respectively.";
        exit(-69);
    }

    Matrix_cu<decltype(T{} * U{})> result(a.rowsCPU, b.colsCPU);
    const uint32_t shiftDown = (result.rowsCPU - 1) / (NO_OF_BLOCKS);
    const uint32_t shiftRight = (result.colsCPU - 1) / (NO_OF_THREADS);
    const uint32_t totalStreams = (shiftDown + 1) * (shiftRight + 1);
    std::vector<cudaStream_t> streams(totalStreams);

    for (auto &i: streams) {
        cudaStreamCreate(&i);
    }

    // Prefer L1 cache over shared memory (because we are not using shared memory)
    cudaFuncSetCacheConfig(matrix_multiply_internal_cu<T, U, decltype(T{} * U{})>, cudaFuncCachePreferL1);

    size_t streamIdx = 0;
    for (uint32_t i = 0; i <= shiftDown; i++) {
        for (uint32_t j = 0; j <= shiftRight; j++) {
            uint32_t blockSize = std::min(NO_OF_BLOCKS, a.rowsCPU - i * NO_OF_BLOCKS);
            uint32_t noOfThreads = std::min(NO_OF_THREADS, b.colsCPU - j * NO_OF_THREADS);
            noOfThreads = max(1, noOfThreads);

            matrix_multiply_internal_cu<<<blockSize, noOfThreads, 0, streams[streamIdx]>>>(
                    a.data, a.colsCPU,
                    b.data, b.colsCPU,
                    result.data,
                    i * NO_OF_BLOCKS, j * NO_OF_THREADS);

            streamIdx++;
        }
    }

    for (auto &i: streams) {
        cudaStreamSynchronize(i);
        cudaStreamDestroy(i);
    }

    return result;
}

template<typename T, typename U>
auto matmult(const Matrix_cu<T> &a, const Matrix_cu<U> &b) {
    return matrix_multiply(a, b);
}

#endif //BACKPROPAGATION_MATRIX_CUH