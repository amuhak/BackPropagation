// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_CUH
#define BACKPROPAGATION_MATRIX_CUH

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <sys/types.h>
#include "Matrix.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_types.h"
#include "driver_types.h"

constexpr long double EPSILON = 0.00001;
constexpr size_t SHIFT_SIZE = 1024;
size_t NO_OF_THREADS{};
size_t NO_OF_BLOCKS{};
size_t *NO_OF_THREADS_GPU{};
size_t *NO_OF_BLOCKS_GPU{};
using uint32_t = unsigned int;
using ulong = unsigned long;

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
        cudaGetDevice(0);
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, 0);
        NO_OF_THREADS = deviceProp.maxThreadsPerBlock;
        NO_OF_BLOCKS = deviceProp.maxThreadsDim[0];
        cudaMalloc(&NO_OF_THREADS_GPU, sizeof(size_t));
        cudaMalloc(&NO_OF_BLOCKS_GPU, sizeof(size_t));
        cudaMemcpy(NO_OF_THREADS_GPU, &NO_OF_THREADS, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(NO_OF_BLOCKS_GPU, &NO_OF_BLOCKS, sizeof(size_t), cudaMemcpyHostToDevice);
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
     * Constructor for casting.
     * @tparam U Type of other matrix
     * @param other Other matrix
     */
    template<class U>
    explicit Matrix_cu(const Matrix<U> &other) {
        init(other.rows, other.cols);
        Matrix<T> temp(other);
        cudaMemcpy(data, temp.data, sizeof(T) * temp.length, cudaMemcpyHostToDevice);
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
    Matrix_cu &operator=(Matrix<T> &&other) noexcept {
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
        return forEach_new([scalar](T val) { return val / scalar; }, *this);
    }

    Matrix_cu<T> operator*(T scalar) {
        return forEach_new([scalar](T val) { return val * scalar; }, *this);
    }

    Matrix_cu<T> operator+(T scalar) {
        return forEach_new([scalar](T val) { return val + scalar; }, *this);
    }

    Matrix_cu<T> operator-(T scalar) {
        return forEach_new([scalar](T val) { return val - scalar; }, *this);
    }

    Matrix<T> toCPU() {
        T *dataCPU = new T[lengthCPU];
        cudaMemcpy(dataCPU, data, lengthCPU * sizeof(T), cudaMemcpyDeviceToHost);
        Matrix<T> result(rowsCPU, colsCPU, dataCPU);
        delete[] dataCPU;
        return result;
    }

    Matrix_cu<T> operator+(Matrix<T> &other) {
        return Matrix_cu<T>(toCPU() + other);
    }

    Matrix_cu<T> operator-(Matrix<T> &other) {
        return Matrix_cu<T>(toCPU() - other);
    }

    Matrix_cu<T> operator*(Matrix<T> &other) {
        return matrix_multiply(*this, other);
    }

    void fillRandom() {
        Matrix<T> temp(rowsCPU, colsCPU);
        temp.fillRandom();
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

template<typename T>
__global__
void forEach_internal(const std::function<void(T (T))> &func, T *data, const size_t length, const size_t shift) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < length; i += shift) {
        data[i] = func(data[i]);
    }
}

template<typename T>
__global__
void
forEach_new_internal(const std::function<void(T (T))> &func, T *out, T *data, const size_t length, const size_t shift) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < length; i += shift) {
        out[i] = func(data[i]);
    }
}

template<typename T>
void forEach(const std::function<void(T (T))> &func, Matrix_cu<T> &input) {
    int noOfThreads = std::min(NO_OF_THREADS, input.lengthCPU);
    int noOfBlocks = std::min(NO_OF_BLOCKS, (input.lengthCPU + noOfThreads - 1) / noOfThreads);
    forEach_internal<<<noOfBlocks, noOfThreads>>>(func, input.data, input.lengthCPU, input.noOfThreads);
}

template<typename T>
Matrix_cu<T> forEach_new(const std::function<void(T (T))> &func, Matrix_cu<T> &input) {
    Matrix_cu<T> result(input.rowsCPU, input.colsCPU);
    int noOfThreads = std::min(NO_OF_THREADS, input.lengthCPU);
    int noOfBlocks = std::min(NO_OF_BLOCKS, (input.lengthCPU + noOfThreads - 1) / noOfThreads);
    forEach_new_internal<<<noOfBlocks, noOfThreads>>>(func, result.data, input.data, input.lengthCPU,
                                                      input.noOfThreads);
}

template<typename T, typename U, typename V>
__global__
void
matrix_multiply_internal_cu(T *__restrict__ a, uint32_t aCols,
                            U *__restrict__ b, uint32_t bCols,
                            V *__restrict__ c,
                            uint16_t block = 0, uint16_t threads = 0,
                            uint16_t shiftDown = 0, uint16_t shiftRight = 0) {
    decltype(T{} * U{}) ans = 0;      // auto may not give you the right type here

    shiftDown *= (block);
    shiftRight *= (threads);
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

    /*
     * size_t NO_OF_THREADS{};
     * size_t NO_OF_BLOCKS{};
     */

    std::cout << "NO_OF_THREADS " << NO_OF_THREADS << " NO_OF_BLOCKS " << NO_OF_BLOCKS << std::endl;

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
    for (uint16_t i = 0; i <= shiftDown; i++) {
        for (uint16_t j = 0; j <= shiftRight; j++) {
            uint32_t blockSize = std::min(NO_OF_BLOCKS, a.rowsCPU - i * NO_OF_BLOCKS);
            uint32_t noOfThreads = std::min(NO_OF_THREADS, b.colsCPU - j * NO_OF_THREADS);
            noOfThreads = max(1, noOfThreads);

            matrix_multiply_internal_cu<<<blockSize, noOfThreads, 0, streams[streamIdx]>>>(
                    a.data, a.colsCPU,
                    b.data, b.colsCPU,
                    result.data,
                    NO_OF_BLOCKS, NO_OF_THREADS,
                    i, j);

            streamIdx++;
        }
    }

    for (auto &i: streams) {
        cudaStreamSynchronize(i);
        cudaStreamDestroy(i);
    }

    return result;
}

#endif //BACKPROPAGATION_MATRIX_CUH