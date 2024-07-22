//
// Created by amuly on 7/19/2024.
//

#ifndef BACKPROPAGATION_BACKPROPAGATION_CUH
#define BACKPROPAGATION_BACKPROPAGATION_CUH

#include "Matrix.h"
#include "Matrix.cuh"
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>
#include "device_launch_parameters.h"
#include "device_types.h"

constexpr long double RAND_RANGE_START = -0.5;
constexpr long double RAND_RANGE_END = 0.5;

template<typename T>
__global__ void relu_kernel(T *data, size_t length, size_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < length; i += shift) {
        data[i] = static_cast<T>(0) < data[i] ? data[i] : static_cast<T>(0);
    }
}

template<typename T>
__global__ void relu_derivative_kernel(T *data, size_t length, size_t shift) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < length; i += shift) {
        data[i] = data[i] > 0 ? 1 : 0;
    }
}

template<typename T>
class backPropagation {
private:
    T alpha;
    Matrix_cu<T> dataT;
    size_t no_of_layers;
    std::vector<std::pair<Matrix_cu<T>, Matrix_cu<T>>> weightsAndBiases;
    std::vector<std::pair<Matrix_cu<T>, Matrix_cu<T>>> activations;
    std::vector<std::pair<Matrix_cu<T>, T>> derivatives;
    Matrix_cu<T> InData, InAns;
public:


    /**
     * Constructor for the backPropagation class
     * In put a vector of pairs of integers.
     * In each pair, the first integer is the number input neurons and the second integer is the number of output neurons.
     * The vectors for bias will automatically be created depending on the number of output neurons.
     */
    backPropagation(const std::vector<std::pair<size_t, size_t>> &layers, T al) : alpha(al),
                                                                                  no_of_layers(layers.size()) {
        for (const auto &[first, second]: layers) {
            Matrix_cu<T> weight(second, first);
            Matrix_cu<T> bias(second, 1);
            weight.fillRandom(RAND_RANGE_START, RAND_RANGE_END);
            bias.fillRandom(RAND_RANGE_START, RAND_RANGE_END);
            weightsAndBiases.emplace_back(weight, bias);
            Matrix_cu<T> Z{};
            Matrix_cu<T> A{};  // Filling up activations with objects of Matrix class so that they can be used later
            activations.emplace_back(Z, A);
            Matrix_cu<T> d{};  // Filling up derivatives with objects of Matrix class so that they can be used later
            derivatives.emplace_back(d, static_cast<T>(0));
        }
    }

    /**
     * Setters for the data
     * @param X Input data
     * @param Y Answers for the input data
     */
    void set_data(Matrix_cu<T> &X, Matrix_cu<T> const &Y) {
        InData = X;
        dataT = X.t();
        InAns = Y;
    }

    Matrix_cu<T> relu(Matrix_cu<T> &m) {
        Matrix_cu<T> ans(m);
        auto *data = ans.data;
        relu_kernel<<<NO_OF_THREADS, NO_OF_BLOCKS>>>(data, ans.lengthCPU, NO_OF_THREADS * NO_OF_BLOCKS);
        return ans;
    }

    Matrix_cu<T> relu_derivative(Matrix_cu<T> &m) {
        Matrix_cu<T> ans(m);
        auto *data = ans.data;
        relu_derivative_kernel<<<NO_OF_THREADS, NO_OF_BLOCKS>>>(data, ans.lengthCPU, NO_OF_THREADS * NO_OF_BLOCKS);
        return ans;
    }

    Matrix_cu<T> softmax(Matrix_cu<T> &m) {
        Matrix<T> result = m.toCPU();
        auto *sum = new T[result.cols]{};
        auto *data = result.data;
        for (size_t i = 0; i < result.length; i++) {
            data[i] = std::exp(data[i]);
            sum[i % result.cols] += data[i];
        }
        for (size_t i = 0; i < result.length; i++) {
            data[i] /= sum[i % result.cols];
        }
        delete[] sum;
        return result;
    }

/**
 * We are assuming that the matrix is a column matrix. This function will catastrophically fail if the matrix is
 * not a column matrix. Please make sure that the matrix is a column matrix before calling this function.
 * @param m The matrix to be converted to one hot encoding
 * @return The one hot encoded matrix
 */
    Matrix<T> one_hot(Matrix_cu<T> &a) {
        Matrix<T> m = a.toCPU();
        const size_t max = (size_t) *std::max_element(m.data, m.data + m.length) + 1;
        Matrix<T> ans(max, m.rows);
        ans.fill0();
        T *data = m.data;
        for (size_t i = 0; i < m.length; i++) {
            ans[(size_t) data[i]][i] = 1;
        }
        return ans;
    }

    void forward_propagation() {
        // Apply relu too all layers except the last one
        size_t i = 0;

        {
            auto &[Z, A] = activations[i];
            auto &[W, b] = weightsAndBiases[i];
            Z = matmult(W, InData) + b;
            A = relu(Z);
        }
        i++;
        for (; i < no_of_layers - 1; i++) {
            auto &[Z, A] = activations[i];
            auto &[W, b] = weightsAndBiases[i];
            Z = matmult(W, activations[i - 1].second) + b;
            A = relu(Z);
        }

        // Now apply softmax to the last layer
        auto &[Z, A] = activations[i];
        auto &[W, b] = weightsAndBiases[i];
        Z = matmult(W, activations[i - 1].second) + b;
        A = softmax(Z);
    }

    void backward_prop() {

        Matrix_cu<T> hot = one_hot(InAns);

        Matrix_cu<T> product; // Needs to be passed on to the next iteration

        size_t i = no_of_layers - 1;

        {
            auto &[Z, A] = activations[i];
            auto &[dW, db] = derivatives[i];
            auto &[W, b] = weightsAndBiases[i];

            Matrix_cu<T> dZ = A - hot;
            dW = matmult(dZ, activations[i - 1].second.t()) * (1.0 / (double) InAns.rowsCPU);
            db = (1.0 / (double) InAns.rowsCPU) * dZ.sum();
            product = matmult(W.t(), dZ);
        }

        i--;

        for (; i > 0; i--) {
            auto &[Z, A] = activations[i];
            auto &[dW, db] = derivatives[i];
            auto &[W, b] = weightsAndBiases[i];

            Matrix_cu<T> der = relu_derivative(Z);
            Matrix_cu<T> dZ = product * der;
            dW = matmult(dZ, activations[i - 1].second.t()) * (1.0 / (double) InAns.rowsCPU);
            db = (1.0 / (double) InAns.rowsCPU) * dZ.sum();
            product = matmult(W.t(), dZ);
        }
        auto &[Z, A] = activations[0];
        auto &[dW, db] = derivatives[0];

        Matrix_cu<T> der = relu_derivative(Z);
        Matrix_cu<T> dZ1 = product * der;
        dW = matmult(dZ1, dataT) * (1.0 / (double) InAns.rowsCPU);
        db = (1.0 / (double) InAns.rowsCPU) * (dZ1.sum());
    }

    void update_params() {
        for (size_t i = 0; i < no_of_layers; ++i) {
            auto &[W, b] = weightsAndBiases[i];
            auto &[dW, db] = derivatives[i];
            auto temp = (dW * alpha);
            W = W - temp;
            b = b - (db * alpha);
        }
    }

    Matrix_cu<T> get_predictions() {
        Matrix<T> A = activations.back().second.toCPU();
        Matrix<T> result(A.cols, 1);
        for (size_t i = 0; i < A.cols; i++) {
            T max = A[0][i];
            size_t maxLocation = 0;
            for (size_t j = 1; j < A.rows; j++) {
                if (max < A[j][i]) {
                    max = A[j][i];
                    maxLocation = j;
                }
            }
            result[0][i] = (T) maxLocation;
        }
        return result;
    }

    Matrix_cu<T> get_predictions(Matrix_cu<T> &a) {
        Matrix<T> A = a.toCPU();
        Matrix<T> result(A.cols, 1);
        for (size_t i = 0; i < A.cols; i++) {
            T max = A[0][i];
            size_t maxLocation = 0;
            for (size_t j = 1; j < A.rows; j++) {
                if (max < A[j][i]) {
                    max = A[j][i];
                    maxLocation = j;
                }
            }
            result[0][i] = (T) maxLocation;
        }
        return result;
    }

    Matrix_cu<T> forward_propagation_no_update(const Matrix_cu<T> &X) {
        size_t i = 0;
        Matrix_cu<T> last_A;
        // Apply relu too all layers except the last one
        {
            auto &[W, b] = weightsAndBiases[i];
            auto Z = matmult(W, X) + b;
            last_A = relu(Z);
        }

        i++;

        for (; i < no_of_layers - 1; i++) {
            auto &[W, b] = weightsAndBiases[i];
            auto Z = matmult(W, last_A) + b;
            last_A = relu(Z);
        }

        {
            auto &[W, b] = weightsAndBiases[i];
            auto Z = matmult(W, last_A) + b;
            last_A = softmax(Z);
        }

        return last_A;

    }

    Matrix_cu<T> evaluate(const Matrix_cu<T> &X) {
        Matrix_cu<T> A = forward_propagation_no_update(X);
        return get_predictions(A);
    }

    long double get_accuracy(Matrix_cu<T> &predictions, Matrix_cu<T> &y) {
        Matrix<T> pre = predictions.toCPU();
        Matrix<T> Y = y.toCPU();
        long double sum = 0;
        for (size_t i = 0; i < Y.rows; i++) {
            sum += pre[0][i] == Y[0][i];
        }
        return sum / Y.length;
    }
};


#endif //BACKPROPAGATION_BACKPROPAGATION_CUH
