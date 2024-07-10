#ifndef BACKPROPAGATION_BACKPROPAGATION_H
#define BACKPROPAGATION_BACKPROPAGATION_H

#include "Matrix.h"
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

constexpr long double RAND_RANGE_START = -0.5;
constexpr long double RAND_RANGE_END = 0.5;

template<typename T>
class backPropagation {
private:
    T alpha;
    Matrix<T> dataT;
    size_t no_of_layers;
    std::vector<std::pair<Matrix<T>, Matrix<T>>> weightsAndBiases;
    std::vector<std::pair<Matrix<T>, Matrix<T>>> activations;
    std::vector<std::pair<Matrix<T>, T>> derivatives;
    Matrix<T> InData, InAns;
public:


    /**
     * Constructor for the backPropagation class
     * In put a vector of pairs of integers.
     * In each pair, the first integer is the number input neurons and the second integer is the number of output neurons.
     * The vectors for bias will automatically be created depending on the number of output neurons.
     */
    backPropagation(const std::vector<std::pair<int, int>> &layers, T al) : alpha(al),
                                                                            no_of_layers(layers.size()) {
        for (const auto &[first, second]: layers) {
            Matrix<T> weight(second, first);
            Matrix<T> bias(second, 1);
            weight.fillRandom(RAND_RANGE_START, RAND_RANGE_END);
            bias.fillRandom(RAND_RANGE_START, RAND_RANGE_END);
            weightsAndBiases.emplace_back(weight, bias);
            Matrix<T> Z{};
            Matrix<T> A{};  // Filling up activations with objects of Matrix class so that they can be used later
            activations.emplace_back(Z, A);
            Matrix<T> d{};  // Filling up derivatives with objects of Matrix class so that they can be used later
            derivatives.emplace_back(d, static_cast<T>(0));
        }
    }

    /**
     * Setters for the data
     * @param X Input data
     * @param Y Answers for the input data
     */
    void set_data(Matrix<T> &X, Matrix<T> const &Y) {
        InData = X;
        dataT = X.t();
        InAns = Y;
    }

    Matrix<T> relu(Matrix<T> const &m) {
        Matrix<T> result(m);
        auto *data = result.data;
        for (int i = 0; i < m.length; i++) {
            data[i] = std::max(0.0, data[i]);
        }
        return result;
    }

    Matrix<T> relu_derivative(Matrix<T> const &m) {
        Matrix<T> result(m);
        auto *data = result.data;
        for (int i = 0; i < m.length; i++) {
            data[i] = data[i] > 0 ? 1 : 0;
        }
        return result;
    }

    Matrix<T> softmax(Matrix<T> const &m) {
        Matrix<T> result(m);
        auto *sum = new T[m.cols]{};
        auto *data = result.data;
        for (int i = 0; i < m.length; i++) {
            data[i] = std::exp(data[i]);
            sum[i % m.cols] += data[i];
        }
        for (int i = 0; i < m.length; i++) {
            data[i] /= sum[i % m.cols];
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
    Matrix<T> one_hot(Matrix<T> const &m) {
        int const max = (int) *std::max_element(m.data, m.data + m.length) + 1;
        Matrix<T> ans(max, m.rows);
        ans.fill0();
        T *data = m.data;
        for (int i = 0; i < m.length; i++) {
            ans[(int) data[i]][i] = 1;
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

        Matrix<T> one_hot_Y = one_hot(InData);

        Matrix<T> mult; // Needs to be passed on to the next iteration

        size_t i = no_of_layers - 1;

        {
            auto &[Z, A] = activations[i];
            auto &[dW, db] = derivatives[i];
            auto &[W, b] = weightsAndBiases[i];

            Matrix<T> dZ = A - one_hot_Y;
            dW = matmult(dZ, activations[i - 1].second.t()) * (1.0 / InData.rows);
            db = (1.0 / InData.rows) * dZ.sum();
            mult = matmult(W.t(), dZ);
        }

        i--;

        for (; i > 0; i--) {
            auto &[Z, A] = activations[i];
            auto &[dW, db] = derivatives[i];
            auto &[W, b] = weightsAndBiases[i];

            Matrix<T> der = relu_derivative(Z);
            Matrix<T> dZ = mult * der;
            dW = matmult(dZ, activations[i - 1].second.t()) * (1.0 / InData.rows);
            db = (1.0 / InData.rows) * dZ.sum();
            mult = matmult(W.t(), dZ);
        }
        auto &[Z, A] = activations[0];
        auto &[dW, db] = derivatives[0];

        Matrix<T> der = relu_derivative(Z);
        Matrix<T> dZ1 = mult * der;
        dW = matmult(dZ1, dataT) * (1.0 / InData.rows);
        db = (1.0 / InData.rows) * (dZ1.sum());
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

    Matrix<T> get_predictions() {
        Matrix<T> &A = activations.back().second;
        Matrix<T> result(A.cols, 1);
        for (int i = 0; i < A.cols; i++) {
            T max = A[0][i];
            int maxLocation = 0;
            for (int j = 1; j < A.rows; j++) {
                if (max < A[j][i]) {
                    max = A[j][i];
                    maxLocation = j;
                }
            }
            result[0][i] = maxLocation;
        }
        return result;
    }

    Matrix<T> get_predictions(const Matrix<T> &A) {
        Matrix<T> result(A.cols, 1);
        for (int i = 0; i < A.cols; i++) {
            T max = A[0][i];
            int maxLocation = 0;
            for (int j = 1; j < A.rows; j++) {
                if (max < A[j][i]) {
                    max = A[j][i];
                    maxLocation = j;
                }
            }
            result[0][i] = maxLocation;
        }
        return result;
    }

    Matrix<T> forward_propagation_no_update(const Matrix<T> &X) {
        size_t i = 0;
        Matrix<T> last_A;
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

    Matrix<T> evaluate(const Matrix<T> &X) {
        Matrix<T> A = forward_propagation_no_update(X);
        return get_predictions(A);
    }

    long double get_accuracy(Matrix<T> const &predictions, Matrix<T> const &Y) {
        long double sum = 0;
        for (int i = 0; i < Y.rows; i++) {
            sum += predictions[0][i] == Y[0][i];
        }
        return sum / Y.length;
    }
};

#endif //BACKPROPAGATION_BACKPROPAGATION_H
