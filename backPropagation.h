//
// Created by amuly on 7/2/2024.
//

#ifndef BACKPROPAGATION_BACKPROPAGATION_H
#define BACKPROPAGATION_BACKPROPAGATION_H

#include <iostream>
#include "Matrix.h"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <numeric>
#include <vector>
#include <utility>

constexpr long double RAND_RANGE_START = -0.5;
constexpr long double RAND_RANGE_END = 0.5;

template<typename T>
class backPropagation {
public:
    // Matrix<T> W1, b1, W2, b2, Z1, A1, Z2, A2, dW1, dW2;
    std::vector<Matrix<T>> weightsAndBiases;
    std::vector<Matrix<T>> activations;
    std::vector<std::pair<Matrix<T>, T>> derivatives;
    // T db2, db1;
    double alpha;

    /**
     * Constructor for the backPropagation class
     * In put a vector of pairs of integers.
     * In each pair, the first integer is the number input neurons and the second integer is the number of output neurons.
     * The vectors for bias will automatically be created depending on the number of output neurons.
     */
    backPropagation(const std::vector<std::pair<int, int>> &layers, double al) : alpha(al) {
        for (const auto &[first, second]: layers) {
            Matrix<T> weight(second, first);
            Matrix<T> bias(second, 1);
            weight.fillRandom(RAND_RANGE_START, RAND_RANGE_END);
            bias.fillRandom(RAND_RANGE_START, RAND_RANGE_END);
            weightsAndBiases.push_back(std::move(weight));
            weightsAndBiases.push_back(std::move(bias));
            Matrix<T> Z{};
            Matrix<T> A{};  // Filling up activations with objects of Matrix class so that they can be used later
            activations.push_back(std::move(Z));
            activations.push_back(std::move(A));
            Matrix<T> d{};  // Filling up derivatives with objects of Matrix class so that they can be used later
            derivatives.emplace_back(d, static_cast<T>(0));
        }
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

    void forward_propagation(Matrix<T> &X) {
        // Apply relu too all layers except the last one
        size_t i = 0;
        for (; i < weightsAndBiases.size() - 2; i += 2) {
            Matrix<T> &Z = activations[i];
            Matrix<T> &A = activations[i + 1];
            Matrix<T> &W = weightsAndBiases[i];
            Matrix<T> &b = weightsAndBiases[i + 1];
            Z = matmult(W, X) + b;
            A = relu(Z);
        }

        // Now apply softmax to the last layer
        Matrix<T> &Z = activations[i];
        Matrix<T> &A = activations[i + 1];
        Matrix<T> &W = weightsAndBiases[i];
        Matrix<T> &b = weightsAndBiases[i + 1];
        Z = matmult(W, activations[i - 1]) + b;
        A = softmax(Z);
    }

    void backward_prop(Matrix<double> &X, Matrix<double> &Y) {

        auto one_hot_Y = one_hot(Y);

        Matrix<T> mult; // Needs to be passed on to the next iteration

        int i = weightsAndBiases.size() - 2;

        {
            Matrix<T> &A = activations[i + 1];
            Matrix<T> &dW = derivatives[i / 2].first;
            T &db = derivatives[i / 2].second;
            Matrix<T> &W = weightsAndBiases[i];

            Matrix<T> dZ = A - one_hot_Y;
            dW = matmult(dZ, activations[i - 1].t()) * (1.0 / Y.rows);
            db = (1.0 / Y.rows) * dZ.sum();
            mult = matmult(W.t(), dZ);
        }

        i -= 2;

        for (; i > 0; i -= 2) {
            Matrix<T> &Z = activations[i];
            Matrix<T> &W = weightsAndBiases[i];
            Matrix<T> &dW = derivatives[i / 2].first;
            T &db = derivatives[i / 2].second;

            auto der = relu_derivative(Z);
            auto dZ = mult * der;
            dW = matmult(dZ, activations[i - 1].t()) * (1.0 / Y.rows);
            db = (1.0 / Y.rows) * dZ.sum();
            mult = matmult(W.t(), dZ);
        }

        Matrix<T> &Z1 = activations[0];
        T &db1 = derivatives[0].second;
        Matrix<T> &dW1 = derivatives[0].first;

        auto der = relu_derivative(Z1);
        auto dZ1 = mult * der;
        dW1 = matmult(dZ1, X) * (1.0 / Y.rows);
        db1 = (1.0 / Y.rows) * std::accumulate(dZ1.data, dZ1.data + dZ1.length, 0.0);
    }

    void update_params() {
        size_t i = 0;
        Matrix<T> t;
        for (; i < weightsAndBiases.size(); i += 2) {
            Matrix<T> dW = derivatives[i / 2].first;
            T db = derivatives[i / 2].second;
            Matrix<T> &W = weightsAndBiases[i];
            Matrix<T> &b = weightsAndBiases[i + 1];
            t = (dW * alpha);
            W = W - t;
            b = b - (db * alpha);
        }
    }

    Matrix<T> get_predictions() {
        Matrix<T> &A = activations[activations.size() - 1];
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

    double get_accuracy(Matrix<T> const &predictions, Matrix<T> const &Y) {
        double sum = 0;
        for (int i = 0; i < Y.rows; i++) {
            if (predictions[0][i] == Y[0][i]) {
                sum++;
            }
        }
        return sum / Y.length;
    }
};

#endif //BACKPROPAGATION_BACKPROPAGATION_H
