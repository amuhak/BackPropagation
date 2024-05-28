#ifdef TESTING
#include <iostream>
#include <cstdlib>
#include "tests/matmul_unit_test.h"

int main(int argc, char* argv[]) {
    std::cout << "Pass the size of the matrix as an comand line argument to the program" << std::endl;
    int no = (1U << 10U) + 1;
    if (argc < 2) {
        std::cout << "Using default size of matrix: " << no << std::endl;
    } else {
        no = std::atoi(argv[1]);
        std::cout << "Using size of matrix: " << no << std::endl;
    }
    std::cout << std::boolalpha;
    bool test;
    std::cout << (test = matmul_unit_test(no));
    if (!test) {
        return 69;
    }
}
#else

#include <iostream>
#include "Matrix.h"
#include "CSV.h"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <numeric>
#include <chrono>
#include <iomanip>

Matrix<double> relu(Matrix<double> const &m) {
    Matrix<double> result(m);
    auto *data = result.data;
    for (int i = 0; i < m.length; i++) {
        data[i] = std::max(0.0, data[i]);
    }
    return result;
}

Matrix<double> relu_derivative(Matrix<double> const &m) {
    Matrix<double> result(m);
    auto *data = result.data;
    for (int i = 0; i < m.length; i++) {
        data[i] = data[i] > 0 ? 1 : 0;
    }
    return result;
}

Matrix<double> softmax(Matrix<double> const &m) {
    Matrix<double> result(m);
    auto *sum = new double[m.cols];
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
Matrix<double> one_hot(Matrix<double> const &m) {
    int const max = (int) *std::max_element(m.data, m.data + m.length) + 1;
    Matrix<int> ans(max, m.rows);
    ans.fill0();
    double *data = m.data;
    for (int i = 0; i < m.length; i++) {
        ans[(int) data[i]][i] = 1;
    }
    return ans;
}

std::tuple<Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>>
forward_propagation(Matrix<double> &W1, Matrix<double> &b1,
                    Matrix<double> &W2, Matrix<double> &b2,
                    Matrix<double> &X) {
    auto Z1 = matmult(W1, X) + b1;
    auto A1 = relu(Z1);
    auto Z2 = matmult(W2, A1) + b2;
    auto A2 = softmax(Z2);
    return {Z1, A1, Z2, A2};
}

std::tuple<Matrix<double>, double, Matrix<double>, double>
backward_prop(Matrix<double> &Z1, Matrix<double> &A1,
              Matrix<double> &Z2, Matrix<double> &A2,
              Matrix<double> &W1, Matrix<double> &W2,
              Matrix<double> &X, Matrix<double> &Y) {

    auto one_hot_Y = one_hot(Y);
    auto dZ2 = A2 - one_hot_Y;
    auto dW2 = matmult(dZ2, A1.t()) * (1.0 / Y.rows);
    auto db2 = (1.0 / Y.rows) * std::accumulate(dZ2.data, dZ2.data + dZ2.length, 0.0);
    std::cout << db2 << std::endl;
    auto der = relu_derivative(Z1);
    auto mutl = matmult(W2.t(), dZ2);
    auto dZ1 = mutl * der;
    auto dW1 = matmult(dZ1, X.t()) * (1.0 / Y.rows);
    auto db1 = (1.0 / Y.rows) * std::accumulate(dZ1.data, dZ1.data + dZ1.length, 0.0);
    return {dW1, db1, dW2, db2};
}

void update_params(Matrix<double> &W1, Matrix<double> &b1,
                   Matrix<double> &W2, Matrix<double> &b2,
                   Matrix<double> &dW1, double &db1,
                   Matrix<double> &dW2, double &db2,
                   double alpha) {
    auto t = (dW1 * alpha);
    W1 = W1 - t;
    b1 = b1 - (db1 * alpha);
    auto tt = (dW2 * alpha);
    W2 = W2 - tt;
    b2 = b2 - (db2 * alpha);
}

Matrix<double> get_predictions(Matrix<double> const &A2) {
    Matrix<double> result(1, A2.cols);
    for (int i = 0; i < A2.cols; i++) {
        double max = A2[0][i];
        int index = 0;
        for (int j = 1; j < A2.rows; j++) {
            if (A2[j][i] > max) {
                max = A2[j][i];
                index = j;
            }
        }
        result[0][i] = index;
    }
    return result;
}

/*
 * def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
 */

double get_accuracy(Matrix<double> const &predictions, Matrix<double> const &Y) {
    double sum = 0;
    for (int i = 0; i < Y.cols; i++) {
        if (predictions[0][i] == Y[0][i]) {
            sum++;
        }
    }
    return sum / Y.length;
}

int main() {
    int trainLen = 60000, testLen = 10000, imgSize = 784;

    Matrix<double> data = CsvToMatrix<double>("./Data/mnist_train.csv");
    Matrix<double> test_data = CsvToMatrix<double>("./Data/mnist_test.csv");


    data = data.t();
    test_data = test_data.t();

    Matrix<double> Y_train(trainLen, 1, data[0]);
    std::cout << Y_train.sum() << std::endl;
    Matrix<double> X_train(imgSize, trainLen, data[1]);

    Matrix<double> Y_test(testLen, 1, test_data[0]);
    Matrix<double> X_test(imgSize, testLen, test_data[1]);

    X_train = X_train / 255.0; // Normalizing the data
    X_test = X_test / 255.0;   // Normalizing the data

    /*
    Matrix<double> W1(10, 784);
    Matrix<double> b1(10, 1);
    Matrix<double> W2(10, 10);
    Matrix<double> b2(10, 1);

    W1.fillRandom(-0.5, 0.5);
    b1.fillRandom(-0.5, 0.5);
    W2.fillRandom(-0.5, 0.5);
    b2.fillRandom(-0.5, 0.5);
    */

    Matrix<double> W1 = CsvToMatrix<double>("./Data/W1.csv");
    Matrix<double> b1 = CsvToMatrix<double>("./Data/b1.csv");
    Matrix<double> W2 = CsvToMatrix<double>("./Data/W2.csv");
    Matrix<double> b2 = CsvToMatrix<double>("./Data/b2.csv");
    std::cout << std::setprecision(10);
    std::cout << std::accumulate(W1.data, W1.data + W1.length, 0.0) << std::endl;

    double const alpha = 0.10;
    int const iterations = 500;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto [Z1, A1, Z2, A2] = forward_propagation(W1, b1, W2, b2, X_train);
        auto [dW1, db1, dW2, db2] = backward_prop(Z1, A1, Z2, A2, W1, W2, X_train, Y_train);
        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
        if (i % 10 == 0) {
            std::cout << "Iteration: " << i << std::endl;
            auto predictions = get_predictions(A2);
            std::cout << get_accuracy(predictions, Y_train) << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;
}

#endif