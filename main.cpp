#ifdef TESTING
#include <iostream>
#include "tests/matmul_unit_test.h"

int main() {
    std::cout << std::boolalpha;
    bool test;
    std::cout << (test = matmul_unit_test());
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

namespace fs = std::filesystem;

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
    auto *data = result.data;
    auto *sum = new double[m.cols]{};
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

/*
 * def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
 */
Matrix<double> one_hot(Matrix<double> const &m) {

}

std::tuple<Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>> forward_propagation(Matrix<double> &X,
                                                                                               Matrix<double> &W1,
                                                                                               Matrix<double> &b1,
                                                                                               Matrix<double> &W2,
                                                                                               Matrix<double> &b2) {
    auto Z1 = matmult(W1, X) + b1;
    auto A1 = relu(Z1);
    auto Z2 = matmult(W2, A1) + b2;
    auto A2 = softmax(Z2);
    return {Z1, A1, Z2, A2};
}

int main() {
    int n = 10, m = 5, a = 2;
    auto *dataP = new double[n * m];
    auto *test_dataP = new double[a * m];

    for (int i = 0; i < 10 * 5; i++) {
        dataP[i] = i;
    }
    for (int i = 0; i < 2 * 5; i++) {
        test_dataP[i] = i;
    }


    Matrix<double> data = CsvToMatrix<double>("./Data/mnist_train.csv", 0, 0, 10);
    Matrix<double> test_data = CsvToMatrix<double>("./Data/mnist_test.csv", 0, 0, 10);

    data.t();
    test_data.t();

    Matrix<double> const Y_train(1, n, data[0]);
    Matrix<double> const X_train(m - 1, n, data[1]);

    Matrix<double> const Y_test(1, a, test_data[0]);
    Matrix<double> const X_test(m - 1, a, test_data[1]);

    Matrix<double> W1(10, 784);
    Matrix<double> b1(10, 1);
    Matrix<double> W2(10, 10);
    Matrix<double> b2(10, 1);

    W1.fillRandom(-0.5, 0.5);
    b1.fillRandom(-0.5, 0.5);
    W2.fillRandom(-0.5, 0.5);
    b2.fillRandom(-0.5, 0.5);


    delete[] dataP;
    delete[] test_dataP;
}

#endif