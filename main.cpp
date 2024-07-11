
#ifdef TESTING

#include <iostream>
#include <cstdlib>
#include "tests/matmul_unit_test.h"

int main(int argc, char *argv[]) {
    std::cout << "Pass the size of the matrix as an comand line argument to the program" << std::endl;
    int no = (1U << 11U) + 1;
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
#include "backPropagation.h"
#include <chrono>
#include <vector>
#include <utility>
#include <iomanip>

int main() {
    constexpr int trainLen = 60000, testLen = 10000, imgSize = 784;

    Matrix<double> data = CsvToMatrix<double>("./Data/mnist_train.csv");
    Matrix<double> test_data = CsvToMatrix<double>("./Data/mnist_test.csv");

    data = data.t();
    test_data = test_data.t();

    Matrix<double> const Y_train(trainLen, 1, data[0]);
    Matrix<double> X_train(imgSize, trainLen, data[1]);

    Matrix<double> const Y_test(testLen, 1, test_data[0]);
    Matrix<double> X_test(imgSize, testLen, test_data[1]);

    X_train = X_train / 255.0; // Normalizing the data
    X_test = X_test / 255.0;   // Normalizing the data

    std::cout << std::setprecision(10);

    double const alpha = 0.10;
    int const iterations = 500;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<int, int>> const layers = {{784, 10},
            // {128, 16},
                                                     {10,  10}};
    backPropagation<double> bp(layers, alpha);
    bp.set_data(X_train, Y_train);
    for (int i = 0; i <= iterations; i++) {
        bp.forward_propagation();
        bp.backward_prop();
        bp.update_params();
        if (i % 10 == 0) {
            std::cout << "Iteration: " << i << std::endl;

            auto predictions = bp.get_predictions();
            std::cout << "Accuracy: " << bp.get_accuracy(predictions, Y_train) << std::endl;

            predictions = bp.evaluate(X_test);
            std::cout << "Test Accuracy: " << bp.get_accuracy(predictions, Y_test) << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s"
              << std::endl;
}

#endif