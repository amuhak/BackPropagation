#ifdef TESTING

#include "tests/matmul_unit_test.h"
#include <cstdlib>
#include <sstream>
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "Pass the size of the matrix as an command line argument to the program" << std::endl;
    size_t no = (1U << 12U);
    if (argc < 2) {
        std::cout << "Using default size of matrix: " << no << std::endl;
    } else {
        std::stringstream ss(argv[1]);
        ss >> no;
        std::cout << "Using size of matrix: " << no << std::endl;
    }
    std::cout << std::boolalpha;
    bool test;
    std::cout << (test = matmul_unit_test(no)) << std::endl;
    if (!test) {
        return 69;
    }
    std::cout << "Tests passed" << std::endl << "Running again to get better performance metrics" << std::endl;
    matmul_unit_test(no);
}

#else

#include <iostream>
#include "Matrix.h"
#include "Matrix.cuh"
#include "CSV.h"
#include "backPropagation.cuh"
#include <chrono>
#include <vector>
#include <utility>
#include <iomanip>
#include <cstddef>

int main() {
    constexpr int trainLen = 60000;
    constexpr int testLen = 10000;
    constexpr int imgSize = 784;

    Matrix_cu<double> data = CsvToMatrix<double>("./Data/mnist_train.csv");
    Matrix_cu<double> test_data = CsvToMatrix<double>("./Data/mnist_test.csv");

    data = data.t();
    test_data = test_data.t();

    Matrix_cu<double> Y_train(trainLen, 1, data.toCPU()[0]);
    Matrix_cu<double> X_train(imgSize, trainLen, data.toCPU()[1]);

    Matrix_cu<double> Y_test(testLen, 1, test_data.toCPU()[0]);
    Matrix_cu<double> X_test(imgSize, testLen, test_data.toCPU()[1]);

    X_train = X_train / 255.0; // Normalizing the data
    X_test = X_test / 255.0;   // Normalizing the data

    std::cout << std::setprecision(10);

    double const alpha = 0.10;
    int const iterations = 500;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<size_t, size_t>> const layers = {{784, 10},
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

            predictions = bp.evaluate(X_test).toCPU();
            std::cout << "Test Accuracy: " << bp.get_accuracy(predictions, Y_test) << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s"
              << std::endl;
}

#endif