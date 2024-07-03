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
#include <iomanip>

int main() {
    constexpr int trainLen = 60000, testLen = 10000, imgSize = 784;

    Matrix<double> data = CsvToMatrix<double>("./Data/mnist_train.csv");
    Matrix<double> test_data = CsvToMatrix<double>("./Data/mnist_test.csv");


    data = data.t();
    test_data = test_data.t();

    Matrix<double> Y_train(trainLen, 1, data[0]);
    Matrix<double> X_train(imgSize, trainLen, data[1]);

    Matrix<double> Y_test(testLen, 1, test_data[0]);
    Matrix<double> X_test(imgSize, testLen, test_data[1]);

    X_train = X_train / 255.0; // Normalizing the data
    X_test = X_test / 255.0;   // Normalizing the data

    Matrix<double> W1(10, 784);
    Matrix<double> b1(10, 1);
    Matrix<double> W2(10, 10);
    Matrix<double> b2(10, 1);


    W1.fillRandom(-0.5, 0.5);
    b1.fillRandom(-0.5, 0.5);
    W2.fillRandom(-0.5, 0.5);
    b2.fillRandom(-0.5, 0.5);

    /*
    Matrix<double> W1 = CsvToMatrix<double>("./Data/W1.csv");
    Matrix<double> b1 = CsvToMatrix<double>("./Data/b1.csv");
    Matrix<double> W2 = CsvToMatrix<double>("./Data/W2.csv");
    Matrix<double> b2 = CsvToMatrix<double>("./Data/b2.csv");
    */

    std::cout << std::setprecision(10);

    double const alpha = 0.10;
    int const iterations = 500;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> layers = {{784, 25},
                                               {25, 10}};
    backPropagation<double> bp(layers, alpha);
    auto X_train_t = X_train.t();
    for (int i = 0; i < iterations; i++) {
        bp.forward_propagation(X_train);
        bp.backward_prop(X_train_t, Y_train);
        bp.update_params();
        if (i % 10 == 0) {
            std::cout << "Iteration: " << i << std::endl;
            auto predictions = bp.get_predictions();
            std::cout << "Accuracy: " << bp.get_accuracy(predictions, Y_train) << std::endl;
            /*
            MatrixToCsv("./Data/W1_" + std::to_string(i) + "_.csv", W1);
            MatrixToCsv("./Data/b1_" + std::to_string(i) + "_.csv", b1);
            MatrixToCsv("./Data/W2_" + std::to_string(i) + "_.csv", W2);
            MatrixToCsv("./Data/b2_" + std::to_string(i) + "_.csv", b2);
             */
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s"
              << std::endl;
}

#endif