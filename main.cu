#include <iostream>

#ifdef TESTING

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

int main() {
    std::cout << "Hello world";
}

#endif