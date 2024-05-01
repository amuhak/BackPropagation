#include <iostream>

#ifdef TESTING

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

int main() {
    std::cout << "Hello world";
}

#endif