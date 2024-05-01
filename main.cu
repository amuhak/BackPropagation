#include <iostream>

#ifdef TESTING

#include "tests/matmul_unit_test.h"

int main() {
    std::cout << std::boolalpha;
    std::cout << matmul_unit_test();
}

#else

#include <iostream>

int main() {
    std::cout << "Hello world";
}

#endif