// main.cu
#include <iostream>
#include "Matrix.h"

__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}

int main() {
    Matrix<float> m(-3, 3);
    m.print();
    return 0;
}
