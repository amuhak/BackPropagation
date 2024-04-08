// main.cu
#include <iostream>
#include "Matrix.h"

int main() {

    Matrix<float> a(3, 3);
    a[0][0] = 2;
    a[0][1] = 3;
    a[0][2] = 4;
    a[1][0] = 5;
    a[1][1] = 6;
    a[1][2] = 7;
    a[2][0] = 8;
    a[2][1] = 9;
    a[2][2] = 10;
    a.print();
    a.fillRandom();
    a.print();
    Matrix<float> b(3, 3);
    b[0][0] = 11;
    b[0][1] = 12;
    b[0][2] = 13;
    b[1][0] = 14;
    b[1][1] = 15;
    b[1][2] = 16;
    b[2][0] = 17;
    b[2][1] = 18;
    b[2][2] = 19;
    b.print();
    auto c = matrix_multiply(a, b);
    c.print();
    return 0;
}
