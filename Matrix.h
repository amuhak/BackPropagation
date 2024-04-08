// Matrix.h
#ifndef BACKPROPAGATION_MATRIX_H
#define BACKPROPAGATION_MATRIX_H


template<typename T>
class Matrix {
    int rows{};
    int cols{};
    int length;
    T *data;

public:
    Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Invalid matrix size");
        }
        this->rows = rows;
        this->cols = cols;
        length = rows * cols;
        this->data = new T[length];
    }

    Matrix(int rows, int cols, T *data) {
        this->rows = rows;
        this->cols = cols;
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Invalid matrix size");
        }
        length = rows * cols;
        this->data = new T[length];
        // Copy data
        for (int i = 0; i < length; i++) {
            this->data[i] = data[i];
        }
    }

    void fill(T value) {
        for (int i = 0; i < length; i++) {
            data[i] = value;
        }
    }

    void set(T *input) {
        for (int i = 0; i < length; i++) {
            this->data[i] = input[i];
        }
    }

    void fill0() {
        fill(0);
    }

    void print() {
        for (int i = 0; i < rows; i++) {
            std::cout << "[ ";
            for (int j = 0; j < cols; j++) {
                std::cout << data[i * cols + j] << ", ";
            }
            std::cout << "]";
            std::cout << std::endl;
        }
    }

    ~Matrix() {
        delete[] this->data;
    }
};


#endif //BACKPROPAGATION_MATRIX_H
