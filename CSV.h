#ifndef BACKPROPAGATION_CSV_H
#define BACKPROPAGATION_CSV_H

#include "Matrix.h"
#include <algorithm>
#include <string>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <cstdlib>

template<typename T>
Matrix<T> CsvToMatrix(const std::string &filename, int skipRow = 0, int skipCol = 0, int noRows = -1,
                      const char delimiter = ',') {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::vector<std::vector<T>> data;
    std::string line;

    while (std::getline(file, line)) {
        if (skipRow > 0) {
            skipRow--;
            continue;
        }
        if (noRows == 0) {
            break;
        }
        noRows--;

        std::vector<T> row;
        const char *linePtr = line.c_str();
        const char *endPtr = linePtr + line.size();
        int temp = skipCol;

        while (linePtr < endPtr) {
            const char *nextDelim = std::find(linePtr, endPtr, delimiter);
            if (temp > 0) {
                temp--;
            } else {
                T value = static_cast<T>(std::strtod(linePtr, nullptr));
                row.push_back(value);
            }
            if (nextDelim == endPtr) {
                break;
            }
            linePtr = nextDelim + 1;
        }
        data.push_back(std::move(row));
    }
    file.close();

    if (data.empty()) {
        throw std::runtime_error("No data found in file");
    }

    Matrix<T> matrix(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            matrix[i][j] = data[i][j];
        }
    }
    return matrix;
}

template<typename T>
void MatrixToCsv(const std::string &filename, const Matrix<T> &matrix, int precision = 6, const char &delimiter = ',') {
    std::ofstream file(filename);
    file.precision(precision);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            file << matrix[i][j];
            if (j < matrix.cols - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
    file.close();
}

#endif //BACKPROPAGATION_CSV_H
