#ifndef BACKPROPAGATION_CSV_H
#define BACKPROPAGATION_CSV_H

#include "Matrix.h"
#include <string>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <sys/types.h>
#include <vector>
#include <sstream>

template<typename T>
Matrix<T> CsvToMatrix(const std::string &filename, int skipRow = 0, int skipCol = 0, int noRows = -1,
                      const char &delimiter = ',') {
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
        std::stringstream ss(line);
        std::string cell;
        int temp = skipCol;
        while (std::getline(ss, cell, delimiter)) {
            if (temp > 0) {
                temp--;
                continue;
            }
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    file.close();
    if (data.empty()) {
        throw std::runtime_error("No data found in file");
    }
    Matrix<T> matrix(data.size(), data[0].size());
    for (uint i = 0; i < data.size(); i++) {
        for (uint j = 0; j < data[0].size(); j++) {
            matrix[i][j] = data[i][j];
        }
    }
    return matrix;
}

template<typename T>
void MatrixToCsv(const std::string &filename, const Matrix<T> &matrix, const char &delimiter = ',') {
    std::ofstream file(filename);
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
