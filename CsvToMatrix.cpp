#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>
#include <sstream>
#include "Matrix.h"
#include "CsvToMatrix.h"


template<typename T>
Matrix<T> CsvToMatrix(const std::string &filename, int skipRow, int skipCol, const char &delimiter) {
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