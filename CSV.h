#ifndef BACKPROPAGATION_CSV_H
#define BACKPROPAGATION_CSV_H

#include "Matrix.h"
#include <string>

template<typename T>
Matrix<T> CsvToMatrix(const std::string& filename, int skipRow = 0, int skipCol = 0, const char& delimiter = ',');

#endif //BACKPROPAGATION_CSV_H
