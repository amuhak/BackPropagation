//
// Created by amuly on 4/7/2024.
//

#ifndef BACKPROPAGATION_RANDOMT_H
#define BACKPROPAGATION_RANDOMT_H

#include <random>
#include <limits>
#include <type_traits>

template<typename T>
class RandomT {
public:
    RandomT() : rng(std::random_device{}()) {}

    T generate() {
        std::uniform_int_distribution<int> distribution(std::numeric_limits<int>::min(),
                                                      std::numeric_limits<int>::max());
        return (T) distribution(rng);
    }

private:
    std::default_random_engine rng;
};

#endif //BACKPROPAGATION_RANDOMT_H
