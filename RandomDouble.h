//
// Created by amuly on 4/7/2024.
//

#ifndef BACKPROPAGATION_RANDOMDOUBLE_H
#define BACKPROPAGATION_RANDOMDOUBLE_H
#include <random>
#include <limits>


class RandomDouble {
public:
    RandomDouble() : rng(std::random_device{}()) {}

    double generate() {
        // Define the range for the random double
        std::uniform_real_distribution<double> distribution(std::numeric_limits<double>::min(),
                                                            std::numeric_limits<double>::max());

        // Generate a random double
        return distribution(rng);
    }

private:
    std::default_random_engine rng;
};
#endif //BACKPROPAGATION_RANDOMDOUBLE_H
