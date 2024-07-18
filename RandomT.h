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

    /**
     * Generate a random number between min and max (inclusive)
     * @param min minimum value
     * @param max maximum value
     * @return
     */
    T generate(T min, T max) {
        if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> distribution(min, max);
            return distribution(rng);
        } else {
            std::uniform_int_distribution<T> distribution(min, max);
            return distribution(rng);
        }
    }

private:
    std::default_random_engine rng;
};

#endif //BACKPROPAGATION_RANDOMT_H
