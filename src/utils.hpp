#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

namespace utils
{
    // Function to generate a random direction for 2D velocities
    void randomDirection(double &vx, double &vy)
    {
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribution(0.0, 2 * M_PI);

        // Generate random angle
        double angle = distribution(gen);

        // Calculate components
        vx = std::cos(angle);
        vy = std::sin(angle);
    }

    // Sample a velocity from Maxwell-Boltzmann distribution
    double maxwellBoltzmann(double temperature, double mass)
    {
        // Boltzmann constant (in SI units)
        constexpr double kB = 1;

        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        // Generate uniform random number
        double u = distribution(gen);

        // Compute inverse CDF of Maxwell-Boltzmann distribution
        double v = std::sqrt(-2 * kB * temperature / mass * std::log(u));

        return v;
    }

    // Geerate a random number between -1 and 1
    double rand_minus1_to_1()
    {
        double random = (double)rand() / (double)RAND_MAX; // This generates a number between 0 and 1
        random = random * 2.0;                             // This expands the range to 0 to 2
        random = random - 1.0;                             // This shifts the range to -1 to 1
        return random;
    }

} // namespace utils

#endif // UTILS_HPP
