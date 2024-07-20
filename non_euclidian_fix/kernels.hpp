#pragma once
#include <cmath>

template<typename T, unsigned DIM>
struct CubicSplineKernel{
    // Support radius
    T supp_radius = 2;
    // Evaluation
    // Modified to match the normalization of the provided W function
    inline T operator()(T r, T h) const {
        const T q = std::abs(r) / h;
        // Normalization constant
        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(2.0 / (3 * h));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 2)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (M_PI * std::pow(h, 3)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, 
              "Only 1, 2, 3 dimensions supported");

        if (q >= 0 && q <= 1)
            return sigma * (1 - 1.5 * std::pow(q, 2) * (1 - q / 2));
        else if (q > 1 && q <= 2)
            return (sigma / 4) * std::pow((2 - q), 3);
        else
            return static_cast<T>(0);
    }

    // Gradient
    // Adapted for gradient calculation
    inline T grad_r(T r, T h) const {
        const T q = std::abs(r) / h;

        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(2.0 / (3 * h*h));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 3)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (M_PI * std::pow(h, 4)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, 
              "Only 1, 2, 3 dimensions supported");

        if (q >= 0 && q <= 1)
            return -sigma * 3 * q * (1 - 0.75 * q);
            //return sigma * (3/4) * q*(3*q-4);
        else if (q > 1 && q <= 2)
            return -(sigma / 4) * 3 * std::pow((2 - q), 2);
        else
            return static_cast<T>(0);
    }

    //inline T grad_h(T r, T h) const { //for now : only dim 2
    //    const T q = std::abs(r) / h;

    //    T sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 2)));

    //    if (q >= 0 && q <= 1)
    //        return -((2*sigma)/h) * (1 - 1.5 * std::pow(q, 2) 
    //            * (1 - q / 2)) + (-q/h)*sigma*3/4 * q*(3*q - 4) ;
    //    else if (q > 1 && q <= 2)
    //        return -(sigma / (2*h)) * std::pow((2 - q), 3) 
    //          + ((q*sigma) / (4*h)) * 3 * std::pow((2 - q), 2);
    //    else
    //        return static_cast<T>(0);
    //}
    inline T grad_h(T r, T h) const { //for now : only dim 2
        const T q = std::abs(r) / h;

        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(2.0 / (3 * h));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 2)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (M_PI * std::pow(h, 3)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, 
              "Only 1, 2, 3 dimensions supported");

        T w, dw;
        T tmp2 = 2. - q;

        if(q > 2.){
          w = 0;
          dw = 0;
        } else if(q > 1.){
          w = 0.25 * tmp2 * tmp2 * tmp2;
          dw = -0.75 * tmp2 * tmp2;
        }
        else{
          w = 1 - 1.5 * q * q * (1 - 0.5 * q);
          dw = -3 * q * (1 - 0.75 * q);
        }
        return -(sigma/h) * (dw * q + w * DIM);
    }
};

// TODO: Check if the formulae are correct
template<typename T, unsigned DIM>
struct QuinticSplineKernel{
    // Evaluation
    // Modified to match the normalization of the provided W function
    inline T operator()(T r, T h) const {
        const T q = std::abs(r) / h;
        // Normalization constant
        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(1.0 / (120 * h));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(7.0 / (478 * M_PI * std::pow(h, 2)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (120 * M_PI * std::pow(h, 3)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, 
              "Only 1, 2, 3 dimensions supported");

        if (q >= 0 && q <= 1)
            return sigma * (1 *  std::pow(3 - q, 5) - 
                6 *  std::pow(2 - q, 5) + 
                15 * std::pow(1 - q, 5));
        else if (q > 1 && q <= 2)
            return sigma * (1 * std::pow(3 - q, 5) - 
                6 * std::pow(2 - q, 5));
        else if (2 < q && q <= 3)
            return sigma * std::pow(3 - q, 5);
        else
            return static_cast<T>(0);
    }

    // Gradient
    // Adapted for gradient calculation
    inline T grad_r(T r, T h) const {
        const T q = std::abs(r) / h;
        // Get normalization
        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(1.0 / (120 * std::pow(h, 2)));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(7.0 / (478 * M_PI * std::pow(h, 3)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (120 * M_PI * std::pow(h, 4)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, 
              "Only 1, 2, 3 dimensions supported");

        if (q >= 0 && q <= 1)
            return sigma * (1  * (-5) * std::pow(3 - q, 4) - 
                6  * (-5) * std::pow(2 - q, 4) + 
                15 * (-1) * std::pow(1 - q, 4));
        else if (q > 1 && q <= 2)
            return sigma * (1 * (-5) * std::pow(3 - q, 4) - 
                6 * (-5) * std::pow(2 - q, 4));
        else if (2 < q && q <= 3)
            return sigma * (-5) * std::pow(3 - q, 4);
        else
            return static_cast<T>(0);
    }
};

