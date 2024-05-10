#pragma once
#include <Kokkos_Macros.hpp>
#include <cmath>

// Avoid div by 0
#define __eps static_cast<T>(1e-8)

template <typename T, typename F>
KOKKOS_INLINE_FUNCTION
T secant(F f, T x0, T x1, 
    T atol = static_cast<T>(1e-3),
    T rtol = static_cast<T>(1e-3),
    const unsigned max_it = 100){
  T x = x0,
    f0, f1;
  unsigned it;
  for(it = 0; it < max_it; ++it){
    f0 = f(x0); f1 = f(x1);
    // std::cout << x1 << " " << f1 << std::endl;
    // Stop?
    if((std::abs(x1 - x0) < atol && 
        std::abs(x1 - x0) < std::abs(x0)*rtol)
     || std::abs(f0)      < __eps
     || std::abs(f1)      < __eps)
      break;
    // std::cout <<"UPDATE";
    // Otherwise, update and continue
    x = x1 - f1*(x1 - x0)/(f1 - f0);
    x0 = x1; x1 = x;
  }
  // If it didn't converge odds are it could not find enough nearby particles
  // or it got stuck deciding whether it should include another particle or not
  // if(it == max_it){
  //   std::cerr << "Failed to converge to appropriate smoothing kernel size, aborting"
  //             << std::endl;
  //   abort();
  // }
  return x;
}
