#pragma once
#include <Kokkos_Macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>

#define __eps (2*std::numeric_limits<T>::epsilon())



template <typename T, typename F>
KOKKOS_INLINE_FUNCTION
T solve(F f, T x0, T x1, 
    T tol = static_cast<T>(1e-6),
    const unsigned max_it = 100){
  T mid       = (x0 + x1)/2,
    d         = x1 - x0,
    fmid      = f(mid),
    fmid_old  = 0;

  // No zero in sight
  // We know that f is monotone
  // Assume that x0 is a good lower bound and 
  // that the domain on which the function is zero is connected
  if(f(x1)*f(x0) >= 0)
    return solve(f, x0, x1*2, tol, max_it);

  unsigned it;
#ifdef DEBUG
  assert(x0 < x1);
#endif 
  for(it = 0; it < max_it; ++it){
    // Stop?
    if( std::abs(x1 - x0) < std::abs(x1 + x0)*tol
     || std::abs(fmid) < __eps)
      break;
    // Otherwise, update and continue
    if(fmid < 0)
      x0 += d/2;
    else 
      x1 -= d/2;

    d = x1 - x0;
    mid = (x1 + x0)/2;
    fmid_old = fmid;
    fmid = f(mid);
  }
  // std::cout << "Terminated with: " << it << " " << std::abs(fmid - fmid_old) << " " << fmid << " " << x0 << " " << x1 << std::endl;
  // If it didn't converge odds are it could not find enough nearby particles
  // or it got stuck deciding whether it should include another particle or not
  if(it == max_it){
    std::cerr << "Failed to converge to appropriate smoothing kernel size, aborting.\n"
              << "x = " << mid << std::endl
              << "fmid = " << fmid
              << std::endl;
    abort();
  }
  return mid;
}



