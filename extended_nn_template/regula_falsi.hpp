#include <iostream>
#include <Kokkos_Macros.hpp>
#include <limits>
#include <cmath>

#define __eps (2*std::numeric_limits<SCALAR>::epsilon())

enum EXIT_REASON { MAXIT_REACHED, ATOL, RTOL };

template <typename SCALAR, typename FUNCTOR>
KOKKOS_INLINE_FUNCTION
SCALAR illinois_lower_bound(SCALAR x0, SCALAR x1, FUNCTOR f, 
                            SCALAR rtol = 10 * __eps, //static_cast<SCALAR>(1e-12),
                            SCALAR atol = __eps,
                            unsigned maxit = 256){
  SCALAR x, fx;
  SCALAR f0 = f(x0),
         f1 = f(x1);
  EXIT_REASON reason = MAXIT_REACHED;

  unsigned it = 0;

  //if(std::abs(f0) < atol){
  //  reason = ATOL;
  //  return x0;
  //}

  //if(std::abs(f1) < atol){
  //  reason = ATOL;
  //  return x1;
  //}

  // std::cout << x0 << " " << x1 << " " << f0 << " " << f1 << std::endl;
  
  // A lower bound is given by x0, so we can freely increase x1
  // if necessary. Also we assume that the root is unique and f continuous.
  if(f0 * f1 >= 0) // Same sign!
    return illinois_lower_bound(x1, 2. * x1, f, rtol, atol, maxit);

  for(; (it < maxit);
      ++it){
    x = x1 - f1 * (x1 - x0)/(f1 - f0);
    fx = f(x);

    if(fx * f1 < 0){
      x0 = x1;
      f0 = f1;
    }
    else f0 *= 0.5;

    x1 = x;
    f1 = fx;

    if(std::abs(fx) < atol) {
      reason = ATOL;
      break;
    }

    //if(std::abs(x1 - x0) < rtol * std::abs(x1)) {
    //  reason = RTOL;
    //  break;
    //}
    if(std::abs(f1 * (x1 - x0)) < rtol * std::abs(x1 * (f1 - f0))) {
      reason = RTOL;
      break;
    }
  }

  if(reason == MAXIT_REACHED){
    std::cerr << std::scientific;
    std::cerr << "Nonlinear solver failed to converge, aborting.\n"
              << "x0 = " << x0 << std::endl
              << "x1 = " << x1 << std::endl
              << "abs(x1 - x0) = " << std::abs(x1 - x0) << std::endl
              << "abs(f1 - f0) = " << std::abs(f(x1) - f(x0)) << std::endl
              << "x = " << x << std::endl
              << "fx = " << fx << std::endl;
    std::abort();
  }
  
  //std::cout << "ERROR: " << f(x) << std::endl
  //          << "REL: " << f1 * (x1 - x0)/(f1 - f0) << " " << x1 * (f1 - f0) << std::endl
  //          << "IT: " << it << std::endl << std::endl;;

  return x;
}
