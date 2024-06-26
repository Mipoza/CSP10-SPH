#include <iostream>
#include <Kokkos_Macros.hpp>
#include <limits>
#include <cmath>

#define __eps (2*std::numeric_limits<SCALAR>::epsilon())

template <typename SCALAR, typename FUNCTOR>
KOKKOS_INLINE_FUNCTION
SCALAR illinois_lower_bound(SCALAR x0, SCALAR x1, FUNCTOR f, 
                            SCALAR rtol = static_cast<SCALAR>(1e-6),
                            SCALAR atol = __eps,
                            unsigned maxit = 100){
  SCALAR x, fx;
  SCALAR f0 = f(x0),
         f1 = f(x1);

  unsigned it = 0;
  
  // A lower bound is given by x0, so we can freely increase x1
  // if necessary. Also we assume that the root is unique and f continuous.
  if(f0 * f1 > 0) // Same sign!
    return illinois_lower_bound(x1, 2. * x1, f, rtol, atol, maxit);

  for(; (it < maxit) && 
        (std::abs(x1 - x0) >= rtol * std::abs(x1)); ++it){
    x = x1 - f1 * (x1 - x0)/(f1 - f0);
    fx = f(x);
    
    if(fx * f1 < 0){
      x0 = x1;
      f0 = f1;
    }
    else f0 *= 0.5;

    x1 = x;
    f1 = fx;

    if(std::abs(fx) < atol) break;
  }

  if(it == maxit){
    std::cerr << "Nonlinear solver failed to converge, aborting.\n"
              << "x = " << x << std::endl
              << "fx = " << fx
              << std::endl;
    std::abort();
  }

  return x;
}
