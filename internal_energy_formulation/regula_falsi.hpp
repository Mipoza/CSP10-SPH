#include <iostream>
#include <Kokkos_Macros.hpp>
#include <limits>
#include <cmath>

#define __eps (16*std::numeric_limits<SCALAR>::epsilon())

enum EXIT_REASON { MAXIT_REACHED, ATOL, RTOL };

template <typename SCALAR, typename FUNCTOR>
KOKKOS_INLINE_FUNCTION
SCALAR illinois(SCALAR x0, SCALAR x1, FUNCTOR f, 
                SCALAR rtol = static_cast<SCALAR>(1e-6),
                SCALAR atol = __eps,
                unsigned maxit = 256){
  SCALAR f0 = f(x0),
         f1 = f(x1);
  SCALAR x = x0, fx = f0;
  EXIT_REASON reason = MAXIT_REACHED;

  unsigned it;

  // We know that the function is *increasing*
  if(f0 * f1 > 0){ // Same sign!
    if(f0 > 0) // Both are positive, x is too large
      return illinois(static_cast<SCALAR>(0.75)*x0, x0, f, rtol, atol, maxit);
    else // Both are negative, x is too small
      return illinois(x1, static_cast<SCALAR>(1.25)*x1, f, rtol, atol, maxit);
  }

  for(it = 0;
      it < maxit;
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

    if(std::abs(f1 * (x1 - x0)) < rtol * std::abs(x1 * (f1 - f0))) {
      reason = RTOL;
      break;
    }
  }

  if(reason == MAXIT_REACHED){
    std::cerr << std::scientific << std::setprecision(10);
    std::cerr << "Nonlinear solver failed to converge, aborting.\n"
              << "x0 = " << x0 << std::endl
              << "x1 = " << x1 << std::endl
              << "f0 = " << f(x0) << std::endl
              << "f1 = " << f(x1) << std::endl
              << "abs(x1 - x0) = " << std::abs(x1 - x0) << std::endl
              << "abs(f1 - f0) = " << std::abs(f(x1) - f(x0)) << std::endl
              << "x = " << x << std::endl
              << "fx = " << fx << std::endl;
    std::abort();
  }
  
  return x;
}

template <typename SCALAR, typename FUNCTOR>
KOKKOS_INLINE_FUNCTION
SCALAR newton(SCALAR x, FUNCTOR f, 
              SCALAR rtol = static_cast<SCALAR>(1e-6),
              SCALAR atol = __eps,
              unsigned maxit = 256){
  SCALAR d = 0;
  EXIT_REASON reason = MAXIT_REACHED;

  unsigned it;

  for(it = 0;
      it < maxit;
      ++it){
    // If fp \approx 0, we have division by 0
    auto [fx, fp] = f(x);
    if(std::abs(fp) < 2*std::numeric_limits<SCALAR>::epsilon())
      break;

    d = fx/fp;

    if(std::abs(x) < rtol * std::abs(d)){
      reason = RTOL;
      break;
    }

    if(std::abs(fx) < atol){
      reason = ATOL;
      break;
    }

    x -= d;
  }
  std::cout << "IT: " << it << std::endl;

  if(reason == MAXIT_REACHED){
    std::cerr << std::scientific << std::setprecision(10);
    std::cerr << "Nonlinear solver failed to converge, aborting.\n"
              << "x = " << x << std::endl
              << "fx = " << f(x).first << std::endl;
    std::abort();
  }
  
  return x;
}
