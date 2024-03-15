#pragma once
#include <cmath>

struct CubicSplineKernel{
  // Evaluation
  inline double operator()(const double r, const double h){
    const double q = std::abs(r)/(2*h);
    if(q < 0.5)
      return (8./3.)*(1. - 6.*std::pow(q, 2) + 6.*std::pow(q, 3));
    else if(q < 1.)
      return (16./3.)*std::pow(1. - q, 3);
    else return 0.;
  }
  // Gradient
  inline double grad_r(const double r, const double h);
  inline double grad_h(const double r, const double h);
};
