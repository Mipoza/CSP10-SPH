#pragma once
#include <cmath>
#include <cstddef>
#include <forward_list>
#include <vector>

struct SPHParticle{
  double mass, pos, velocity,
         density, pressure, kernel_size,
         grad_h_term, dvdt;
  // Default values, ideal gas
  double adiabatic_index = 1.4;
  double entropic_const = 1.;
  // Assume a simple, ideal gas
  double P(const double rho){
    return entropic_const*std::pow(rho, adiabatic_index);
  }
  // Nearest/Smoothing neighbours
  std::vector<const SPHParticle*> neighbours;
  // Compute QOI from smoothed estimates
  template <class KERNEL_, KERNEL_ KERNEL>
  void smoothen();
  // Compute the acceleration on this particle
  // using the smoothed variables from before
  template <class KERNEL_, KERNEL_ KERNEL>
  void accel();
  // We need to sort at some point
  bool operator<(const SPHParticle& other){
    return (this->pos < other.pos);
  }
};
