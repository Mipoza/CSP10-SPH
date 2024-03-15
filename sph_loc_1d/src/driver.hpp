#pragma once
#include <vector>
#include "particle.hpp"

template <class KERNEL_, KERNEL_ KERNEL>
struct SPHDriver{
  // Store the particles
  std::vector<SPHParticle> particles;
  // Update neighbours
  void find_neighbours();
  // Find largest allowed time-step (using the CFL condition)
  double find_dt();
  // Time-step (verlet/leap-frog) Kick-Drift-Kick
  void step(const double dt);
};
