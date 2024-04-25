#include "../include/Manager/BaseManager.h"
#include "SPHParticle.hpp"

// #pragma once
// #include <iostream>

// T: scalar, DIM: dimension, PC: Particle-Class
template <typename T, unsigned DIM, class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHManager : public ippl::BaseManager {
  // From ParticleBase
  std::shared_ptr<SPHParticle<T, DIM, KERNEL>> particles;
  // Time-step size
  T dt;
  
  SPHManager(std::shared_ptr<SPHParticle<T, DIM, KERNEL>> P,
      double dt_): dt(dt_), particles(P) {}

  // Leap-Frog time-integration
  void advance() override {
    /* --- KDK scheme --- */
    // v_{i+1/2}
    particles->velocity = particles->velocity + particles->accel*dt/2.;
    // x_{i + 1}
    particles->position = particles->position + particles->velocity*dt;
    // Enforce bd conditions
    particles->set_bd();
    // Compute acceleration at new position
    particles->smoothen();
    // v_{i + 1}
    particles->velocity = particles->velocity + particles->accel*dt/2.;

    // For debugging only
    // std::cout << "Time-step\n";
  }
  // pre_step and post_step are no-ops here, not needed
};
