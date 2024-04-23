#include "../include/Manager/BaseManager.h"
#include "SPHParticle.hpp"

// #pragma once
// #include <iostream>


template<unsigned DIM, unsigned N>
unsigned getdim(Vector<Vector<double, DIM>,N>)
{
    return N;
}

// T: scalar, DIM: dimension, PC: Particle-Class
template <typename T, unsigned DIM, class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHManager : public ippl::BaseManager {
  // From ParticleBase
  SPHParticle<T, DIM, KERNEL> particles;
  // Time-step size
  T dt;
  
  SPHManager(SPHParticle<T, DIM, KERNEL>& P, T dt_): dt(dt_), particles(P) {}

  void pre_run(Vector<Vector<T, DIM>,N> R_part, Vector<Vector<T, DIM>,N> v_part) 
  {
        unsigned N_new = getdim(R_part);
        particles.create(N_new);

        typename Manager<DIM, N>::particle_position_type::HostMirror R_host = particles.R.getHostMirror();
        typename Manager<DIM, N>::particle_position_type::HostMirror v_host = particles.vel.getHostMirror();

        for (unsigned int i = 0; i < N_new; ++i) {

            R_host(i) = R_part[i];
            v_host(i) = v_part[i];

        }
        Kokkos::deep_copy(particles.R.getView(), R_host);
        Kokkos::deep_copy(particles.vel.getView(), v_host);

        particles.update();
  }

  //As a preliminary way we pass the function as an argument of pre_step()
  void pre_step(Vector<T, DIM> (*func)(Vector<T, DIM>)) 
  { 
      //particles.acceleration_function_external(func);
  }

  /**
  * @brief A method that should be used after perfoming a step of simulation.
  *
  * Derived classes can override this method to dump data, increment time, etc.
  * The default implementation does nothing.
  */
  void post_step() 
  { cout << "post_step" << endl;}


  // Leap-Frog time-integration
  void advance() override {
    /* --- KDK scheme --- */
    // v_{i+1/2}
    particles.velocity = particles.velocity + particles.accel*dt/2.;
    // x_{i + 1}
    particles.position = particles.position + particles.velocity*dt;
    // Enforce bd conditions
    particles.set_bd();
    // Compute acceleration at new position
    particles.smoothen();
    // v_{i + 1}
    particles.velocity = particles.velocity + particles.accel*dt/2.;

    // For debugging only
    // std::cout << "Time-step\n";
  }
  // pre_step and post_step are no-ops here, not needed
};
