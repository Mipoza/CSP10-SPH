#include "../include/Particle/ParticleBase.h"
#include "../include/Particle/ParticleAttrib.h"
#pragma once
#include <cmath>
#include <iostream>

#include "ChainingMesh.hpp"

template<typename T, unsigned DIM>
struct CubicSplineKernel{
    // Evaluation
    // Modified to match the normalization of the provided W function
    inline T operator()(T r, T h) const {
        const T q = std::abs(r) / h;
        // Normalization constant
        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(2.0 / (3 * h));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 2)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (M_PI * std::pow(h, 3)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, "Only 1, 2, 3 dimensions supported");

        if (q >= 0 && q <= 1)
            return sigma * (1 - 1.5 * std::pow(q, 2) * (1 - q / 2));
        else if (q > 1 && q <= 2)
            return (sigma / 4) * std::pow((2 - q), 3);
        else
            return static_cast<T>(0);
    }

    // Gradient
    // Adapted for gradient calculation
    inline T grad_r(T r, T h) const {
        const T q = std::abs(r) / h;

        T sigma;
        if constexpr(DIM == 1)
          sigma = static_cast<T>(2.0 / (3 * h*h));
        else if constexpr(DIM == 2)
          sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 3)));
        else if constexpr(DIM == 3)
          sigma = static_cast<T>(1.0 / (M_PI * std::pow(h, 4)));
        else
          static_assert(DIM == 1 || DIM == 2 || DIM == 3, "Only 1, 2, 3 dimensions supported");

        if (q >= 0 && q <= 1)
            return sigma * (3/4) * q*(3*q-4);
        else if (q > 1 && q <= 2)
            return -(sigma / 4) * 3 * std::pow((2 - q), 2);
        else
            return static_cast<T>(0);
    }
};


/* Particle class for SPH with nearest neighbor search and 
 * smoothening. */
template <typename T, unsigned DIM, class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHParticle: public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, DIM>>{
  typedef ippl::ParticleSpatialLayout<T, DIM> PLayout_t;
  // Smoothing kernel length
  T kernel_size;
  T alpha = 1.2,
    beta  = 2.4,
    eps = 1e-2,
    gamm = 1.4;
  // The kernel itself
  KERNEL K;
  
  // Helper for nearest neighbors
  ChainingMeshHelper<T, DIM, false> CMHelper;

  // Physical quantities
  ippl::ParticleAttrib<T> mass, density, pressure, entropy, d_entropy;
  ippl::ParticleBase<PLayout_t>::particle_position_type 
    position, velocity, accel;


  template <class VEC>
  SPHParticle(PLayout_t& PL, VEC& low, VEC& L, T h_): ippl::ParticleBase<PLayout_t>(PL), kernel_size(h_),
  CMHelper(low, L, h_) {
    registerAttributes();
  }

  void registerAttributes(){
    this->addAttribute(mass);
    this->addAttribute(density);
    this->addAttribute(pressure);
    this->addAttribute(position);
    this->addAttribute(velocity);
    this->addAttribute(accel);
    this->addAttribute(entropy);
    this->addAttribute(d_entropy);
  }

  void updateNeighbors(){
    CMHelper.partition(position);
    // Sort for data locality
    CMHelper.sort(position.getView(), density.getView());
    CMHelper.sort(position.getView(), pressure.getView());
    CMHelper.sort(position.getView(), entropy.getView());
    CMHelper.sort(position.getView(), d_entropy.getView());
    CMHelper.sort(position.getView(), velocity.getView());
    CMHelper.sort(position.getView(), accel.getView());
    CMHelper.sort(position.getView(), position.getView());
  }
};
