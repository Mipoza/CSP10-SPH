#include "../include/Particle/ParticleBase.h"
#include "../include/Particle/ParticleAttrib.h"
#pragma once
#include <cmath>
#include <iostream>

#include "ChainingMesh.hpp"

/* CubicSplineKernel for smoothening the QOI */
template<typename T, unsigned DIM>
struct CubicSplineKernel{
  CubicSplineKernel() = default;
  // Evaluation
  // TODO: Implement DIM-dimensional normalization
  // atm only 1d
  inline T operator()(T r, T h){
    const T q = std::abs(r)/(2*h);
    if(q < 0.5)
      return (8./3.)*(1. - 6.*std::pow(q, 2) + 6.*std::pow(q, 3));
    else if(q < 1.)
      return (16./3.)*std::pow(1. - q, 3);
    else return 0.;
  }
  // Gradient
  // TODO: Implement
  inline T grad_r(T r, T h);
}; 


/* Particle class for SPH with nearest neighbor search and 
 * smoothening. */
template <typename T, unsigned DIM, class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHParticle: public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, DIM>>{
  typedef ippl::ParticleSpatialLayout<T, DIM> PLayout_t;
  // Smoothing kernel length
  T kernel_size;
  // The kernel itself
  KERNEL K;
  
  // Helper for nearest neighbors
  ChainingMeshHelper<T, DIM> CMHelper;

  // Physical quantities
  ippl::ParticleAttrib<T> mass, density, pressure;
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
  }

  void updateNeighbors(){
    // Clear ChainingMesh
    CMHelper.clear();
    // Add all particles to the "Hash"
    CMHelper.add_particles(position);
  }

  // Find nearest neighbors and smoothen
  // quantities of interest
  void smoothen(){
    updateNeighbors();
    // Potential form of this function

    const std::size_t N_particles = position.size();
    // TODO: Kokkos here?
    for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
      auto nn = CMHelper.neighbors(position(p_idx));
      for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
        const auto& other_pos = position(*p_it);
        T rij; // Distance
        // rij = ...

        // TODO: Complete (with viscosity switch etc)
        // accel(p_idx) -= mass(p_idx)*...
      }
    }

  }

  void set_bd(){
    // TODO
  }

};
