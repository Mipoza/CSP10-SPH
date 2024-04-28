#include "../include/Particle/ParticleBase.h"
#include "../include/Particle/ParticleAttrib.h"
#pragma once
#include <cmath>
#include <iostream>

#include "ChainingMesh.hpp"

/* CubicSplineKernel for smoothening the QOI */
/*template<typename T, unsigned DIM>
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
  inline T grad_r(T r, T h) {

    const T q = std::abs(r)/(2*h);
    if(q < 0.5)
      return (8./3.)*(- 12.*std::pow(q, 1) + 18.*std::pow(q, 2));
    else if(q < 1.)
      return -(16.)*std::pow(1. - q, 2);
    else return 0.;
  }
};*/

template<typename T, unsigned DIM>
struct CubicSplineKernel{
    CubicSplineKernel() = default;

    // Evaluation
    // Modified to match the normalization of the provided W function
    inline T operator()(T r, T h) {
        const T q = std::abs(r) / h;
        const T sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 2))); // normalization constant for dim 2

        if (q >= 0 && q <= 1)
            return sigma * (1 - 1.5 * std::pow(q, 2) * (1 - q / 2));
        else if (q > 1 && q <= 2)
            return (sigma / 4) * std::pow((2 - q), 3);
        else
            return static_cast<T>(0);
    }

    // Gradient
    // Adapted for gradient calculation
    inline T grad_r(T r, T h) {
        const T q = std::abs(r) / h;
        const T sigma = static_cast<T>(10.0 / (7 * M_PI * std::pow(h, 2))); // normalization constant for dim 2

        if (q >= 0 && q <= 1)
            return sigma * (3/4) * q*(3*q-4)*(1/h);
        else if (q > 1 && q <= 2)
            return -(sigma / 4) * 3 * std::pow((2 - q), 2) * (1/h);
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
  ChainingMeshHelper<T, DIM, true> CMHelper;

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
    //this->addAttribute(energy_density);
    this->addAttribute(accel);
  }

  void updateNeighbors(){
    // Clear ChainingMesh
    CMHelper.clear();
    // Add all particles to the "Hash"
    CMHelper.add_particles(position);
  }

  // inline T viscositySwitch(const std::size_t& i, const std::size_t& j,
  //     const T& rij, const ippl::Vector<T, DIM>& ri, const ippl::Vector<T, DIM>& rj){
  //   T Pij = 0;

  //   ippl::Vector<T, DIM> rij = ri - rj; 
  //   ippl::Vector<T, DIM> vij = velocity(i) - velocity(j); 
  //   const T dot_product = rij.dot(vij);

  //   if(dot_product >= 0)
  //     return Pij;
  //   // Else continue computation
  //   const T rho_ij_bar = (pressure(i) + pressure(j))/2.;
  //   const T c_ij_bar = (std::sqrt(gamm*pressure(i)/density(i))
  //       + std::sqrt(gamm*pressure(j)/density(j)))/2.;

  //   const T mu_ij = kernel_size*dot_product/(rij*rij + eps*kernel_size*kernel_size);

  //   Pij = (-alpha*c_ij_bar*mu_ij + beta*mu_ij*mu_ij)/rho_ij_bar;
  //   return Pij;
  // }

  // Find nearest neighbors and smoothen
  // quantities of interest
  void smoothen(){
    updateNeighbors();
    // Potential form of this function

    const std::size_t N_particles = position.size();
    // TODO: Kokkos here?
    // for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
    //   auto nn = CMHelper.neighbors(position(p_idx));
    //   const auto& this_pos = position(p_idx);
    //   // Term constant for this particle
    //   const T p_idx_constant = pressure(p_idx)/(density(p_idx)*density(p_idx));
    //   for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
    //     const auto& other_pos = position(*p_it);
    //     ippl::Vector<T, DIM> d = (this_pos - other_pos);

    //     const T rij_m = std::sqrt(d.dot(d));

    //     // TODO: Complete (with viscosity switch etc)
    //     accel(p_idx) -= 
    //       mass(*p_it)*
    //       (
    //         p_idx_constant 
    //         + pressure(*p_it)/(density(*p_it)*density(*p_it))
    //         + viscositySwitch(p_idx, *p_it, rij_m)
    //       )
    //       *K.grad_r(rij_m, kernel_size);
    //   }
    // }

  }

  void set_bd(){
    // TODO
  }

};