#include "../include/Particle/ParticleBase.h"
#include "../include/Particle/ParticleAttrib.h"
#include <cmath>
#include <forward_list>

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
  inline T grad_r(T r, T h);
}; 

/* --- For convenience, do not use outside of the particle class -- */
#define getNeighbors(p_idx_) (this->n_idx[p_idx_])

/* Particle class for SPH with nearest neighbor search and 
 * smoothening. */
template <typename T, unsigned DIM, class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHParticle: public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, DIM>>{
  typedef ippl::ParticleSpatialLayout<T, DIM> PLayout_t;
  // Smoothing kernel length
  T kernel_size;
  // The kernel itself
  KERNEL K;
  
  // TODO: Add a member variable named n_idx or smth
  //       We still need to solve the nearest neighbor problem
  // Helper for nearest neighbors (?)
  ippl::ParticleAttrib<std::size_t> idx;
  ippl::ParticleAttrib<std::forward_list<std::size_t>> neighbor_list;

  // Physical quantities
  ippl::ParticleAttrib<T> mass, density, pressure;
  ippl::ParticleBase<PLayout_t>::particle_position_type 
    position, velocity, accel;


  SPHParticle(PLayout_t& PL, T h_): ippl::ParticleBase<PLayout_t>(PL), kernel_size(h_){
    registerAttributes();
  }

  void registerAttributes(){
    this->addAttribute(idx);

    this->addAttribute(mass);
    this->addAttribute(density);
    this->addAttribute(pressure);
    this->addAttribute(position);
    this->addAttribute(velocity);
    this->addAttribute(accel);
  }


  void updateNeighbors(){
    // TODO: solve nearest neighbor problem
  }

  // Find nearest neighbors and smoothen
  // quantities of interest
  // Still a TODO
  void smoothen(){
    updateNeighbors();
    /* Potential form of this function

    // OpenMP here?
    for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
      for(auto neighbor_idx : getNeighbors(p_idx)){
        T rij; // Distance
        rij = ...

        // TODO: Complete (with viscosity switch etc)
        accel(p_idx) -= mass(p_idx)*...
      }
    }

    */
  }

  void set_bd(){
    // TODO
  }

};
