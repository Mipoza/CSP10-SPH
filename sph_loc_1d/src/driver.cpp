#pragma once
#include <algorithm>
#include <cmath>
#include "driver.hpp"
#include "particle.hpp"

template <class KERNEL_, KERNEL_ KERNEL>
void SPHDriver<KERNEL_, KERNEL>::step(const double dt){
  #warning "Initialize acceleration before the first time-step"
  /* --- KDK scheme --- */
  // v_{i+1/2}
  for(auto& p : this->particles)
    p.velocity += p.dvdt*dt/2.;
  // x_{i + 1}
  for(auto& p : this->particles)
    p.pos += p.velocity*dt;
  // Compute acceleration at new position
  for(auto& p : this->particles)
    p.template accel<KERNEL_, KERNEL>();
  // v_{i + 1}
  for(auto& p : this->particles)
    p.velocity += p.dvdt*dt/2.;
}


template <class KERNEL_, KERNEL_ KERNEL>
void SPHDriver<KERNEL_, KERNEL>::find_neighbours(){
  // Sort particles so we can find neighbours
  // Maybe redunant in 1D? Not sure, just do it
  std::sort(this->particles.begin(), this->particles.end());
  for(std::size_t idx = 0; idx < this->particles.size(); ++idx){
    SPHParticle* particle = &this->particles[idx];
    // Reset list
    particle->neighbours.clear();

    const SPHParticle* potential_neighbour;
    // Search backwards
    // n_idx ~ neighbour_idx
    for(std::size_t n_idx = idx; n_idx > 0; --n_idx){
      potential_neighbour = &this->particles[n_idx];
      const double dist = std::abs(particle->pos - potential_neighbour->pos);
      // We have exited the range of potential neighbours
      if(dist > particle->kernel_size)
        break;
      // else add to list
      particle->neighbours.push_back(potential_neighbour);
    }
    // Search forwards
    // start at + 1, since the last loop already checked n_idx == idx
    for(std::size_t n_idx = idx + 1; n_idx < this->particles.size(); ++n_idx){
      potential_neighbour = &this->particles[n_idx];
      const double dist = std::abs(particle->pos - potential_neighbour->pos);
      // We have exited the range of potential neighbours
      if(dist > particle->kernel_size)
        break;
      // else add to list
      particle->neighbours.push_back(potential_neighbour);
    }
  } 
}
