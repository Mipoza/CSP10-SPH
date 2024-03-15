#include "particle.hpp"

template <class KERNEL_T, KERNEL_T KERNEL>
void SPHParticle::smoothen(){
  this->density = 0.;
  double drho_dh = 0.;
  // Assume "this" is also its own neighbour
  for(const SPHParticle* p : this->neighbours){
    // Density estimate
    this->density += p->mass*KERNEL(this->pos - p->pos,
                                   this->kernel_size);
    // Correction factor, computed alongisde it
    drho_dh += p->mass*KERNEL.grad_h(this->pos - p->pos,
                                    this->kernel_size);
  }
  // Factor of 3 falls away as we are in 1D
  this->grad_h_term = 1. + (this->kernel_size/this->density)*drho_dh;
  // Pressure from density, ideal gas EOS
  this->pressure = this->P(this->density);
}


template <class KERNEL_, KERNEL_ KERNEL>
void SPHParticle::accel(){
  this->dvdt = 0;
  double rij;
  #warning "TODO: Implement viscosity switch"
  for(const SPHParticle* p : this->neighbours){
    rij = this->pos - p->pos;
    this->dvdt -= p->mass*(
        this->pressure*KERNEL.grad_r(rij, this->kernel_size)/
          (this->grad_h_term*std::pow(this->density, 2))
        +
        p->pressure*KERNEL.grad_r(rij, p->kernel_size)/
          (p->grad_h_term*std::pow(p->density, 2))
        );
  }
}
