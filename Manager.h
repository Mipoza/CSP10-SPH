
#include <iostream>
#include <cmath>

#include "../include/Ippl.h"

#include "../include/Expression/IpplExpressions.h" 
//#include "../include/Expression/IpplOperators.h" 

#include "../include/Types/Vector.h"
#include "../include/Particle/ParticleLayout.h"
#include "../include/Particle/ParticleSpatialLayout.h"
#include "../include/Particle/ParticleBase.h"
#include "../include/Particle/ParticleAttribBase.h"
#include "../include/Particle/ParticleAttrib.h"
#include "../include/FieldLayout/FieldLayout.h"
#include "../include/Field/Field.h"
#include "../include/Field/BareField.h"
#include "../include/Meshes/CartesianCentering.h"
#include "../include/Particle/ParticleBC.h"

#include "../include/Manager/BaseManager.h"
#include "SPHParticle_radovan.hpp"


using namespace ippl;
using namespace std;

#define eps 1e-4


template <unsigned int dim>
struct viscosity_factor{
  double alpha, beta;

  viscosity_factor(double alpha_, double beta_): alpha(alpha_), beta(beta_){}

    inline double operator()(double c, double density, Vector<double, dim> dis, Vector<double, dim> vel, double h) {

        double rij = std::sqrt(dis.dot(dis));
        double dot_product = dis.dot(vel);
        double mu = h*dot_product/(std::pow(rij,2) + eps*pow(h,2));

        
        if (dot_product < 0) {return (-(alpha * c * mu) + (beta * mu * mu)) / (density + eps);} 
        else {return 0;}
    }

};

template<unsigned int Dim, unsigned int N>
unsigned int getdim(Vector<Vector<double, Dim>,N>)
{
    return N;
}

template<unsigned int Dim, unsigned int N>
class Manager: public BaseManager {
public:
    using particle_position_type  = ParticleAttrib<Vector<double, Dim>>;
    using particle_scalar_type  = ParticleAttrib<double>;

    SPHParticle<double, Dim> particles;
    double dt;
    double Adiabatic_index;
    double h;
    std::array<ippl::BC,2*Dim>  bcs;


    Manager(ParticleSpatialLayout<double,Dim>& L, ippl::Vector<double,Dim>& low, ippl::Vector<double,Dim>& fin, double dt_, double h_, double Adiabatic_index_) : 
    dt(dt_), h(h_), particles(L, low, fin, h_), Adiabatic_index(Adiabatic_index_) {}

    void pre_run(std::vector<Vector<double, Dim>> R_part, std::vector<Vector<double, Dim>> v_part, std::vector<double> E_part, std::vector<double> m_part, std::array<ippl::BC,2*Dim> bcs) 
    {
        //int N_new = getdim(R_part);
        int N_new = R_part.size();
        particles.create(N_new);

        typename Manager<Dim, N>::particle_position_type::HostMirror R_host = particles.position.getHostMirror();
        typename Manager<Dim, N>::particle_position_type::HostMirror v_host = particles.velocity.getHostMirror();
        typename Manager<Dim, N>::particle_scalar_type::HostMirror m_host = particles.mass.getHostMirror();
        typename Manager<Dim, N>::particle_scalar_type::HostMirror E_host = particles.energy_density.getHostMirror();


        for (unsigned int i = 0; i < N_new; ++i) {

            R_host(i) = R_part[i];
            v_host(i) = v_part[i];
            m_host(i) = m_part[i];
            E_host(i) = E_part[i];

        }

        Kokkos::deep_copy(particles.position.getView(), R_host);
        Kokkos::deep_copy(particles.velocity.getView(), v_host);
        Kokkos::deep_copy(particles.mass.getView(), m_host);
        Kokkos::deep_copy(particles.energy_density.getView(), E_host);

        // particles.update();

        particles.setParticleBC(bcs);
        this->bcs = bcs;
    }

    void pre_step(bool viscous = false)
        { 
            // viscosity_factor<Dim> visc(2.0, 1.2);
            viscosity_factor<Dim> visc(0.1, 0.2);
            particles.updateNeighbors();

            const std::size_t N_particles = particles.position.size();
            // TODO: Kokkos here?
            CubicSplineKernel<double, Dim> W;
            #pragma omp parllel for
            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
                auto nn = particles.CMHelper.neighbors(particles.position(p_idx));
                particles.density(p_idx) = 0.0;

                for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
                    const auto& other_pos = particles.position(*p_it);
                    ippl::Vector<double, Dim> d =  other_pos - particles.position(p_idx);
                    double rij = std::sqrt(d.dot(d));
                    particles.density(p_idx) += particles.mass(*p_it)*W(rij, h);
                    
                }
                if(viscous){
                    particles.pressure(p_idx) = (Adiabatic_index-1)*particles.density(p_idx)*particles.energy_density(p_idx);
                }
                else{
                    particles.pressure(p_idx) = pow(particles.density(p_idx), Adiabatic_index);
                }
            }
            
            #pragma omp parllel for
            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
                auto nn = particles.CMHelper.neighbors(particles.position(p_idx));
                particles.accel(p_idx) = 0.0;
                for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
                    if(p_idx != *p_it){
                        const auto& other_pos = particles.position(*p_it);
                        const auto& other_vel = particles.velocity(*p_it);
                        ippl::Vector<double, Dim> d =  other_pos - particles.position(p_idx);
                        ippl::Vector<double, Dim> vel =  other_vel - particles.velocity(p_idx);
                        double rij = std::sqrt(d.dot(d));
                        double vij = std::sqrt(vel.dot(vel));

                        if (viscous) 
                        {
                            double aux = ((particles.pressure(*p_it)/pow(particles.density(*p_it) + eps,2))
                                -((particles.pressure(p_idx)/pow(particles.density(p_idx) + eps,2))));
                            double aux_1 = (((particles.pressure(p_idx)/pow(particles.density(p_idx) + eps,2))));

                            particles.accel(p_idx) += (-((d)*W.grad_r(rij, h))/(rij + eps))*(particles.mass(*p_it))*aux;
                            double density_mean = (particles.density(*p_it) + particles.density(p_idx))/2;

                            particles.accel(p_idx) += particles.mass(*p_it)*(-((d)*W.grad_r(rij, h))/(rij + eps))*visc(0.1, density_mean, d, vel, h);
                            particles.d_energy_density(p_idx) += (aux_1*particles.mass(*p_it)*vij*W.grad_r(rij, h))
                              + (0.5*particles.mass(*p_it)*visc(0.1, density_mean, d, vel, h)*vij*W.grad_r(rij, h));

                        } 

                        else 
                        {
                            double aux_1 = (((particles.pressure(p_idx)/pow(particles.density(p_idx) + eps,2))));
                            particles.d_energy_density(p_idx) += (aux_1*particles.mass(*p_it)*vel.dot((-(d)*W.grad_r(rij, h))/(rij + eps)));
                            double aux = ((particles.pressure(*p_it)/pow(particles.density(*p_it) + eps,2))
                                -((particles.pressure(p_idx)/pow(particles.density(p_idx) + eps,2))));
                            particles.accel(p_idx) += (-((d)*W.grad_r(rij, h))/(rij + eps))*(particles.mass(*p_it))*aux;
                        }
                    }
                }
            }

        }

       /**
        * @brief A method that should be used after perfoming a step of simulation.
        *
        * Derived classes can override this method to dump data, increment time, etc.
        * The default implementation does nothing.
        */
        void post_step() 
        { cout << "post_step" << endl;}

       /**
        * @brief A method that should be used to execute/advance a step of simulation.
        *
        *  In a derived class, the user must override this method to implement
        *  their time integration method for solving the considered governing equation.
        */
        void advance() 
        { 
           

            /*particles.velocity = particles.velocity + particles.accel*dt/2.;

            particles.position = particles.position + particles.velocity*dt;

            particles.velocity = particles.velocity + particles.accel*dt/2.;*/

            const std::size_t N_particles = particles.position.size();
            
            #pragma omp parallel for
            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx) {
                // Update velocity with half of the acceleration
                particles.velocity(p_idx) += particles.accel(p_idx) * dt / 2.;

                // Update position
                particles.position(p_idx) += particles.velocity(p_idx) * dt;

                particles.energy_density(p_idx) += particles.d_energy_density(p_idx) * dt;

                // Reflective boundary conditions
                for(unsigned int i = 0; i < Dim; ++i) {
                    if(particles.position(p_idx)[i] < 0.05 || particles.position(p_idx)[i] > 0.95) {
                        // Reflect the position
                        particles.position(p_idx)[i] = std::max(0.05, std::min(0.95, particles.position(p_idx)[i])); // Clamp position to [0,1]

                        // Reverse the speed in the respective direction
                        particles.velocity(p_idx)[i] *= -1.0;
                    }
                }

                // Update velocity with the other half of the acceleration
                particles.velocity(p_idx) += particles.accel(p_idx) * dt / 2.;
            }


        }

       /**
        * @brief The main for loop fro running a simulation.
        *
        * This method performs a simulation run by calling pre_step, advance, and post_step
        * in a loop for a specified number of time steps.
        *
        * @param nt The number of time steps to run the simulation.
        */
};

