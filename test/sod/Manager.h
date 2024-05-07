
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <cmath>

#include "../../include/Ippl.h"

#include "../../include/Expression/IpplExpressions.h" 
//#include "../../include/Expression/IpplOperators.h" 

#include "../../include/Types/Vector.h"
#include "../../include/Particle/ParticleLayout.h"
#include "../../include/Particle/ParticleSpatialLayout.h"
#include "../../include/Particle/ParticleBase.h"
#include "../../include/Particle/ParticleAttribBase.h"
#include "../../include/Particle/ParticleAttrib.h"
#include "../../include/FieldLayout/FieldLayout.h"
#include "../../include/Field/Field.h"
#include "../../include/Field/BareField.h"
#include "../../include/Meshes/CartesianCentering.h"
#include "../../include/Particle/ParticleBC.h"

#include "../../include/Manager/BaseManager.h"
#include "SPHParticle_radovan.hpp"


using namespace ippl;
using namespace std;

#define eps 1e-3


template <unsigned int dim>
struct viscosity_factor{
  double alpha, beta;

  viscosity_factor() = default;

  viscosity_factor(double alpha_): alpha(alpha_), beta(2*alpha_){}

    inline double operator()(double c, double density, Vector<double, dim> dis, Vector<double, dim> vel, double h)  const {

        double rij = std::sqrt(dis.dot(dis));
        double dot_product = dis.dot(vel);
        double mu = h*dot_product/(std::pow(rij,2) + 0.01*std::pow(h,2));

        if (dot_product < 0) {return (-(alpha * c * mu) + (beta * mu * mu)) / (density + eps);} 
        else {return 0;}
    }

};

template<unsigned int Dim>
class Manager: public BaseManager {
public:
    using particle_position_type  = ParticleAttrib<Vector<double, Dim>>;
    using particle_scalar_type  = ParticleAttrib<double>;

    SPHParticle<double, Dim> particles;
    double dt;
    double Adiabatic_index;
    double h;
    ippl::Vector<double, Dim> L_, low_;

    //Manager constructor
    Manager(ParticleSpatialLayout<double,Dim>& L, ippl::Vector<double,Dim>& low,
        ippl::Vector<double,Dim>& extent, double dt_, double h_, double Adiabatic_index_) : 
    dt(dt_), h(h_), particles(L, low, extent, h_), Adiabatic_index(Adiabatic_index_), L_(extent), low_(low) {}

    // IN pre_run we create the particles and set their initial values
    void pre_run(std::vector<Vector<double, Dim>> R_part, std::vector<Vector<double, Dim>> v_part, std::vector<double> E_part, 
    std::vector<double> m_part, std::vector<double> entropy_part) 
    {
        int N_new = R_part.size();
        particles.create(N_new);

        // Create the host mirrors
        typename Manager<Dim>::particle_position_type::HostMirror R_host = particles.position.getHostMirror();
        typename Manager<Dim>::particle_position_type::HostMirror v_host = particles.velocity.getHostMirror();
        typename Manager<Dim>::particle_scalar_type::HostMirror m_host = particles.mass.getHostMirror();
        typename Manager<Dim>::particle_scalar_type::HostMirror entropy_host = particles.entropy.getHostMirror();


        for (unsigned int i = 0; i < N_new; ++i) {

            R_host(i) = R_part[i];
            v_host(i) = v_part[i];
            m_host(i) = m_part[i];
            entropy_host(i) = entropy_part[i];

        }

        Kokkos::deep_copy(particles.position.getView(), R_host);
        Kokkos::deep_copy(particles.velocity.getView(), v_host);
        Kokkos::deep_copy(particles.mass.getView(), m_host);
        Kokkos::deep_copy(particles.entropy.getView(), entropy_host);
    }
    //In pre_step we calculate the density, pressure, acceleration and time derivative of entropy
    //given the position, velocity and entropy of the particles at the current time step
    void pre_step(bool viscous = false)
        { 
            //This is the viscosity factor that is used in the calculation of the acceleration in viscous flows
            viscosity_factor<Dim> visc(1.);
            particles.updateNeighbors();

            CubicSplineKernel<double, Dim> W;
            const std::size_t N_particles = particles.position.size();

            #pragma omp parallel for
            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
                //Nearest neighbors are obtained from here, and the density is calculated
                auto nn = particles.CMHelper.neighbors(particles.position(p_idx));
                particles.density(p_idx) = 0.0;
                for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
                    const auto& other_pos = particles.position(*p_it);
                    ippl::Vector<double, Dim> d =  other_pos - particles.position(p_idx);
                    double rij = std::sqrt(d.dot(d));
                    particles.density(p_idx) = particles.density(p_idx) + particles.mass(*p_it)*W(rij, h);
                }
                particles.pressure(p_idx) = particles.entropy(p_idx)*pow(particles.density(p_idx), Adiabatic_index);
            }

            #pragma omp parallel for
            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
                //Nearest neighbors once again and acceleration and time derivative of entropy is calculated
                auto nn = particles.CMHelper.neighbors(particles.position(p_idx));
                particles.accel(p_idx) = 0.0;
                particles.d_energy_density(p_idx) = 0.0;
                particles.d_entropy(p_idx) = 0.0;
                for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){

                        const auto& other_pos = particles.position(*p_it);
                        const auto& other_vel = particles.velocity(*p_it);
                        ippl::Vector<double, Dim> d =  other_pos - particles.position(p_idx);
                        ippl::Vector<double, Dim> vel =  other_vel - particles.velocity(p_idx);
                        double rij = std::sqrt(d.dot(d)) + eps;
                        double vij = std::sqrt(vel.dot(vel));

                        if (viscous) 
                        {
                            double aux = ((particles.pressure(*p_it)/pow(particles.density(*p_it) + eps,2))
                                +((particles.pressure(p_idx)/pow(particles.density(p_idx) + eps,2))));
                            double aux_1 = (((particles.pressure(p_idx)/pow(particles.density(p_idx) + eps,2))));

                            particles.accel(p_idx) = particles.accel(p_idx) -
                                (-((d)*W.grad_r(rij, h))/(rij ))*(particles.mass(*p_it))*aux;
                            double density_mean = (particles.density(*p_it) + particles.density(p_idx))/2;

                            // Mean sound velocity
                            const double c_ij_bar = (std::sqrt(abs(Adiabatic_index*particles.pressure(p_idx)/
                                    (particles.density(p_idx) + eps)))
                                + std::sqrt(abs(Adiabatic_index*particles.pressure(*p_it)/
                                    (particles.density(*p_it) + eps))))/2.;


                            particles.accel(p_idx) = particles.accel(p_idx) -
                              particles.mass(*p_it)*(-((d)*W.grad_r(rij, h))/(rij ))*visc(c_ij_bar, density_mean, d, vel, h);

                            particles.d_entropy(p_idx) = particles.d_entropy(p_idx) + 
                            ((Adiabatic_index-1.0)/std::pow(particles.density(p_idx)+eps, Adiabatic_index -1.0))*
                            (0.5*particles.mass(*p_it)*
                             visc(c_ij_bar, density_mean, d, vel, h)*vel.dot((-(d)*W.grad_r(rij, h))/(rij )));
                        } 

                        else 
                        {
                            double aux = ((particles.pressure(*p_it)/
                                          pow(particles.density(*p_it) + eps,2))
                                        +((particles.pressure(p_idx)/
                                          pow(particles.density(p_idx) + eps,2))));
                            particles.accel(p_idx) = particles.accel(p_idx) -
                              (-((d)*W.grad_r(rij, h))/(rij + eps))*(particles.mass(*p_it))*aux;
                        }
                }
            }
        }

        //In advance we update the position, velocity and entropy of the particles for the next time step
        //only for entropy we use a Runge-Kutta method, as it is a first order differential equation
        void advance() 
        { 
            const std::size_t N_particles = particles.position.size();

            Kokkos::parallel_for(N_particles, 
              KOKKOS_LAMBDA (const std::size_t p_idx){

                // Update velocity with half of the acceleration
                particles.velocity(p_idx) = particles.velocity(p_idx) + particles.accel(p_idx) * dt / 2.;

                // Update position
                particles.position(p_idx) = particles.position(p_idx) + particles.velocity(p_idx) * dt;

                // Update velocity with the other half of the acceleration
                particles.velocity(p_idx) = particles.velocity(p_idx) + particles.accel(p_idx) * dt / 2.;

                // Initial values
                auto initial_entropy = particles.entropy(p_idx);
                
                // First step of Runge-Kutta
                auto k1_entropy = particles.d_entropy(p_idx) * dt;

                // Second step of Runge-Kutta
                auto k2_entropy = particles.d_entropy(p_idx) * dt;

                // Third step of Runge-Kutta
                auto k3_entropy = particles.d_entropy(p_idx) * dt;

                // Fourth step of Runge-Kutta
                auto k4_entropy = particles.d_entropy(p_idx) * dt;

                // Update entropy
                particles.entropy(p_idx) = initial_entropy + (k1_entropy + 2.0 * k2_entropy + 2.0 * k3_entropy + k4_entropy) / 6.0;

                // Boundary conditions for the the particular KH instability problem i.e. periodic in x and reflecting in y
                //We expect to set the BC in a more efficient way in the future, but for now this is the way to do it

                // Reflective boundary conditions
                for(unsigned int i = 1; i < Dim; ++i) {
                    if(particles.position(p_idx)[i] < 0.01 || particles.position(p_idx)[i] > 0.99) {
                        // Reflect the position
                        particles.position(p_idx)[i] = std::max(0.01, std::min(0.99, particles.position(p_idx)[i])); // Clamp position to [0,1]

                        // Reverse the speed in the respective direction
                        particles.velocity(p_idx)[i] *= -1.0;
                    }
                }  

            });
        }

        // The following functions are used to calculate the density, 
        //pressure and velocity at an arbitrary position using the SPH interpolation method
       double density_arbitrary_pos(const ippl::Vector<double, Dim> pos) const
       {
            CubicSplineKernel<double, Dim> W;
            double density_final = 0.0;
            for (int i = 0; i < particles.position.size(); i++)
            {
                ippl::Vector<double,Dim> r_vec= pos - particles.position(i);
                double r = std::sqrt(r_vec.dot(r_vec));

                density_final = density_final + particles.mass(i)*W(r, 2*h);
            }
            return density_final;
       }

       double pressure_arbitrary_pos(const ippl::Vector<double, Dim> pos) const
       {
            CubicSplineKernel<double, Dim> W;
            double pressure_final = 0;
            for (int i = 0; i < particles.position.size(); i++)
            {
                ippl::Vector<double,Dim> r_vec= pos - particles.position(i);
                double r = std::sqrt(r_vec.dot(r_vec));
                double volume_element = particles.mass(i)/(particles.density(i)+eps);

                pressure_final = pressure_final + particles.pressure(i)*W(r, 2*h)*volume_element;
            }
            return pressure_final;
       }

       double velocity_arbitrary_pos(const ippl::Vector<double, Dim> pos) const
       {
            CubicSplineKernel<double, Dim> W;
            double velocity_final = 0.0;
            for (int i = 0; i < particles.position.size(); i++)
            {
                ippl::Vector<double,Dim> r_vec= pos - particles.position(i);
                double r = std::sqrt(r_vec.dot(r_vec));
                double volume_element = particles.mass(i)/(particles.density(i)+eps);

                velocity_final = velocity_final + particles.velocity(i)[0]*W(r, 2*h)*volume_element;
            }
            return velocity_final;
       }
};


