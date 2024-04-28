
#include <iostream>
#include <cmath>

#include "include/Ippl.h"

#include "include/Expression/IpplExpressions.h" 
//#include "include/Expression/IpplOperators.h" 

#include "include/Types/Vector.h"
#include "include/Particle/ParticleLayout.h"
#include "include/Particle/ParticleSpatialLayout.h"
#include "include/Particle/ParticleBase.h"
#include "include/Particle/ParticleAttribBase.h"
#include "include/Particle/ParticleAttrib.h"
#include "include/FieldLayout/FieldLayout.h"
#include "include/Field/Field.h"
#include "include/Field/BareField.h"
#include "include/Meshes/CartesianCentering.h"
#include "include/Particle/ParticleBC.h"

#include "include/Manager/BaseManager.h"
#include "SPHParticle_radovan.hpp"


using namespace ippl;
using namespace std;

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

    Manager(ParticleSpatialLayout<double,Dim>& L, ippl::Vector<double,Dim>& low, ippl::Vector<double,Dim>& fin, double dt_, double h_) : dt(dt_), h(h_), particles(L, low, fin, h_) {}

    void pre_run(std::vector<Vector<double, Dim>> R_part, std::vector<Vector<double, Dim>> v_part, std::vector<double> E_part, std::vector<double> m_part, ippl::BC bc) 
    {
        //int N_new = getdim(R_part);
        int N_new = R_part.size();
        cout << "N_new: " << N_new << endl;
        particles.create(N_new);

        typename Manager<Dim, N>::particle_position_type::HostMirror R_host = particles.position.getHostMirror();
        typename Manager<Dim, N>::particle_position_type::HostMirror v_host = particles.velocity.getHostMirror();
        typename Manager<Dim, N>::particle_scalar_type::HostMirror m_host = particles.mass.getHostMirror();

        for (unsigned int i = 0; i < N_new; ++i) {

            R_host(i) = R_part[i];
            v_host(i) = v_part[i];
            m_host(i) = m_part[i];

        }

        Kokkos::deep_copy(particles.position.getView(), R_host);
        Kokkos::deep_copy(particles.velocity.getView(), v_host);
        Kokkos::deep_copy(particles.mass.getView(), m_host);

        // particles.update();

        particles.setParticleBC(bc);
    }

        //As a preliminary way we pass the function as an argument of pre_step()
    void pre_step()
        { 
            //particles.acceleration_function_external(func);
            cout << "before_update" << endl;
            particles.updateNeighbors();
            cout << "after_update" << endl;
            const std::size_t N_particles = particles.position.size();
            // TODO: Kokkos here?
            CubicSplineKernel<double, Dim> W;
            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
                auto nn = particles.CMHelper.neighbors(particles.position(p_idx));
                particles.density(p_idx) = 0.0;

                for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
                    const auto& other_pos = particles.position(*p_it);
                    ippl::Vector<double, Dim> d =  other_pos - particles.position(p_idx);
                    double rij = std::sqrt(d.dot(d));
                    particles.density(p_idx) += particles.mass(*p_it)*W(rij, h);
                    
                }
                particles.pressure(p_idx) = pow(particles.density(p_idx), Adiabatic_index);
                cout << "density: " << particles.density(p_idx) << endl;
            }

            for(std::size_t p_idx = 0; p_idx < N_particles; ++p_idx){
                auto nn = particles.CMHelper.neighbors(particles.position(p_idx));
                particles.accel(p_idx) = 0.0;
                for(auto p_it = nn.begin(); p_it != nn.end(); ++p_it){
                    const auto& other_pos = particles.position(*p_it);
                    ippl::Vector<double, Dim> d =  other_pos - particles.position(p_idx);
                    double rij = std::sqrt(d.dot(d));
                    double aux = ((particles.pressure(*p_it)/pow(particles.density(*p_it),2))-((particles.pressure(p_idx)/pow(particles.density(p_idx),2))));
                    particles.accel(p_idx) += (-((d)*W.grad_r(rij, h))/rij)*(particles.mass(*p_it))*aux;
                }
            }
            //std::vector<SizeListCollection<Dim>> vector_NN = particles.CMHelper.neighbor_lists;
            //cout << "vector_NN.size" << vector_NN.size() << endl;
            //N_nn = NN_array.size() // for each particle
            //for (int i = 0; i < N_nn; ++i) {particles.density(i) = sum_in_j(m[j] * CubicSplineKernel(particles.R(i)-NN_array[i][j]))
            //particles.pressure(i) = (adiabatic_index-1)*particles.density(i)*particles.energy_density(i)}
            //double aux = sum_in_j((particles.pressure(i)/pow(particles.density(i),2)) + (particles.pressure(j)/pow(particles.density(j),2)));
            //annoying part with pressure and density = ...
            //particles.acceleration(i) = sum_in_j(W(particles.R(i)-NN_array[i][j]))
            //particles.d_energy_density(i) = sum_in_j(W(particles.R(i)-NN_array[i][j])) 
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
            /* --- KDK scheme --- */
            // v_{i+1/2}

            //vector operators are not working?? Velocity is not changing vectorially

            particles.velocity = particles.velocity + particles.accel*dt/2.;
            // x_{i + 1}
            particles.R = particles.R + particles.velocity*dt;
            // Enforce bd conditions
            //particles.set_bd();
            // Compute acceleration at new position
            //particles.smoothen();
            // v_{i + 1}
            particles.velocity = particles.velocity + particles.accel*dt/2.;

            //How we compute energy density??? particles.energy_density = particles.energy_density + particles.d_energy_density*dt;??

            // For debugging only
            // std::cout << "Time-step\n";

            //particles.update();

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


