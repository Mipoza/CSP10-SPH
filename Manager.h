
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

#include "include/Manager/BaseManager.h"
#include "Particle_SPH.h"


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

    SPH_Particle<Dim> particles;
    double dt;

    Manager(ParticleSpatialLayout<double,Dim>& L, double dt_) : dt(dt_), particles(L) {}

    void pre_run(Vector<Vector<double, Dim>,N> R_part, Vector<Vector<double, Dim>,N> v_part) 
    {
        int N_new = getdim(R_part);
        particles.create(N_new);

        typename Manager<Dim, N>::particle_position_type::HostMirror R_host = particles.R.getHostMirror();
        typename Manager<Dim, N>::particle_position_type::HostMirror v_host = particles.vel.getHostMirror();

        for (unsigned int i = 0; i < N_new; ++i) {

            R_host(i) = R_part[i];
            v_host(i) = v_part[i];

        }
        Kokkos::deep_copy(particles.R.getView(), R_host);
        Kokkos::deep_copy(particles.vel.getView(), v_host);

        particles.update();
    }

        //As a preliminary way we pass the function as an argument of pre_step()
        void pre_step(Vector<double, Dim> (*func)(Vector<double, Dim>)) 
        { 
            particles.acceleration_function_external(func);
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

            particles.vel = particles.vel + particles.acceleration*dt/2.;
            // x_{i + 1}
            particles.R = particles.R + particles.vel*dt;
            // Enforce bd conditions
            //particles.set_bd();
            // Compute acceleration at new position
            //particles.smoothen();
            // v_{i + 1}
            particles.vel = particles.vel + particles.acceleration*dt/2.;

            // For debugging only
            // std::cout << "Time-step\n";

            particles.update();

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


