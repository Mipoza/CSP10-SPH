
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
    return Dim;
}

template<unsigned int Dim, unsigned int N>
class Manager: public BaseManager {
public:
    using particle_position_type  = ParticleAttrib<Vector<double, Dim>>;

    SPH_Particle<Dim> particles;

    Manager(ParticleSpatialLayout<double,Dim>& L) : particles(L) {}

    void pre_run(Vector<Vector<double, Dim>,N> R_part, Vector<Vector<double, Dim>,N> v_part) 
    {
        typename Manager<Dim, N>::particle_position_type::HostMirror R_host = particles.R.getHostMirror();
        typename Manager<Dim, N>::particle_position_type::HostMirror v_host = particles.vel.getHostMirror();
        int N_new = getdim(R_part);

        particles.create(N_new);

        for (unsigned int i = 0; i < N_new; ++i) {

            R_host(i) = R_part[i];
            v_host(i) = v_host[i];

        }
        particles.update();
    }

        void pre_step() 
        { cout << "pre_step" << endl; }

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
        { cout << "advance" << endl; }

       /**
        * @brief The main for loop fro running a simulation.
        *
        * This method performs a simulation run by calling pre_step, advance, and post_step
        * in a loop for a specified number of time steps.
        *
        * @param nt The number of time steps to run the simulation.
        */
};


// int main(int argc, char* argv[]) {

// 	initialize(argc, argv);
//     {   
//         constexpr unsigned int dim = 3;
//  	    array<int, dim> points = {256, 256, 256};
//  		int pt = 25;
//  		Index I(pt);
//  		NDIndex<dim> owned(I, I, I);
//  		array<bool, dim> isParallel;
//  		isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims
//  		// all parallel layout, standard domain, normal axis order
//  		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);	
//  		double dx = 1.0 / 256.0;

//  		Vector<double,dim> hx = {dx, dx, dx};
//  		Vector<double,dim> origin = {0.0, 0.0, 0.0};
//  		UniformCartesian<double,dim> mesh(owned, hx, origin);

//  		ParticleSpatialLayout<double,3> myparticlelayout(layout, mesh);

//         Manager<3, 10> manager(myparticlelayout);
//         Vector<Vector<double, 3>, 10> R_part_0;
//         Vector<Vector<double, 3>, 10> v_part_0;

//         //Initializing random particle positions and velocities within Manager object

//         for (int i = 0; i < 10; i++) {
//             double i_double = static_cast<double>(i);
//             R_part_0[i] = Vector<double, 3>(i_double*2.0, i_double*i_double, -3*i_double);
//             v_part_0[i] = Vector<double, 3>(i_double*2.6, -i_double*i_double, 8.4*i_double);
//         }

//         manager.pre_run(R_part_0, v_part_0);
//     }
// 	finalize();

// 	return 0;
// }


