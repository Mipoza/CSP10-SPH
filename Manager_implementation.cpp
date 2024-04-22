#include <iostream>
#include <cmath>
#include <fstream>

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
#include "Manager.h"


using namespace ippl;
using namespace std;


Vector<double, 2> Repulsive_radial(Vector<double, 2> position) 
{
	double r = sqrt(position[0]*position[0] + position[1]*position[1]);
	double force = (r*r);
	return force * position;
}


int main(int argc, char* argv[]) {

	initialize(argc, argv);
    {   
        constexpr unsigned int dim = 2;
 	    array<int, dim> points = {256, 256};
 		int pt = 25;
 		Index I(pt);
 		NDIndex<dim> owned(I, I);
 		array<bool, dim> isParallel;
 		isParallel.fill(false);  // Specifies SERIAL, PARALLEL dims
 		// all parallel layout, standard domain, normal axis order
 		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);	
 		double dx = 1.0 / 256.0;
		double dt = 0.1;

 		Vector<double,dim> hx = {dx, dx};
 		Vector<double,dim> origin = {0.0, 0.0};
 		UniformCartesian<double,dim> mesh(owned, hx, origin);

 		ParticleSpatialLayout<double,2> myparticlelayout(layout, mesh);

        Manager<2, 2> manager(myparticlelayout, dt);
        Vector<Vector<double, 2>, 2> R_part_0;
        Vector<Vector<double, 2>, 2> v_part_0;

        //Initializing random particle positions and velocities within Manager object

    
        R_part_0[0] = Vector<double, 2>(0.1, 0.1);
		R_part_0[1] = Vector<double, 2>(0.1, -0.1);

        v_part_0[0] = Vector<double, 2>(-0.1, 0.0);
		v_part_0[1] = Vector<double, 2>(0.0, 0.0);


        manager.pre_run(R_part_0, v_part_0);

		const unsigned int N_times = 5;
		Vector<Vector<Vector<double, 2>, N_times>,2> Position_times;

		for (int i = 0; i < N_times; i++)
		{
			manager.pre_step(Repulsive_radial);
			manager.advance();
			Position_times[0][i] = manager.particles.R(0);
			Position_times[1][i] = manager.particles.R(1);
			
		}
		
    }
	finalize();

	return 0;
}