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
#include "Manager.h"


using namespace ippl;
using namespace std;


int main(int argc, char* argv[]) {

	initialize(argc, argv);
    {   
        constexpr unsigned int dim = 3;
 	    array<int, dim> points = {256, 256, 256};
 		int pt = 25;
 		Index I(pt);
 		NDIndex<dim> owned(I, I, I);
 		array<bool, dim> isParallel;
 		isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims
 		// all parallel layout, standard domain, normal axis order
 		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);	
 		double dx = 1.0 / 256.0;

 		Vector<double,dim> hx = {dx, dx, dx};
 		Vector<double,dim> origin = {0.0, 0.0, 0.0};
 		UniformCartesian<double,dim> mesh(owned, hx, origin);

 		ParticleSpatialLayout<double,3> myparticlelayout(layout, mesh);

        Manager<3, 10> manager(myparticlelayout);
        Vector<Vector<double, 3>, 10> R_part_0;
        Vector<Vector<double, 3>, 10> v_part_0;

        //Initializing random particle positions and velocities within Manager object

        for (int i = 0; i < 10; i++) {
            double i_double = static_cast<double>(i);
            R_part_0[i] = Vector<double, 3>(i_double*2.0, i_double*i_double, -3*i_double);
            v_part_0[i] = Vector<double, 3>(i_double*2.6, -i_double*i_double, 8.4*i_double);
        }

        manager.pre_run(R_part_0, v_part_0);
    }
	finalize();

	return 0;
}