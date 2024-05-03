#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

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
#include "Manager.h"


using namespace ippl;
using namespace std;


// Function to generate a random direction for 2D velocities
void randomDirection(double& vx, double& vy) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 2 * M_PI);

    // Generate random angle
    double angle = distribution(gen);

    // Calculate components
    vx = std::cos(angle);
    vy = std::sin(angle);
}

double maxwellBoltzmann(double temperature, double mass) {
    // Boltzmann constant (in SI units)
    constexpr double kB = 1;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Generate uniform random number
    double u = distribution(gen);

    // Compute inverse CDF of Maxwell-Boltzmann distribution
    double v = std::sqrt(-2 * kB * temperature / mass * std::log(u));

    return v;
}

// Call this function once at the beginning of your program
void initialize_random_number_generator() {
    srand((unsigned) time(NULL));
}

double rand_minus1_to_1() {
    double random = (double)rand() / (double)RAND_MAX; // This generates a number between 0 and 1
    random = random * 2.0; // This expands the range to 0 to 2
    random = random - 1.0; // This shifts the range to -1 to 1
    return random;
}


int main(int argc, char* argv[]) {

	initialize(argc, argv);
	{

    constexpr unsigned int dim = 1;
 	  array<int, dim> points = {8};
 		int pt = 8;
 		Index I(pt);
 		NDIndex<dim> owned(I);
 		array<bool, dim> isParallel;
 		isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims
 		// all parallel layout, standard domain, normal axis order
 		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);	
 		double dx = 1.0 / 8.0;
		double dt = 1e-4;

 		Vector<double,dim> hx = {dx, dx};
 		Vector<double,dim> origin = {0.0};
		Vector<double,dim> fin = {1.0};
 		UniformCartesian<double,dim> mesh(owned, hx, origin);

 		ParticleSpatialLayout<double, 1> myparticlelayout(layout, mesh);

		double h = 0.01;

        Manager<1> manager(myparticlelayout, origin, fin, dt, h, 1.4);
        std::vector<Vector<double, 1>> R_part_0;
        std::vector<Vector<double, 1>> v_part_0;
		std::vector<double> m_part_0;
		std::vector<double> E_part_0;
        std::vector<double> entropy_part_0;

    // //Initializing random particle positions and velocities within Manager object

        const double mass = 1e-3;
        double T = 2730000;
        std::random_device rd;  // Non-deterministic random number generator
        std::mt19937 gen(rd()); // Mersenne Twister pseudo-random generator, seeded with rd()
        std::uniform_real_distribution<> dist(0.0, 1.0); // Uniform distribution between 0 and 1
        const unsigned N_particles = 5000;
        for(unsigned i = 0; i < N_particles; ++i){
            double x = dist(gen)/2.0;
            double v = maxwellBoltzmann(mass, T);

            R_part_0.push_back(Vector<double, 1>(x));
            v_part_0.push_back(Vector<double, 1>(v));
            m_part_0.push_back(mass);
            E_part_0.push_back(10.0);
            entropy_part_0.push_back(1.0);
        }

        double T2 = 2730000;

        for (unsigned i = 0; i < 500; i++)
        {   
            double v = maxwellBoltzmann(mass, T2);
            R_part_0.push_back(Vector<double, 1>(0.51 + (i * 0.0007)));
            v_part_0.push_back(Vector<double, 1>(v));
            m_part_0.push_back(mass);
            E_part_0.push_back(10.0);
            entropy_part_0.push_back(1.0);
        }

		array<ippl::BC, 2> bcs = {ippl::BC::PERIODIC, ippl::BC::PERIODIC};

        const bool visc = true;
        manager.pre_run(R_part_0, v_part_0, E_part_0, m_part_0, entropy_part_0, bcs);
		manager.pre_step(visc);


		const unsigned int N_times = 1000;
		//integration loop of the time eovlution

		for (unsigned int i = 0; i < N_times; i++)
		{
			manager.pre_step(visc);
			manager.advance();

			cout << "energy_density " << manager.particles.energy_density(200) << endl;
            cout << "entropy " << manager.particles.entropy(200) << endl;
			cout << "density " << manager.particles.density(200) << endl;
			cout << "pressure " << manager.particles.pressure(200) << endl;
		}



    }
	finalize();

	return 0;
}