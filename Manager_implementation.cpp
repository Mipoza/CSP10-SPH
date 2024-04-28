#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include "../include/Ippl.h"

#include "../include/Expression/IpplExpressions.h" 
//#include "include/Expression/IpplOperators.h" 

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

#include <SFML/Graphics.hpp>

using namespace ippl;
using namespace std;

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

Vector<double, 2> Repulsive_radial(Vector<double, 2> position) 
{
	double r = sqrt(position[0]*position[0] + position[1]*position[1]);
	double force = (r*r);
	return force * position;
}


int main(int argc, char* argv[]) {
	int width = 800;
	int height = 600;
	sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");

	initialize(argc, argv);
	{

        constexpr unsigned int dim = 2;
 	    array<int, dim> points = {8, 8};
 		int pt = 8;
 		Index I(pt);
 		NDIndex<dim> owned(I, I);
 		array<bool, dim> isParallel;
 		isParallel.fill(false);  // Specifies SERIAL, PARALLEL dims
 		// all parallel layout, standard domain, normal axis order
 		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);	
 		double dx = 1.0 / 8.0;
		double dt = 0.001;

 		Vector<double,dim> hx = {dx, dx};
 		Vector<double,dim> origin = {0.0, 0.0};
		Vector<double,dim> fin = {1.0, 1.0};
 		UniformCartesian<double,dim> mesh(owned, hx, origin);

 		ParticleSpatialLayout<double,2> myparticlelayout(layout, mesh);

        Manager<2, 2> manager(myparticlelayout, origin, fin, dt, 0.1);
        std::vector<Vector<double, 2>> R_part_0;
        std::vector<Vector<double, 2>> v_part_0;
		std::vector<double> m_part_0;
		std::vector<double> E_part_0;

        // //Initializing random particle positions and velocities within Manager object


		for (int i = 0; i < 1000; i++)
		{	double random_1 = 0.46*rand_minus1_to_1();
			double random_2 = 0.46*rand_minus1_to_1();

			R_part_0.push_back(Vector<double, 2>(i*0.001, 0.2));
			v_part_0.push_back(Vector<double, 2>(random_1, random_2));
			m_part_0.push_back(1.0);
			E_part_0.push_back(0.0);
		}

		std::array<ippl::BC,4> bcs = {ippl::BC::REFLECTIVE, ippl::BC::REFLECTIVE, ippl::BC::REFLECTIVE, ippl::BC::REFLECTIVE};

        manager.pre_run(R_part_0, v_part_0, E_part_0, m_part_0 , bcs);

		const unsigned int N_times = 10000;

		//integration loop of the time eovlution
		
		for (int i = 0; i < N_times; i++)
		{
			window.clear();

			float minRho = manager.particles.density(0);
			float maxRho = manager.particles.density(0);

			for(int k = 0; k < manager.particles.density.size(); k++){
				if(minRho > manager.particles.density(k))
					minRho = manager.particles.density(k);
				if(maxRho < manager.particles.density(k))
					maxRho = manager.particles.density(k);
			}


			auto pos = manager.particles.position;
			for (int j = 0; j < pos.size(); j++) {
				sf::CircleShape circle(8.0f);   

				

				float t = (manager.particles.density(j) - minRho) / ((maxRho - minRho + 0.001));
				//cout << manager.particles.position(j)[1] << endl;
				sf::Color color(static_cast<sf::Uint8>(255 * t), 0, static_cast<sf::Uint8>(255 * (1-t)));
				//sf::Color color(0,0,255);
				circle.setFillColor(color);

				circle.setPosition(pos(j)[0]*width,pos(j)[1]*height); 
				window.draw(circle);
        	}

			manager.pre_step();
			manager.advance();
			
			window.display();

		}
		//manager.advance();
			// cout << "density " << manager.particles.density(3) << endl;
			// cout << "mass " << manager.particles.mass(3) << endl;



    }
	finalize();

	return 0;
}