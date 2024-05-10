#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

#include "../include/Ippl.h"

#include "../include/Expression/IpplExpressions.h" 
//#include "../../include/Expression/IpplOperators.h" 

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
 		isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims
 		// all parallel layout, standard domain, normal axis order
 		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);	
 		double dx = 1.0 / 8.0;
        //D

 		Vector<double,dim> hx = {dx, dx};
 		Vector<double,dim> origin = {0.0, 0.0};
		Vector<double,dim> extent = {1.0, 1.0};
 		UniformCartesian<double, dim> mesh(owned, hx, origin);

 		ParticleSpatialLayout<double, dim> myparticlelayout(layout, mesh);

		double h = 0.02;
        double dt = 1e-3;

        Manager<dim> manager(myparticlelayout, origin, extent, dt, h, 1.4);
        std::vector<Vector<double, dim>> R_part_0;
        std::vector<Vector<double, dim>> v_part_0;
        std::vector<double> m_part_0;
        std::vector<double> E_part_0;
        std::vector<double> entropy_part_0;

        std::random_device rd_2;  // Non-deterministic random number generator
        std::mt19937 gen_2(rd_2()); // Mersenne Twister pseudo-random generator, seeded with rd()
        std::uniform_real_distribution<> dist_2(0.0, 1.0); // Uniform distribution between 0 and 1
        unsigned N_particles = 10000;
        double box_width = 1.0;
        double box_height = 1.0;
        double density_1 = 1.0; // Density of the top layer
        double density_2 = 1.0; // Density of the bottom layer
        double velocity_1 = 5; // Velocity of the top layer
        double velocity_2 = -5; // Velocity of the bottom layer

        // Example mass per particle assuming equal distribution initially
        double mass_1 = 1;
        double mass_2 = 1;
 
        unsigned N_side = static_cast<unsigned>(std::sqrt(N_particles / 2.0));
        dx = box_width / N_side;
        double dy = (box_height / 2.0) / N_side;

        for (unsigned i = 0; i < N_side; i++) {
                for (unsigned j = 0; j < N_side; j++) {
                    // Upper region
                    double x = i * dx;
                    double y = (box_height / 2.0) + j * dy;

                    R_part_0.push_back(Vector<double, 2>(x, y ));
                    v_part_0.push_back(Vector<double, 2>(velocity_1, 0.0));
                    m_part_0.push_back(mass_1);
                    E_part_0.push_back(10.0);
                    entropy_part_0.push_back(1.25);
                    
                    if(i%1 == 0 || j %1  == 0){
                      // Lower region
                      y = j * dy ;
                      R_part_0.push_back(Vector<double, 2>(x, y ));
                      v_part_0.push_back(Vector<double, 2>(velocity_2, 0.0));
                      m_part_0.push_back(mass_2);
                      E_part_0.push_back(10.0);
                      entropy_part_0.push_back(1.25);
                    }
                }
            }



    const bool visc = true;
    manager.pre_run(R_part_0, v_part_0, E_part_0, m_part_0, entropy_part_0);


		const unsigned int N_times = 10000;
		//integration loop of the time eovlution

		for (unsigned int i = 0; i < N_times; i++)
		{
			window.clear();

			float minRho = manager.particles.density(0);
			float maxRho = manager.particles.density(0);

			for(int k = 0; k < manager.particles.density.size(); k++){
                //if(i==1)
                //    std::cout << manager.particles.density(k) << std::endl;
				if(minRho > manager.particles.density(k))
					minRho = manager.particles.density(k);
				if(maxRho < manager.particles.density(k))
					maxRho = manager.particles.density(k);
			}
			auto pos = manager.particles.position;
            if(i==1)
                std::cout << minRho << " " << maxRho << std::endl;
			for (int j = 0; j < pos.size(); j++) {
				sf::CircleShape circle(4.0f);   

				float t = (manager.particles.density(j) - minRho) / ((maxRho - minRho + 0.001));
				sf::Color color(static_cast<sf::Uint8>(255 * t), 0, static_cast<sf::Uint8>(255 * (1-t)));
				circle.setFillColor(color);

				circle.setPosition(pos(j)[0]*width,pos(j)[1]*height); 
				window.draw(circle);
            }
			window.display();

			manager.pre_step(visc);
            manager.advance();
        }


  }
	finalize();

	return 0;
}
