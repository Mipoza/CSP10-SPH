#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include "../include/Ippl.h"

#include "../include/Expression/IpplExpressions.h"
// #include "../include/Expression/IpplOperators.h"

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
#include "../SPHParticle_radovan.hpp"
#include "../Manager.h"

// #include <SFML/Graphics.hpp>

using namespace ippl;
using namespace std;

const char* TestName = "IPPL_Test";

// Call this function once at the beginning of your program
void initialize_random_number_generator()
{
	srand((unsigned)time(NULL));
}

double rand_minus1_to_1()
{
	double random = (double)rand() / (double)RAND_MAX; // This generates a number between 0 and 1
	random = random * 2.0;							   // This expands the range to 0 to 2
	random = random - 1.0;							   // This shifts the range to -1 to 1
	return random;
}

int main(int argc, char *argv[])
{
	// int width = 800;
	// int height = 600;
	// sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");

	ippl::initialize(argc, argv);
	{

		constexpr unsigned int dim = 1;
		array<int, dim> points = {8};
		int pt = 8;
		Index I(pt);
		NDIndex<dim> owned(I);
		array<bool, dim> isParallel;
		isParallel.fill(false); // Specifies SERIAL, PARALLEL dims
		// all parallel layout, standard domain, normal axis order
		FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);
		double dx = 1.0 / 8.0;
		double dt = 0.001;

		Vector<double, dim> hx = {dx};
		Vector<double, dim> origin = {0.0};
		Vector<double, dim> fin = {1.0};
		UniformCartesian<double, dim> mesh(owned, hx, origin);

		ParticleSpatialLayout<double, 1> myparticlelayout(layout, mesh);

		double h = 0.075;

		Manager<1, 1> manager(myparticlelayout, origin, fin, dt, h, 1.4);
		std::vector<Vector<double, 1>> R_part_0;
		std::vector<Vector<double, 1>> v_part_0;
		std::vector<double> m_part_0;
		std::vector<double> E_part_0;

		// Initial conditions
		double rho_left = 1.0;  // Density on the left side
		double rho_right = 0.125;  // Density on the right side
		double v_left = 0.0;  // Velocity on the left side
		double v_right = 0.0;  // Velocity on the right side
		double p_left = 1.0;  // Pressure on the left side
		double p_right = 0.1;  // Pressure on the right side

		// Initializing random particle positions and velocities within Manager object
		int N_particles = 100;
		for (int j = 0; j < N_particles; j++) {
			double r1 = (rand_minus1_to_1() + 1) / 2;
			double rho = (r1 < 0.5) ? rho_left : rho_right;
			double v = (r1 < 0.5) ? v_left : v_right;
			double p = (r1 < 0.5) ? p_left : p_right;

			R_part_0.push_back(Vector<double, 1>(r1));
			v_part_0.push_back(Vector<double, 1>(v));
			m_part_0.push_back(1.0);
			E_part_0.push_back(p / (layout.getDomain().size() * dx));
		}

		array<ippl::BC, 2> bcs = {ippl::BC::REFLECTIVE, ippl::BC::REFLECTIVE};

		const bool visc = false;
		manager.pre_run(R_part_0, v_part_0, E_part_0, m_part_0, bcs);
		manager.pre_step(visc);

		const unsigned int N_times = 1000;
		// integration loop of the time eovlution

		for (unsigned int i = 0; i < N_times; i++)
		{
			// window.clear();

			float minRho = manager.particles.density(0);
			float maxRho = manager.particles.density(0);

			for (int k = 0; k < manager.particles.density.size(); k++)
			{
				if (minRho > manager.particles.density(k))
					minRho = manager.particles.density(k);
				if (maxRho < manager.particles.density(k))
					maxRho = manager.particles.density(k);
			}
			// auto pos = manager.particles.position;
			// for (int j = 0; j < pos.size(); j++) {
			// 	sf::CircleShape circle(4.0f);

			// 	float t = (manager.particles.density(j) - minRho) / ((maxRho - minRho + 0.001));
			// 	//cout << manager.particles.position(j)[1] << endl;
			// 	sf::Color color(static_cast<sf::Uint8>(255 * t), 0, static_cast<sf::Uint8>(255 * (1-t)));
			// 	//sf::Color color(0,0,255);
			// 	circle.setFillColor(color);

			// 	circle.setPosition(pos(j)[0]*width,pos(j)[1]*height);
			// 	// window.draw(circle);
			//   }

			manager.pre_step(visc);
			manager.advance();
			// window.display();

			std::cout << "energy_density " << manager.particles.energy_density(200) << std::endl;
			std::cout << "density " << manager.particles.density(200) << std::endl;
		}
	}
	ippl::finalize();

	return 0;
}
