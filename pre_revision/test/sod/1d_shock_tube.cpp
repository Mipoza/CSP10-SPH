#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

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
#include "Manager.h"

//#include <SFML/Graphics.hpp>

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
	// int width = 800;
	// int height = 600;
	// sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");
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

 		Vector<double,dim> hx = {dx};
 		Vector<double,dim> origin = {-10.0};
		Vector<double,dim> extent = {20.0};
 		UniformCartesian<double, dim> mesh(owned, hx, origin);

 		ParticleSpatialLayout<double, 1> myparticlelayout(layout, mesh);

		double h = 0.05;
    double dt = 1e-3;

    Manager<1> manager(myparticlelayout, origin, extent, dt, h, 1.4);
    std::vector<Vector<double, 1>> R_part_0;
    std::vector<Vector<double, 1>> v_part_0;
		std::vector<double> m_part_0;
		std::vector<double> E_part_0;
    std::vector<double> entropy_part_0;

    // //Initializing random particle positions and velocities within Manager object

    const double mass = 1.0;
    double T = 0.0;
    std::random_device rd;  // Non-deterministic random number generator
    std::mt19937 gen(rd()); // Mersenne Twister pseudo-random generator, seeded with rd()
    std::uniform_real_distribution<> dist(0.0, 1.0); // Uniform distribution between 0 and 1

    const unsigned N_particles = 1000;
    const unsigned fac = 5;
    for(unsigned i = 0; i < fac*N_particles; ++i){
        double x = dist(gen)/2.0;
        double v = maxwellBoltzmann(T, mass);

        R_part_0.push_back(Vector<double, 1>(origin[0] + extent[0]*i/((double)N_particles*fac) * 0.5));
        v_part_0.push_back(Vector<double, 1>(v));
        m_part_0.push_back(mass);
        E_part_0.push_back(10.0);
        entropy_part_0.push_back(1.0);
    }


    double T2 = 0.0; 
    std::random_device rd_2;  // Non-deterministic random number generator
    std::mt19937 gen_2(rd_2()); // Mersenne Twister pseudo-random generator, seeded with rd()
    std::uniform_real_distribution<> dist_2(0.0, 1.0); // Uniform distribution between 0 and 1

    for (unsigned i = 0; i < N_particles; i++)
    {   
        double x = dist_2(gen_2)/2.0;
        double v = maxwellBoltzmann(T2, mass);
        R_part_0.push_back(Vector<double, 1>(extent[0]*i/((double)N_particles) * 0.5));
        v_part_0.push_back(Vector<double, 1>(v));
        m_part_0.push_back(mass);
        E_part_0.push_back(10.0);
        entropy_part_0.push_back(1.25);
    }


    const bool visc = true;
    manager.pre_run(R_part_0, v_part_0, E_part_0, m_part_0, entropy_part_0);
		//manager.pre_step(visc);


        // std::vector<double> position;
        // std::vector<double> pressure;
        // for (unsigned int i = 0; i < 50; i++)
        // {
        //     position.push_back(i*0.019);
        //     pressure.push_back(manager.pressure_arbitrary_pos(i*0.019));
        // }

        // std::ofstream file_p("pressures_pos_new.dat");

        // // Write the current time and positions to the file
        // for (unsigned int j = 0; j < position.size(); j++)
        // {
        //     file_p  << ' ' << position[j] << ' ' << pressure[j] << '\n';
        // }

	    // file_p.close();


    std::vector<double> position;
    std::vector<double> pressure;
    std::vector<double> density;
    std::vector<double> velocity;

		const unsigned int N_times = 100;
		//integration loop of the time eovlution

		for (unsigned int i = 0; i < N_times; i++)
		{
		// 	window.clear();

		// 	float minRho = manager.particles.density(0);
		// 	float maxRho = manager.particles.density(0);

		// 	for(int k = 0; k < manager.particles.density.size(); k++){
		// 		if(minRho > manager.particles.density(k))
		// 			minRho = manager.particles.density(k);
		// 		if(maxRho < manager.particles.density(k))
		// 			maxRho = manager.particles.density(k);
		// 	}
		// 	auto pos = manager.particles.position;
		// 	for (int j = 0; j < pos.size(); j++) {
		// 		sf::CircleShape circle(4.0f);   

		// 		float t = (manager.particles.density(j) - minRho) / ((maxRho - minRho + 0.001));
		// 		//cout << manager.particles.position(j)[1] << endl;
		// 		sf::Color color(static_cast<sf::Uint8>(255 * t), 0, static_cast<sf::Uint8>(255 * (1-t)));
		// 		//sf::Color color(0,0,255);
		// 		circle.setFillColor(color);

		// 		circle.setPosition((10 + pos(j)[0])*width/20, 0.5*height); 
		// 		window.draw(circle);
    //   }
		// 	window.display();

			manager.pre_step(visc);
			// manager.pre_step(visc);
            //cout << "density " << manager.particles.density(200) << endl;
            // cout << "energy_density " << manager.particles.energy_density(200) << endl;
            // cout << "entropy " << manager.particles.entropy(200) << endl;
            // cout << "d_entropy " << manager.particles.d_entropy(200) << endl;
			// cout << "pressure " << manager.particles.pressure(200) << endl;
            // cout << "accel " << manager.particles.accel(200) << endl;
        

      // std::vector<double>().swap(position);
      // std::vector<double>().swap(pressure);
      // std::vector<double>().swap(density);
      // std::vector<double>().swap(velocity);
      position.clear();
      pressure.clear();
      density.clear();
      velocity.clear();

      double x_length;
      for (unsigned int i = 0; i < 5000; i++)
      {
          x_length = -5.0+i*0.002;
          position.push_back(x_length);
          pressure.push_back(manager.pressure_arbitrary_pos(x_length));
          density.push_back(manager.density_arbitrary_pos(x_length));
          velocity.push_back(manager.velocity_arbitrary_pos(x_length));

      }

      std::ofstream file_p, file_rho, file_v;
      if(visc){
        file_p.open("../pressures/pressures_pos_neww_f" + std::to_string(i) + ".dat");
        file_rho.open("../densities/densities_pos_neww_f" + std::to_string(i) + ".dat");
        file_v.open("../velocities/velocities_pos_neww_f" + std::to_string(i) + ".dat");
      } else{
        file_p.open("../pressures/pressures_pos_no_visc" + std::to_string(i) + ".dat");
        file_rho.open("../densities/densities_pos_no_visc" + std::to_string(i) + ".dat");
        file_v.open("../velocities/velocities_pos_no_visc" + std::to_string(i) + ".dat");
      }

      // Write the current time and positions to the file
      for (unsigned int j = 0; j < position.size(); j++)
      {
          file_p  << ' ' << position[j] << ' ' << pressure[j] << '\n';
          file_rho << ' ' << position[j] << ' ' << density[j] << '\n';
          file_v << ' ' << position[j] << ' ' << velocity[j] << '\n';
      }

      file_p.close();
      file_rho.close();
      file_v.close(); 

      manager.advance();
      cout << "Time step " << i << endl;
		}

        // std::vector<double> position;
        // std::vector<double> pressure;
        // for (unsigned int i = 0; i < 500; i++)
        // {
        //     position.push_back(i*0.019);
        //     pressure.push_back(manager.pressure_arbitrary_pos(i*0.019));
        // }

        // std::ofstream file_p("pressures_pos.dat");

        // // Write the current time and positions to the file
        // for (unsigned int j = 0; j < position.size(); j++)
        // {
        //     file_p  << ' ' << position[j] << ' ' << pressure[j] << '\n';
        // }

	    // file_p.close();



  }
	finalize();

	return 0;
}
