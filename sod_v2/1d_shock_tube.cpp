#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

#include "SPHManager.hpp"
#include "std_array_overloads.hpp"

#include <SFML/Graphics.hpp>

#define T float


int main(int argc, char* argv[]) {
	const int width = 800;
	const int height = 800;
	sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");
  Kokkos::ScopeGuard kokkos(argc, argv);

  constexpr unsigned int DIM = 1;

  const Vec<T, DIM> origin = {0.0};
	const Vec<T, DIM> extent = {1.0};

  const T dt_max = 1;
  const T target_number_density_factor = 8;

  constexpr bool visc = true;
  constexpr bool balsara = false;
  const T alpha = 1.2;
  const T beta = 2*alpha;
  const T CFL = 0.8;
  const T Adiabatic_index = 1.4;
  // Ideal EOS
  auto internal_energy = [&] (const T A, const T rho){
    return A * std::pow(rho, Adiabatic_index - 1) / (Adiabatic_index - 1) ;
  };

  const unsigned N_particles = 2 << 8;
  const unsigned fac = 8;
  const unsigned N_tot = (fac + 1) * N_particles;
  // Just set it to something, the simulation will adjust it if needed
  // NOTE: The target number density depends on the initial h
	const T h = target_number_density_factor/N_particles;

  const T entr_left = 1.0;
  const T entr_right = 1.25;
  const T rhoL = 1;
  const T rhoR = rhoL/fac;

  const T mass = rhoR * extent[0] * 0.5 / N_particles;

  const T uL = internal_energy(entr_left, rhoL);
  const T uR = internal_energy(entr_right, rhoR);

  const unsigned no_plotted_particles = 2 << 8;
  const unsigned plot_skip = MAX(1, N_tot/no_plotted_particles);

  constexpr static const bool periodic[1] = {false};

  SPHManager<T, DIM, periodic, visc, balsara,
             CubicSplineKernel<T, DIM>> 
  manager(origin, extent, CFL, h, Adiabatic_index, dt_max, alpha, beta);

  std::vector<Vec<T, 1>> R_part_0;
  std::vector<Vec<T, 1>> v_part_0;
	std::vector<T> m_part_0;
  std::vector<T> uint_part_0;

  std::cout << N_tot << " particles in the simulation" << std::endl;

  for(unsigned i = 0; i < fac*N_particles - 1; ++i){
      T x = origin[0] + 0.5 * (i + 0.5) * extent[0]/(fac*N_particles - 1);
      R_part_0.push_back(Vec<T, 1>(x));
      v_part_0.push_back(Vec<T, 1>(0));
      m_part_0.push_back(mass);
      uint_part_0.push_back(uL);
  }

  for (unsigned i = 0; i < N_particles - 1; i++){   
      T x = origin[0] + extent[0] / 2. +
            0.5 * (i + 0.5) * extent[0]/(N_particles - 1);
      R_part_0.push_back(Vec<T, 1>(x));
      v_part_0.push_back(Vec<T, 1>(0));
      m_part_0.push_back(mass);
      uint_part_0.push_back(uR);
  }

  manager.init(R_part_0, v_part_0, m_part_0, uint_part_0, 
               -target_number_density_factor);

  std::vector<T> position;
  std::vector<T> pressure;
  std::vector<T> density;
  std::vector<T> velocity;

  const T t_final = 0.25;

  std::ofstream file;
  if(visc)
    file.open("../data/visc.dat");
  else
    file.open("../data/no_visc.dat");
  file << "x,pressure,density,velocity,i,t\n";
	//integration loop of the time evolution

  T t = 0;

	for (unsigned int i = 0; t < t_final; i++){
		window.clear();
    manager.smoothen();

		float minRho = manager.density(0);
		float maxRho = manager.density(0);

		for(unsigned k = 0; k < manager.density.size(); k++){
			if(minRho > manager.density(k))
				minRho = manager.density(k);
			if(maxRho < manager.density(k))
				maxRho = manager.density(k);
		}

		auto pos = manager.position;
		for (unsigned j = 0; j < pos.size(); j++) {
			sf::CircleShape circle(4.0f);   

			float t = (manager.density(j) - minRho) / ((maxRho - minRho + 0.001));
			//cout << manager.position(j)[1] << std::endl;
			sf::Color color(static_cast<sf::Uint8>(255 * t), 
                      0, 
                      static_cast<sf::Uint8>(255 * (1-t)));
			//sf::Color color(0,0,255);
			circle.setFillColor(color);

			circle.setPosition((pos(j)[0] - origin[0])*width/extent[0], 0.5*height); 
			window.draw(circle);
    }

    position.clear();
    pressure.clear();
    density.clear();
    velocity.clear();

    for (unsigned int j = 0; j < manager.position.size(); j += plot_skip){
      file << manager.position(j)[0] << ',' << manager.pressure(j) << ',' 
           << manager.density(j) << ',' << manager.velocity(j)[0] << ','
           << i << ',' << t << std::endl;
    }

		window.display();
    manager.step();
    t += manager.dt;
    std::cout << "Time step " << i <<  ", t = " << t << std::endl;
	}
  
  file.close();

	return 0;
}
