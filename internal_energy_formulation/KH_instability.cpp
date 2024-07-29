#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

#include "SPHManager.hpp"

#include <SFML/Graphics.hpp>


#define T double


// Call this function once at the beginning of your program
void initialize_random_number_generator() {
    srand((unsigned) time(NULL));
}

T rand_minus1_to_1() {
    T random = (T)rand() / (T)RAND_MAX; // This generates a number between 0 and 1
    random = random * 2.0; // This expands the range to 0 to 2
    random = random - 1.0; // This shifts the range to -1 to 1
    return random;
}


int main(int argc, char* argv[]) {
	int width = 800;
	int height = 800;
	sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");
  Kokkos::initialize(argc, argv);

  constexpr unsigned DIM = 2;

 	Vec<T, DIM> origin = {0.0, 0.0};
	Vec<T, DIM> extent = {1.0, 1.0};

  T CFL = 0.8;
  T dt_max = 1;
  unsigned N_particles = 2 << 10;
	const T h = 4/std::sqrt(N_particles);

  constexpr static const bool periodic[2] = {true, true};
  constexpr bool visc = true;
  constexpr bool balsara = true;
  const T alpha = 0.2;
  const T beta = 2*alpha;
  const T Adiabatic_index = 1.4;

  SPHManager<T, DIM, periodic, visc, balsara,
             CubicSplineKernel<T, DIM>> 
  manager(origin, extent, CFL, h, Adiabatic_index, dt_max, alpha, beta);
  std::vector<Vec<T, DIM>> R_part_0;
  std::vector<Vec<T, DIM>> v_part_0;
  std::vector<T> m_part_0;
  std::vector<T> uint_part_0;

  std::mt19937 gen_2(0); // Mersenne Twister pseudo-random generator, seeded with 0
  // Uniform distribution between 0 and 1
  std::uniform_real_distribution<> rand01(0.0, 1.0); 

  const T v = 10;
  const T amplitude_pert = 1e-1;
  const T amplitude_freq = 2;
  const T box_width = 1.0;
  const T box_height = 1.0;
  const T density_ = 1e3;
  const T velocity_1 = -v; // Velocity of the top layer
  const T velocity_2 =  v; // Velocity of the bottom layer

  // Example mass per particle assuming equal distribution initially
  const T mass = (density_ * box_width * box_height) / N_particles;
  const T A = 1e3 * mass;

  unsigned N_side = static_cast<unsigned>(std::sqrt(N_particles));
  const T dx = box_width / (N_side - 1);
  // const T dy = (box_height / 2.0) / N_side;
  const unsigned Ny = 2 * (N_side / 2);
  const T dy = box_height / (Ny - 1);

  const bool random_positions = false;

  for (unsigned i = 0; i < N_side - !random_positions; i++) {
    // T x = rand01(gen_2);
    for (unsigned j = 0; j < Ny/(1 + random_positions) - !random_positions; j++) {
      T x, y;
      if(random_positions){
        y = rand01(gen_2)*extent[1]/2.;
        x = rand01(gen_2)*extent[1];

        R_part_0.push_back(Vec<T, 2>(x, y));
        R_part_0.push_back(Vec<T, 2>(x, extent[1] - y));

        v_part_0.push_back(Vec<T, 2>(velocity_1, 0.0));
        v_part_0.push_back(Vec<T, 2>(velocity_2, 0.0));

        m_part_0.push_back(mass);
        m_part_0.push_back(mass);

        uint_part_0.push_back(A);
        uint_part_0.push_back(A);
      } else {
        x = (i + 0.5) * dx;
        y = (j + 0.5) * dy;
        //x = 1e-4 + i * dx * (1 - 2e-4);
        //y = 1e-4 + j * dy * (1 - 2e-4);
        R_part_0.push_back(Vec<T, 2>(x, y));
        v_part_0.push_back(Vec<T, 2>(
              y < 1/3. ? velocity_1 :(y > 2/3. ? velocity_1 : velocity_2), 
              ((std::abs(y - 1./3) < 1e-1) || (std::abs(y - 2./3) < 1e-1)) ? 
                amplitude_pert*std::sin(2*M_PI*amplitude_freq*x) : 0
              )
            );
        m_part_0.push_back(mass);
        uint_part_0.push_back(A);
      }
    }
  }

  const T target_number_density_factor = 8;
  manager.init(R_part_0, v_part_0, m_part_0, uint_part_0, 
               -target_number_density_factor);

  std::vector<T> position;
  std::vector<T> pressure;
  std::vector<T> density;
  std::vector<T> velocity;
  
  std::cout << std::scientific;

	//integration loop of the time eovlution
  window.clear(sf::Color::Black);

  auto color_q = [&](const std::size_t& p_idx){
    // return manager.pressure(p_idx);
    // return manager.density(p_idx);
    // return manager.pressure(p_idx)/std::pow(manager.density(p_idx), Adiabatic_index);
    // return manager.uint(p_idx);
    // return std::abs(manager.div_v(p_idx));
    // return std::sqrt(manager.velocity(p_idx).norm());
    // return std::abs(manager.velocity(p_idx)[1]);
    return manager.curl_v(p_idx).norm();
  };

  const unsigned print_freq = 16;
	const unsigned int N_times = 10000;
	for (unsigned int i = 0; i < N_times; i++){
		window.clear();
    manager.smoothen();

		float minv = color_q(0);
		float maxv = color_q(0);
    T avg = 0;
    T minp = manager.pressure(0);
    T maxp = 0;
    T minh = 1, maxh = 0;
    T mind = manager.density(0), maxd = 0;
    T minm = manager.compute_n_density(0, 
        manager.smoothing_kernel_sizes(0)/manager.h), maxm = 0,
      avgh = 0;
    std::size_t minn = manager.count_neighbors(0), maxn = 0;

		for(std::size_t k = 0; k < manager.density.size(); k++){
      minv = (minv > color_q(k) ? color_q(k) : minv);
      maxv = (maxv < color_q(k) ? color_q(k) : maxv);
      avg += color_q(k)/manager.density.size();
      avgh += manager.smoothing_kernel_sizes(k)/manager.density.size();
      minp = (minp > manager.pressure(k) ? manager.pressure(k) : minp); 
      maxp = (maxp < manager.pressure(k) ? manager.pressure(k) : maxp);
      T m = manager.compute_n_density(k, 
          manager.smoothing_kernel_sizes(k)/manager.h);
      minm = (minm > m ? m : minm);
      maxm = (maxm < m ? m : maxm);
      std::size_t nn = manager.count_neighbors(k);
      minn = (minn > nn ? nn : minn);
      maxn = (maxn < nn ? nn : maxn);

		}
    minh = *Kokkos::Experimental::min_element(Kokkos::DefaultExecutionSpace(),
        manager.smoothing_kernel_sizes);
    maxh = *Kokkos::Experimental::max_element(Kokkos::DefaultExecutionSpace(),
        manager.smoothing_kernel_sizes);

    mind = *Kokkos::Experimental::min_element(Kokkos::DefaultExecutionSpace(),
        manager.density);
    maxd = *Kokkos::Experimental::max_element(Kokkos::DefaultExecutionSpace(),
        manager.density);


		auto pos = Kokkos::create_mirror_view(manager.position);
		for (std::size_t j = 0; j < pos.size(); j++) {
			// sf::CircleShape circle(4.0f);

			// float t = (color_q(j) - minv) / ((maxv - minv + 0.001));
      float x = color_q(j);
			//float L0 = (x - avg)  * (x - maxv)/((minv - avg)  * (minv - maxv)),
      //      L1 = (x - minv) * (x - maxv)/((avg - minv)  * (avg - maxv)),
      //      L2 = (x - minv) * (x - avg) /((maxv - minv) * (maxv - avg));
      //float t = 0. * L0 + 0.5 * L1 + 1. * L2;
      float t = x/(maxv - minv) - minv/(maxv - minv);
			sf::Color color(static_cast<sf::Uint8>(255 * t),
                      0, 
                      static_cast<sf::Uint8>(255 * (1 - t)));
      //if(minn == manager.count_neighbors(j))
			//  color = sf::Color(0, 255, 0);

      //x = manager.smoothing_kernel_sizes(j);
			//L0 = (x - avgh)  * (x - maxh)/((minh - avgh)  * (minh - maxh));
      //L1 = (x - minh) * (x - maxh)/((avgh - minh)  * (avgh - maxh));
      //L2 = (x - minh) * (x - avgh) /((maxh - minh) * (maxh - avgh));

      //t = 0. * L0 + 0.5 * L1 + 1. * L2;

			sf::CircleShape circle(5);//. + 4.*t);

			circle.setFillColor(color);

			circle.setPosition(pos(j)[0]*width, pos(j)[1]*height); 
			window.draw(circle);
    }
    if(i % print_freq == 0){
      std::cout << "dt: " << manager.dt << std::endl;
      std::cout << "Color q. min/max: " << minv << "/" << maxv << std::endl;
      std::cout << "Pressure min/max: " << minp << "/" << maxp << std::endl;
      std::cout << "Density min/max: " << mind << "/" << maxd << std::endl;
      std::cout << "Kernel n density min/max: " << minm << "/" << maxm << std::endl;
      std::cout << "No Neibours min/max: " << minn << "/" << maxn << std::endl;
      std::cout << "h0: " << manager.h << std::endl;
      std::cout << "h:h0 min/max/avg: " << minh/manager.h << "/" 
                << maxh/manager.h << "/" << manager.avgh/manager.h << std::endl;
      std::cout << std::endl;
    }

		window.display();
    manager.step();
    // sleep(1);
  }

  Kokkos::finalize();

	return 0;
}
