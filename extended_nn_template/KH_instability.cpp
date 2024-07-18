#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

#include "SPHManager.h"

#include <SFML/Graphics.hpp>


#define T double

// Function to generate a random direction for 2D velocities
void randomDirection(T& vx, T& vy) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> distribution(0.0, 2 * M_PI);

    // Generate random angle
    T angle = distribution(gen);

    // Calculate components
    vx = std::cos(angle);
    vy = std::sin(angle);
}

T maxwellBoltzmann(T temperature, T mass) {
    // Boltzmann constant (in SI units)
    constexpr T kB = 1;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    // Generate uniform random number
    T u = distribution(gen);

    // Compute inverse CDF of Maxwell-Boltzmann distribution
    T v = std::sqrt(-2 * kB * temperature / mass * std::log(u));

    return v;
}

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
	int height = 600;
	sf::RenderWindow window(sf::VideoMode(width, height), "SPH Simulation");
  Kokkos::initialize(argc, argv);

  constexpr unsigned DIM = 2;

 	Vec<T, DIM> origin = {0.0, 0.0};
	Vec<T, DIM> extent = {1.0, 1.0};

	T h = 0.1;
  T dt = 1e-3;
  unsigned N_particles = 1024;//10'000;

  constexpr static const bool periodic[2] = {true, false};
  constexpr bool visc = true;
  SPHManager<T, DIM, periodic, visc> manager(origin, extent, dt, h, 1.4);
  std::vector<Vec<T, DIM>> R_part_0;
  std::vector<Vec<T, DIM>> v_part_0;
  std::vector<T> m_part_0;
  std::vector<T> entropy_part_0;

  //Initializing random particle positions and velocities within Manager object

  std::mt19937 gen_2(0); // Mersenne Twister pseudo-random generator, seeded with rd()
  // Uniform distribution between 0 and 1
  std::uniform_real_distribution<> rand01(0.0, 1.0); 

  const T box_width = 1.0;
  const T box_height = 1.0;
  const T density_ = 10.0;
  const T velocity_1 = 1; // Velocity of the top layer
  const T velocity_2 = -1; // Velocity of the bottom layer

  // Example mass per particle assuming equal distribution initially
  T mass = (density_ * box_width * box_height) / (N_particles / 2.0);


  unsigned N_side = static_cast<unsigned>(std::sqrt(N_particles / 2.0));
  T dx = box_width / N_side;
  T dy = (box_height / 2.0) / N_side;

  for (unsigned i = 0; i < N_side; i++) {
    for (unsigned j = 0; j < N_side; j++) {
      // Upper region
      T x = rand01(gen_2); //i * dx;
      T y = (box_height / 2.0) + rand01(gen_2)/2.; //j * dy;

      R_part_0.push_back(Vec<T, 2>(x, y));
      v_part_0.push_back(Vec<T, 2>(velocity_1, 0.0));
      m_part_0.push_back(mass);
      entropy_part_0.push_back(1.25);

      // Lower region
      y = extent[1] - y; //j * dy;
      R_part_0.push_back(Vec<T, 2>(x, y));
      v_part_0.push_back(Vec<T, 2>(velocity_2, 0.0));
      m_part_0.push_back(mass);
      entropy_part_0.push_back(1.25);
    }
  }

  const T mass_target_factor = 5;
  manager.init(R_part_0, v_part_0, m_part_0, entropy_part_0, -mass_target_factor);

  std::vector<T> position;
  std::vector<T> pressure;
  std::vector<T> density;
  std::vector<T> velocity;
  
  std::cout << std::scientific;

	const unsigned int N_times = 10000;
	//integration loop of the time eovlution
  window.clear(sf::Color::Black);

  auto color_q = [&](const std::size_t& p_idx){
    return manager.density(p_idx);
    //return std::sqrt(manager.velocity(p_idx)
    //                 .dot(manager.velocity(p_idx)));
  };

	manager.smoothen();
	for (unsigned int i = 0; i < N_times; i++){
		window.clear();

		float minv = color_q(0);
		float maxv = color_q(0);
    T avg = 0;

		for(int k = 0; k < manager.density.size(); k++){
      minv = (minv > color_q(k) ? color_q(k) : minv);
      maxv = (maxv < color_q(k) ? color_q(k) : maxv);
      avg += color_q(k)/manager.density.size();
		}

		auto pos = Kokkos::create_mirror_view(manager.position);
		for (int j = 0; j < pos.size(); j++) {
			sf::CircleShape circle(4.0f);

			// float t = (color_q(j) - minv) / ((maxv - minv + 0.001));
      const float x = color_q(j);
			const float L0 = (x - avg)  * (x - maxv)/((minv - avg)  * (minv - maxv)),
                  L1 = (x - minv) * (x - maxv)/((avg - minv)  * (avg - maxv)),
                  L2 = (x - minv) * (x - avg) /((maxv - minv) * (maxv - avg));
      const float t = 0 * L0 + 0.5 * L1 + 1. * L2;

			sf::Color color(static_cast<sf::Uint8>(255 * t),
                      0, 
                      static_cast<sf::Uint8>(255 * (1 - t)));
			circle.setFillColor(color);

			circle.setPosition(pos(j)[0]*width, pos(j)[1]*height); 
			window.draw(circle);
    }
    std::cout << "pos :" << manager.position(500) << std::endl;
    std::cout << "dens :" << manager.density(500) << std::endl;
    std::cout << "pressure :" << manager.pressure(500) << std::endl;
    std::cout << "entro :" << manager.entropy(500) << std::endl;
    std::cout << "d_entropy :" << manager.d_entropy(500) << std::endl;
    std::cout << "vel :" << manager.velocity(500) << std::endl;
    std::cout << "accel :" << manager.accel(500) << std::endl;
    std::cout << "h :" << manager.smoothing_kernel_sizes(500) << std::endl;
    std::cout << "Density min/max: " << minv << "/" << maxv << std::endl;

		window.display();

    manager.step();
  }

  Kokkos::finalize();

	return 0;
}
