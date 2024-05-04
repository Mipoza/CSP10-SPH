#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ratio>
#include <vector>
#include <array>
#include "ChainingMesh.hpp"
#include <fstream>
#include <ctime>
#include <chrono>
#include <unistd.h>

#define T float
#define nruns 64


T randomT(){
  return (T)(std::rand())/(T)(RAND_MAX);
}

void plot_2d(std::size_t N, T mesh_width) {
  typedef std::array<T, 2> Vec;
  std::srand(std::time(nullptr));

  // const int N = 64;
  const T x0 = 0.,
          y0 = 0.;
  // const T mesh_width = 6./N;
  Vec low = {x0, y0};
  Vec L = {1., 1.};

  // std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
    Vec v;
    v[0] = randomT();
    v[1] = randomT();
    pos_vec.emplace_back(v);
  }
  // std::cout << "Done" << std::endl;

  // std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 2, true> CM(low, L, mesh_width);
  // std::cout << "Done" << std::endl;

  // std::cout << "Adding particles" << std::endl;
  CM.add_particles(pos_vec);
  // std::cout << "Done" << std::endl;

  // std::cout << "Creating Lists" << std::endl;
  CM.create_neighbor_lists();
  // std::cout << "Done" << std::endl;


  Vec pt;
  pt[0] = 1 - mesh_width/2;
  pt[1] = 1 - mesh_width/2;
  auto adj_pts = CM.neighbors(pt);

  std::ofstream file;

  file.open("../data_2d/pos.csv");
  file << "x,y" << std::endl;
  for(auto& p : pos_vec)
    file << p[0] << "," << p[1] << std::endl;
  file.close();

  file.open("../data_2d/neighbors.csv");
  file << "x,y" << std::endl;
  for(auto it = adj_pts.begin(); it != adj_pts.end(); ++it){
    Vec p = pos_vec[*it];
    file << p[0] << "," << p[1] << std::endl;
  }
  file.close();

  file.open("../data_2d/particle.csv");
  file << "x,y" << std::endl;
  file << pt[0] << "," << pt[1] << std::endl;
  file.close();

  file.open("../data_2d/rad.csv");
  file << "r, nx, ny" << std::endl;
  file << mesh_width << ", " << CM.NN[0] << ", " << CM.NN[1] << std::endl;
  file.close();
}

void runtime_test(std::size_t N, T mesh_width) {
  typedef std::array<T, 2> Vec;
  std::srand(0);

  const T x0 = 0.,
          y0 = 0.;
  const T dx = 1./N;
  Vec low = {x0, y0};
  Vec L = {N*dx, N*dx};

  // std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
    Vec v;
    v[0] = randomT();
    v[1] = randomT();
    pos_vec.emplace_back(v);
  }
  // std::cout << "Done" << std::endl;

  // std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 2, true> CM(low, L, mesh_width);
  // std::cout << "Done" << std::endl;

  std::cout << "--- Averaging over " << nruns << " runs ---" << std::endl;

  std::cout << "Adding " << N << " particles" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  for(unsigned i = 0; i < nruns; ++i){
    CM.add_particles(pos_vec);
    CM.clear();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto dur_ = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  double dur = (double)dur_/nruns;

  std::cout << "Done, average time: " << dur << " microseconds" << std::endl;
  std::cout << "### " << N << ", " << dur << " ###" << std::endl;

  CM.add_particles(pos_vec);
  std::cout << "Iterating through " << N << " particles" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  for(unsigned run = 0; run < nruns; ++run){
    #pragma omp parallel for
    for(std::size_t p_idx = 0; p_idx < N; ++p_idx){
      const auto& my_pos = pos_vec[p_idx];
      auto nn = CM.neighbors(my_pos);
      T dummy = 0;
      for(auto it = nn.begin(); it != nn.end(); ++it){
        // Neighbor position
        const auto& other_pos = pos_vec[*it];
        // Compute some dummy quantity
        dummy += other_pos[0]*my_pos[0] + other_pos[1]*my_pos[1];
      }
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  dur_ = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  dur = (double)dur_/nruns;

  std::cout << "Done, average time: " << dur << " microseconds" << std::endl;
  std::cout << "### " << N << ", " << dur << " ###" << std::endl;

  CM.create_neighbor_lists();
}

int main(int argc, char* argv[]){
  constexpr std::size_t N = 5'000;
  constexpr T h = 0.1;
  plot_2d(N, h);

  constexpr std::size_t N_test = 1'000'000;
  constexpr T h_test = 0.01;
  runtime_test(N_test, h_test);
}
