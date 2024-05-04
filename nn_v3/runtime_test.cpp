#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include "ChainingMesh.hpp"
#include <ctime>
#include <chrono>
#include <unistd.h>

#define T float
#define nruns 64


T randomT(){
  return (T)(std::rand())/(T)(RAND_MAX);
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

  auto start = std::chrono::high_resolution_clock::now();
  for(unsigned i = 0; i < nruns; ++i){
    CM.add_particles(pos_vec);
    CM.clear();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto dur_ = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  double dur_insert = (double)dur_/nruns;

  CM.add_particles(pos_vec);
  // std::cout << "Iterating through " << N << " particles" << std::endl;
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
  double dur_it = (double)dur_/nruns;

  std::cout << "### " << N << ", " <<
    dur_insert << ", " << 
    dur_it << " ###" << std::endl;

  CM.create_neighbor_lists();
}

int main(int argc, char* argv[]){
  if(argc != 3){
    std::cerr << "Invalid number of arguments." << std::endl;
    std::abort();
  }
  unsigned N = atoi(argv[1]);
  T h = atof(argv[2]);
  runtime_test(N, h);
}
