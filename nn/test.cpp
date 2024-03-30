#include "particle.hpp"
#include "nn.hpp"
#include <iostream>
#include <vector>

#define DIM 2
#define SCALAR double

typedef Particle<SCALAR, DIM> PARTICLE;

int main() {
  const int N = 1000;
  const double x0 = 0.5,
               y0 = 0.5;
  const double dx = 1e-2;

  std::vector<PARTICLE> p_vec;
  for(unsigned int i = 0; i < N; ++i){
    for(unsigned int j = 0; j < N; ++j){
      PARTICLE P(i*dx + x0, j*dx + y0);
      p_vec.push_back(P);
    }
  }
  
  std::cout << "Building Chainig-Mesh" << std::endl;
  ChainingMesh<SCALAR, DIM, PARTICLE> CM(p_vec, 0.1); 
  std::cout << "Done" << std::endl;
  // It seems to work as intended from a few simple tests
  // try playing around with this
  auto uuh = CM.at({2, 1});
  for(auto it = uuh.first; it != uuh.second; ++it){
    PARTICLE p = *(it->second);
    std::cout << p << CM.idx_to_key(p.chaining_mesh_idx) << std::endl;
  }
  return 0;
}
