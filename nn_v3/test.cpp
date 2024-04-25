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

  std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
    Vec v;
    v[0] = randomT();
    v[1] = randomT();
    pos_vec.emplace_back(v);
  }
  std::cout << "Done" << std::endl;

  std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 2, true> CM(low, L, mesh_width);
  std::cout << "Done" << std::endl;

  std::cout << "Adding particles" << std::endl;
  CM.add_particles(pos_vec);
  std::cout << "Done" << std::endl;

  std::cout << "Creating Lists" << std::endl;
  CM.create_neighbor_lists();
  std::cout << "Done" << std::endl;


  Vec pt;
  pt[0] = randomT();
  pt[1] = randomT();
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
  file << "r" << std::endl;
  file << mesh_width << std::endl;
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

  std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
    Vec v;
    v[0] = randomT();
    v[1] = randomT();
    pos_vec.emplace_back(v);
  }
  std::cout << "Done" << std::endl;

  std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 2, true> CM(low, L, mesh_width);
  std::cout << "Done" << std::endl;

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

  CM.create_neighbor_lists();
}

int main(int argc, char* argv[]){
  if(argc != 4)
    std::abort();

  int type = atoi(argv[3]);

  std::size_t N = atoi(argv[1]);
  T h = atof(argv[2]);

  if(!type)
    runtime_test(N, h);
  else
    plot_2d(N, h);
}
