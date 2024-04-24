#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include "ChainingMesh.hpp"
#include <fstream>
#include <ctime>

#define T float


T randomT(){
  return (T)(std::rand())/(T)(RAND_MAX);
}

void plot_1d(){
  typedef std::array<T, 1> Vec;
  std::srand(std::time(nullptr));

  const int N = 32;
  const T x0 = 0.;
  const T dx = 1./N;
  const T mesh_width = 3./N;
  Vec low = {x0};
  Vec L = {N*dx};

  std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
      Vec v;
      v[0] = randomT();
      pos_vec.emplace_back(v);
  }
  std::cout << "Done" << std::endl;

  std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 1, true> CM(low, L, mesh_width);
  std::cout << "Done" << std::endl;

  std::cout << "Adding particles" << std::endl;
  for(unsigned i = 0; i < pos_vec.size(); ++i)
    CM.add_particle(pos_vec[i], i);
  std::cout << "Done" << std::endl;

  std::cout << "Creating Lists" << std::endl;
  CM.create_neighbor_lists();
  std::cout << "Done" << std::endl;


  Vec pt;
  pt[0] = randomT();
  auto adj_pts = CM.neighbors(pt);

  std::ofstream file;

  file.open("data_1d/pos.csv");
  file << "x" << std::endl;
  for(auto& p : pos_vec)
    file << p[0] << std::endl;
  file.close();

  file.open("data_1d/neighbors.csv");
  file << "x" << std::endl;
  for(auto it = adj_pts.begin(); it != adj_pts.end(); ++it){
    Vec p = pos_vec[*it];
    file << p[0] << std::endl;
  }
  file.close();

  file.open("data_1d/particle.csv");
  file << "x" << std::endl;
  file << pt[0] << std::endl;
  file.close();

  file.open("data_1d/rad.csv");
  file << "r" << std::endl;
  file << mesh_width << std::endl;
  file.close();
}

void plot_2d() {
  typedef std::array<T, 2> Vec;
  std::srand(std::time(nullptr));

  const int N = 64;
  const T x0 = 0.,
          y0 = 0.;
  const T dx = 1./N;
  const T mesh_width = 6./N;
  Vec low = {x0, y0};
  Vec L = {N*dx, N*dx};

  std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
    for(unsigned int j = 0; j < N; ++j){
      Vec v;
      v[0] = randomT();
      v[1] = randomT();
      pos_vec.emplace_back(v);
    }
  }
  std::cout << "Done" << std::endl;

  std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 2, true> CM(low, L, mesh_width);
  std::cout << "Done" << std::endl;

  std::cout << "Adding particles" << std::endl;
  for(unsigned i = 0; i < pos_vec.size(); ++i)
    CM.add_particle(pos_vec[i], i);
  std::cout << "Done" << std::endl;

  std::cout << "Creating Lists" << std::endl;
  CM.create_neighbor_lists();
  std::cout << "Done" << std::endl;


  Vec pt;
  pt[0] = randomT();
  pt[1] = randomT();
  auto adj_pts = CM.neighbors(pt);

  std::ofstream file;

  file.open("data_2d/pos.csv");
  file << "x,y" << std::endl;
  for(auto& p : pos_vec)
    file << p[0] << "," << p[1] << std::endl;
  file.close();

  file.open("data_2d/neighbors.csv");
  file << "x,y" << std::endl;
  for(auto it = adj_pts.begin(); it != adj_pts.end(); ++it){
    Vec p = pos_vec[*it];
    file << p[0] << "," << p[1] << std::endl;
  }
  file.close();

  file.open("data_2d/particle.csv");
  file << "x,y" << std::endl;
  file << pt[0] << "," << pt[1] << std::endl;
  file.close();

  file.open("data_2d/rad.csv");
  file << "r" << std::endl;
  file << mesh_width << std::endl;
  file.close();
}

void plot_3d() {
  typedef std::array<T, 3> Vec;
  std::srand(std::time(nullptr));

  const int N = 64;
  const T x0 = 0.,
          y0 = 0.,
          z0 = 0.;
  const T dx = 1./N;
  const T mesh_width = 1./N;
  Vec low = {x0, y0, z0};
  Vec L = {N*dx, N*dx, N*dx};

  std::cout << "Creating particles" << std::endl;
  std::vector<Vec> pos_vec;
  for(unsigned int i = 0; i < N; ++i){
    for(unsigned int j = 0; j < N; ++j){
      for(unsigned int k = 0; k < N; ++k){
        Vec v;
        v[0] = randomT();
        v[1] = randomT();
        v[2] = randomT();
        pos_vec.emplace_back(v);
      }
    }
  }
  std::cout << "Done" << std::endl;

  std::cout << "Initializing CM" << std::endl;
  ChainingMeshHelper<T, 3, true> CM(low, L, mesh_width);
  std::cout << "Done" << std::endl;

  std::cout << "Adding particles" << std::endl;
  for(unsigned i = 0; i < pos_vec.size(); ++i)
    CM.add_particle(pos_vec[i], i);
  std::cout << "Done" << std::endl;

  std::cout << "Creating Lists" << std::endl;
  CM.create_neighbor_lists();
  std::cout << "Done" << std::endl;


  Vec pt;
  pt[0] = randomT();
  pt[1] = randomT();
  pt[2] = randomT();
  auto adj_pts = CM.neighbors(pt);

  std::ofstream file;

  file.open("data_3d/pos.csv");
  file << "x,y,z" << std::endl;
  for(auto& p : pos_vec)
    file << p[0] << "," << p[1] << "," << p[2]<< std::endl;
  file.close();

  file.open("data_3d/neighbors.csv");
  file << "x,y,z" << std::endl;
  for(auto it = adj_pts.begin(); it != adj_pts.end(); ++it){
    Vec p = pos_vec[*it];
    file << p[0] << "," << p[1] << "," << p[2] << std::endl;
  }
  file.close();

  file.open("data_3d/particle.csv");
  file << "x,y,z" << std::endl;
  file << pt[0] << "," << pt[1] << "," << pt[2] << std::endl;
  file.close();

  file.open("data_3d/rad.csv");
  file << "r" << std::endl;
  file << mesh_width << std::endl;
  file.close();
}

int main(){
  plot_1d();
  plot_2d();
  // The plots take too many points, not helpful
  // plot_3d();
}
