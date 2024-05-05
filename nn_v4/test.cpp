#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include "ChainingMesh.hpp"
#include <fstream>
#include <ctime>
#include <chrono>
#include <unistd.h>

#define T float
#define nruns 64
#define PERIODIC true


T randomT(){
  return (T)(std::rand())/(T)(RAND_MAX);
}

void plot_2d(std::size_t N, T mesh_width) {
  typedef std::array<T, 2> Vec;
  std::srand(std::time(nullptr));

  const T x0 = 0.,
          y0 = 0.;
  const Vec low = {x0, y0};
  const Vec L = {1., 1.};
  
  Kokkos::View<Vec*> pos_vec_view("pos_vec", N);
  Kokkos::View<T*> some_view("some_view", N);
  Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), some_view, 0);
  auto pos_vec_host = Kokkos::create_mirror_view(pos_vec_view);
  for(unsigned int i = 0; i < N; ++i){
    Vec v;
    v[0] = randomT();
    v[1] = randomT();
    pos_vec_host(i) = v;
  }
  Kokkos::deep_copy(pos_vec_view, pos_vec_host);

  ChainingMeshHelper<T, 2, PERIODIC> CM(low, L, mesh_width);

  CM.partition(pos_vec_view);

  Vec pt;
  pt[0] = 1 - mesh_width/3;
  pt[1] = 1 - mesh_width/3;

  CM.sort(pos_vec_view);

  std::ofstream file;
  auto pos_vec = Kokkos::create_mirror_view(pos_vec_view);

  file.open("../data_2d/pos.csv");
  file << "x,y" << std::endl;
  for(std::size_t i = 0; i < N; ++i)
    file << pos_vec(i)[0] << "," << pos_vec(i)[1] << std::endl;
  file.close();

  file.open("../data_2d/neighbors.csv");
  file << "x,y" << std::endl;
  auto key = CM.cell_idx(pt);
  auto my_neighbor_cells = CM.get_cell_neighbor_idx(key);
  auto cell_size = Kokkos::create_mirror_view(CM.cell_size);
  auto start_idx = Kokkos::create_mirror_view(CM.start_idx);
  // Loop over neighbor cells
  for(std::size_t n_cell_idx = 0; n_cell_idx < my_neighbor_cells.size(); ++n_cell_idx){ 
    // Loop over particles in neighbor cells
    for(std::size_t i_ = 0; i_ < cell_size(my_neighbor_cells[n_cell_idx]); ++i_){
      // Get starting index
      const std::size_t i = start_idx(my_neighbor_cells[n_cell_idx]) + i_;
      file << pos_vec(i)[0] << "," << pos_vec(i)[1] << std::endl;
    }
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

void plot_1d(std::size_t N, T mesh_width) {
  typedef std::array<T, 1> Vec;
  std::srand(std::time(nullptr));

  const T x0 = 0;
  const Vec low = {x0};
  const Vec L = {1.};
  
  Kokkos::View<Vec*> pos_vec_view("pos_vec", N);
  Kokkos::View<T*> some_view("some_view", N);
  Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), some_view, 0);
  auto pos_vec_host = Kokkos::create_mirror_view(pos_vec_view);
  for(unsigned int i = 0; i < N; ++i){
    Vec v;
    v[0] = randomT();
    pos_vec_host(i) = v;
  }
  Kokkos::deep_copy(pos_vec_view, pos_vec_host);

  ChainingMeshHelper<T, 1, PERIODIC> CM(low, L, mesh_width);

  CM.partition(pos_vec_view);

  Vec pt;
  pt[0] = 1 - mesh_width/3;

  CM.sort(pos_vec_view);

  std::ofstream file;
  auto pos_vec = Kokkos::create_mirror_view(pos_vec_view);

  file.open("../data_1d/pos.csv");
  file << "x,y" << std::endl;
  for(std::size_t i = 0; i < N; ++i)
    file << pos_vec(i)[0] << std::endl;
  file.close();

  file.open("../data_1d/neighbors.csv");
  file << "x" << std::endl;
  auto key = CM.cell_idx(pt);
  auto my_neighbor_cells = CM.get_cell_neighbor_idx(key);
  auto cell_size = Kokkos::create_mirror_view(CM.cell_size);
  auto start_idx = Kokkos::create_mirror_view(CM.start_idx);
  // Loop over neighbor cells
  for(std::size_t n_cell_idx = 0; n_cell_idx < my_neighbor_cells.size(); ++n_cell_idx){ 
    // Loop over particles in neighbor cells
    for(std::size_t i_ = 0; i_ < cell_size(my_neighbor_cells[n_cell_idx]); ++i_){
      // Get starting index
      const std::size_t i = start_idx(my_neighbor_cells[n_cell_idx]) + i_;
      file << pos_vec(i)[0] << std::endl;
    }
  }
  file.close();

  file.open("../data_1d/particle.csv");
  file << "x,y" << std::endl;
  file << pt[0] << std::endl;
  file.close();

  file.open("../data_1d/rad.csv");
  file << "r, nx" << std::endl;
  file << mesh_width << ", " << CM.NN[0] << std::endl;
  file.close();
}

int main(int argc, char* argv[]){
  Kokkos::initialize(argc, argv);
  std::size_t N = 10'000;
  T h = 0.125;
  plot_2d(N, h);

  N = 64;
  h = 0.1;
  plot_1d(N, h);

  Kokkos::finalize();
}
