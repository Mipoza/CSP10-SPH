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

void runtime_test_2d(std::size_t N, T mesh_width) {
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

  auto start = std::chrono::high_resolution_clock::now();
  for(unsigned i = 0; i < nruns; ++i){
    CM.partition(pos_vec_view);
    CM.sort(pos_vec_view);
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto dur_ = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  double dur_insert = (double)dur_/nruns;

  Vec pt;
  pt[0] = 1 - mesh_width/3;
  pt[1] = 1 - mesh_width/3;

  start = std::chrono::high_resolution_clock::now();
  for(unsigned run = 0; run < nruns; ++run){
    // Loop over particles
    Kokkos::parallel_for(N, 
      KOKKOS_LAMBDA (const std::size_t p_idx){
        auto key = CM.cell_idx(pos_vec_view(p_idx));
        const auto my_neighbor_cells = CM.get_cell_neighbor_idx(key);
        /*
          Here goes some code (initialize variables used later etc.)
        */
        // Loop over neighbor cells
        for(std::size_t n_cell_idx = 0;
            n_cell_idx < my_neighbor_cells.size();
            ++n_cell_idx){
          // Loop over particles in neighbor cells
          const std::size_t start_idx_base = CM.start_idx(my_neighbor_cells[n_cell_idx]);
          for(std::size_t other_idx_ = 0;
              other_idx_ < CM.cell_size(my_neighbor_cells[n_cell_idx]);
              ++other_idx_){
            const std::size_t other_idx = start_idx_base + other_idx_;
            /*
              Here goes some code (do the actual compute)
            */
            some_view(p_idx) += pos_vec_view(p_idx)[0]*pos_vec_view(other_idx)[0]
                              + pos_vec_view(p_idx)[1]*pos_vec_view(other_idx)[1];
          }
        }
      }
    );
  }
  stop = std::chrono::high_resolution_clock::now();
  dur_ = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  double dur_it = (double)dur_/nruns;
  
  std::cout << "### " << N << ", " <<
    dur_insert << ", " << 
    dur_it << " ###" << std::endl;
}

int main(int argc, char* argv[]){
  Kokkos::initialize(argc, argv);
  if(argc != 3){
    std::cerr << "Invalid number of arguments." << std::endl;
    std::abort();
  }
  unsigned N = atoi(argv[1]);
  T h = atof(argv[2]);
  runtime_test_2d(N, h);
  Kokkos::finalize();
}
