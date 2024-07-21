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
#include "std_array_overloads.hpp"
#include <iomanip>
#include <unordered_set>

#define T float
#define MAX(x, y) (x > y ? x : y)
#define MIN(x, y) (x < y ? x : y)

T randomT(){
  return (T)(std::rand())/(T)(RAND_MAX);
}

typedef Kokkos::View<std::unordered_multiset<std::size_t>*> IDX_SET;

// Compute the distance with periodicity in mind
template <unsigned DIM, const bool PERIODIC[DIM]>
inline T unit_sq_periodic_dist(const Vec<T, DIM>& x, const Vec<T, DIM>& y){
  Vec<T, DIM> mindist_vec;

  for(unsigned d = 0; d < DIM; ++d){
    mindist_vec[d] = std::abs(y[d] - x[d]);
    if(PERIODIC[d]){
      mindist_vec[d] = MIN(std::abs(y[d] - (x[d] + static_cast<T>(1))),
                           mindist_vec[d]);
      mindist_vec[d] = MIN(std::abs((y[d] + static_cast<T>(1)) - x[d]),
                           mindist_vec[d]);
    }
  }

  return std::sqrt(mindist_vec.dot(mindist_vec));
}

// Perform a simple, naive reduction over the "neighbors"
template <unsigned DIM, const bool PERIODIC[DIM], typename VEC_ARR, 
          typename VEC_T, typename VEC_I>
IDX_SET naive_reduction(const VEC_T& h, 
                        const VEC_ARR& positions, 
                        const VEC_I& id){
  const std::size_t N = positions.size();
  IDX_SET result("Result", N);

  Kokkos::parallel_for(N,
    KOKKOS_LAMBDA (const std::size_t i){
      for(std::size_t j = 0; j < N; ++j)
        if(unit_sq_periodic_dist<DIM, PERIODIC>(positions(i), positions(j)) < h(i))
          result(i).insert(id(j));
    }
  );

  return result;
}

template <unsigned DIM, const bool PERIODIC[DIM]>
void test_cm(const unsigned N,
             const T min_h,
             const T max_h,
             const unsigned N_passes = 1,
             const int seed = 0){
  std::srand(seed);

  Vec<T, DIM> origin, L;
  for(unsigned d = 0; d < DIM; ++d){
    origin[d] = std::sqrt(2) * d;
    L[d] = 1;
  }

  const T h_mesh = (min_h + max_h)/2.;
  ChainingMesh<T, DIM, PERIODIC> CM(origin, L, h_mesh);

  Kokkos::View<Vec<T, DIM>*> pos_vec_view("pos_vec", N);
  Kokkos::View<std::size_t*> id("id", N);
  Kokkos::View<T*> h("smoothing kernel lengths", N);

  // Randomize positions, give new IDs and smoothing lengths
  auto shuffle = [&](){
    auto pos_vec_host = Kokkos::create_mirror_view(pos_vec_view);
    auto id_host = Kokkos::create_mirror_view(id);
    auto h_host = Kokkos::create_mirror_view(h);
    for(std::size_t i = 0; i < N; ++i){
      Vec<T, DIM> v;
      for(unsigned d = 0; d < DIM; ++d) v[d] = origin[d] + L[d] * randomT();

      pos_vec_host(i) = v;
      id_host(i) = i; // id
      h(i) = min_h + randomT() * (max_h - min_h); // smoothing kernel length 
    }
    Kokkos::deep_copy(pos_vec_view, pos_vec_host);
    Kokkos::deep_copy(id, id_host);
    Kokkos::deep_copy(h, h_host);
  };

  std::cout << "Pass\t" << "No. incorrect points" << std::endl;
  for(unsigned k = 0; k < N_passes; ++k){
    shuffle();
    // Reduce with the ChainingMesh first,
    // since it re-orders the particles
    CM.partition(pos_vec_view);
    CM.sort(pos_vec_view, id);
    CM.sort(pos_vec_view, h);
    CM.sort(pos_vec_view, pos_vec_view);

    auto pos_vec_host = Kokkos::create_mirror_view(pos_vec_view);
    auto id_host = Kokkos::create_mirror_view(id);
    auto h_host = Kokkos::create_mirror_view(h);

    IDX_SET cm_res("CM result", N);

    Kokkos::parallel_for(N, 
      KOKKOS_LAMBDA (const std::size_t p_idx){
        // Iterate over neighbors
        CM.it_over_neighbors(pos_vec_view(p_idx), h(p_idx), 
          [&] (const std::size_t other_idx){
            // Accumulate
            if(unit_sq_periodic_dist<DIM, PERIODIC>(pos_vec_view(p_idx),
                                                    pos_vec_view(other_idx)) < h(p_idx))
              cm_res(p_idx).insert(id(other_idx));
          }
        );
      }
    );

    // Reduce with the naive method
    IDX_SET naive_res = naive_reduction<DIM, PERIODIC>(h_host,
        pos_vec_host, id_host);

    std::size_t err = 0;
    Kokkos::parallel_reduce(N, 
      KOKKOS_LAMBDA (const std::size_t idx, std::size_t& val){
        val += !(naive_res(idx) == cm_res(idx));
      },
      Kokkos::Sum<std::size_t>(err)
    );

    std::cout << k << "\t" << err << std::endl;
  }

}


int main(int argc, char* argv[]){
  Kokkos::initialize(argc, argv);

  constexpr unsigned DIM = 2;
  constexpr const static bool PERIODIC[DIM] = {true, true};

  const unsigned N = 10'000;
  const T min_h = 1e-3;
  const T max_h = 1e-1;

  test_cm<DIM, PERIODIC>(N, min_h, max_h, 8);

  Kokkos::finalize();
}
