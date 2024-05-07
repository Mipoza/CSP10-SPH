#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#define COMPILATION_EVAL(e) (std::integral_constant<decltype(e), e>::value)
constexpr unsigned powi(int base, unsigned exp){return std::pow(base, exp);}
#define NNCells COMPILATION_EVAL(powi(3, DIM))

// IDX_TYPE, unsigned should suffice for our purposes, as exceeding 4,294,967,295 will
// likely not happen
// If we were getting overflows, std::size_t would be the next-best choice
template <typename T, unsigned DIM, bool PERIODIC = false, typename IDX_TYPE = unsigned>
struct ChainingMeshHelper{
  std::array<T, DIM> low;           // lower left of grid
  std::array<T, DIM> L;             // extent in each dir
  std::array<IDX_TYPE, DIM> NN;     // No. cells in each dir
  IDX_TYPE ncells;                  // Total no. cells

  // TODO(Maybe): Maybe figure out if we can do this more
  // memory-efficiently with a hash
  Kokkos::View<unsigned*> cell_size;
  Kokkos::View<IDX_TYPE*> start_idx;
  
  template <class VEC>
  ChainingMeshHelper(const VEC& low_, const VEC& L_, T mesh_width){
    ncells = 1;
    // Read from vec-like elements, mesh_width
    // Undefined behavior if VEC has less than DIM elements
    for(unsigned i = 0; i < DIM; ++i){
      low[i] = low_[i];
      L[i]   = L_[i];
      NN[i]  = static_cast<IDX_TYPE>(L[i]/mesh_width + 0.5);
      ncells *= NN[i];
    }

    // Allocate memory
    Kokkos::View<unsigned*> temp1("Cell sizes", ncells);
    cell_size = temp1;
     
    Kokkos::View<IDX_TYPE*> temp2("Starting indices", ncells);
    start_idx = temp2;
  }

  // Get the grid index as a "tuple" (here array)
  template <class VEC>
  inline std::array<IDX_TYPE, DIM> cell_idx(const VEC& pos) const{
    std::array<IDX_TYPE, DIM> idx;
    for(unsigned i = 0; i < DIM; ++i){
      const T temp = (pos[i] - low[i])/L[i];
      idx[i] = static_cast<IDX_TYPE>(NN[i]*temp);
    }
    return idx;
  }

  // Turn the "tuple" into a key/index
  inline IDX_TYPE idx_to_key(const std::array<IDX_TYPE, DIM>& idx) const{
    IDX_TYPE key_val = 0;
    // Horner-like evaluation of the key
    for(int i = static_cast<int>(DIM) - 1; i >= 0; --i)
      key_val = idx[i] + key_val*NN[i];
    return key_val;
  }

  // ``Partition'': get the starting indices and cell sizes
  // based on particle positions in pos_arr
  template <class VEC_ARRAY>
  void partition(const VEC_ARRAY& pos_arr){
    // Reset everything to 0
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), 
        cell_size, 0);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), 
        start_idx, 0);
    // No. particles
    const IDX_TYPE nparticles = pos_arr.size();
    // Loop and add
    Kokkos::parallel_for("Cell size count loop", nparticles, 
      KOKKOS_LAMBDA (const IDX_TYPE p_idx){
        const IDX_TYPE key = idx_to_key(cell_idx(pos_arr(p_idx)));
        Kokkos::atomic_increment(&cell_size(key));
      }
    );

    // Add up the sizes to get start and end indexes
	  Kokkos::parallel_scan(Kokkos::RangePolicy(0, ncells), 
      KOKKOS_LAMBDA(const IDX_TYPE i, unsigned& localSum, bool isFinal){
	      if(isFinal) start_idx(i) = localSum;
	      localSum += cell_size(i);
	    }
    );
  }

  // Add neighbors (indices of the mesh cells) 
  // of ``current''' to n_list
  template <int DIR>
  void add_neighbor_cells(
      std::vector<IDX_TYPE>& n_list,
      std::array<IDX_TYPE, DIM>& current) const{
    // Degenerate case
    if constexpr (DIR == -1){
      n_list.push_back(idx_to_key(current));
    }
    else{ // Else continue recursion 
      std::array<IDX_TYPE, DIM> temp = current;
      // +, ``right''
      if(current[DIR] + 1 < NN[DIR]){
        ++temp[DIR];
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      else if constexpr(PERIODIC){
        temp[DIR] = 0;
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      // -, ``left''
      temp = current;
      if(current[DIR] > 0){
        --temp[DIR];
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      else if constexpr(PERIODIC){
        temp[DIR] = NN[DIR] - 1;
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      // 0, add self/center
      add_neighbor_cells<DIR - 1>(n_list, current);
    }
  }

  // Get a list of indices of cells neighboring ``current'''
  inline std::vector<IDX_TYPE> 
  get_cell_neighbor_idx(std::array<IDX_TYPE, DIM>& current) const{
    std::vector<IDX_TYPE> res;
    // Most of the time, a good estimate
    // (boundaries might have less neighbors, depends on if it's periodic)
    res.reserve(NNCells);
    add_neighbor_cells<DIM - 1>(res, current); 
    return res;
  }
  
  // Re-orders the target view so we can access 
  // the neighbors efficiently
  template <typename D_TYPE, typename VEC_TYPE>
  void sort(const Kokkos::View<VEC_TYPE*>& pos, Kokkos::View<D_TYPE*>& target){
    // Current index in each cell
    Kokkos::View<IDX_TYPE*> current_idx("Current Index", ncells);
    Kokkos::deep_copy(current_idx, start_idx);
    // Create temporary
    Kokkos::View<D_TYPE*> temp_target("Temporary", target.size());
    // Copy in the right order
    Kokkos::parallel_for("Reorder-Loop", target.size(),
      KOKKOS_LAMBDA (const IDX_TYPE i){
        const IDX_TYPE key = idx_to_key(cell_idx(pos(i)));
        const IDX_TYPE new_idx = Kokkos::atomic_fetch_add(&current_idx(key), 1);
        temp_target(new_idx) = target(i);
      }
    );
    // Copy it back
    Kokkos::deep_copy(target, temp_target);
  }
};
