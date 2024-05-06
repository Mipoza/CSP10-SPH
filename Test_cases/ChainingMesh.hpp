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
// likely suffice
// I we were getting overflows, std::size_t would be the next best choice
template <typename T, unsigned DIM, bool PERIODIC = false, typename IDX_TYPE = unsigned>
struct ChainingMeshHelper{
  std::array<T, DIM> low;           // lower left of grid
  std::array<T, DIM> L;             // extent in each dir
  std::array<IDX_TYPE, DIM> NN;  // No cells in each dir
  IDX_TYPE ncells;

  // TODO: Maybe figure out if we can do this more
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

  // Turn the "tuple" into a key
  inline IDX_TYPE idx_to_key(const std::array<IDX_TYPE, DIM>& idx) const{
    IDX_TYPE key_val = 0;
    // Horner-like evaluation of the key
    for(int i = static_cast<int>(DIM) - 1; i >= 0; --i)
      key_val = idx[i] + key_val*NN[i];
    return key_val;
  }

  template <class VEC_ARRAY>
  void partition(const VEC_ARRAY& pos_arr){
    // Reset everything to 0
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), 
        cell_size, 0);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), 
        start_idx, 0);
    // No particles
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

  inline std::vector<IDX_TYPE> 
  get_cell_neighbor_idx(std::array<IDX_TYPE, DIM>& current) const{
    std::vector<IDX_TYPE> res;
    // Most of the time, a good estimate
    // (boundaries might have less neighbors, depends on if it's periodic)
    res.reserve(NNCells);
    // std::array<IDX_TYPE, DIM> current = current_;
    add_neighbor_cells<DIM - 1>(res, current); 
    return res;
  }
  

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
        // set_equal(i, new_idx, target, temp_target);
        temp_target(new_idx) = target(i);
      }
    );
    // Copy it back
    Kokkos::deep_copy(target, temp_target);
  }
 
  // Dummy iteration
  template <typename F, class VEC>
  void apply_F(F func, VEC& P_VEC){
    const IDX_TYPE nparticles = P_VEC.size();
    // Loop over particles
    Kokkos::parallel_for(nparticles, 
      KOKKOS_LAMBDA (const IDX_TYPE p_idx){
        std::array<IDX_TYPE, 2> key = cell_idx(P_VEC(p_idx));
        const auto my_neighbor_cells = get_cell_neighbor_idx(key);
        /*
          Here goes some code (initialize variables used later etc.)
        */
        // Loop over neighbor cells
        for(IDX_TYPE n_cell_idx = 0;
            n_cell_idx < my_neighbor_cells.size();
            ++n_cell_idx){
          // Loop over particles in neighbor cells
          const IDX_TYPE start_idx_base = start_idx(my_neighbor_cells[n_cell_idx]);
          for(IDX_TYPE other_idx_ = 0; 
              other_idx_ < cell_size(my_neighbor_cells[n_cell_idx]);
              ++other_idx_){
            const IDX_TYPE other_idx = start_idx_base + other_idx_;
            /*
              Here goes some code (do the actual compute)
              DUMMY, just an example code
            */
            func(P_VEC(p_idx), P_VEC(other_idx));
          }
        }
      }
    );
  }
};
