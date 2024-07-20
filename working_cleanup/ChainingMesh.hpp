#pragma once
#include <Kokkos_Macros.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#define MIN(x, y) (x < y ? x : y)

#define COMPILATION_EVAL(e) (std::integral_constant<decltype(e), e>::value)
constexpr unsigned powi(int base, unsigned exp){return std::pow(base, exp);}
#define NNCells COMPILATION_EVAL(powi(3, DIM))

// IDX_TYPE, unsigned should suffice for our purposes, as exceeding 4'294'967'295 will
// likely not happen
// If we were getting overflows, std::size_t would be the next-best choice
// shorts are unsuitable, as they cannot represent a large enough range
template <typename T, unsigned DIM, const bool PERIODIC[DIM], typename IDX_TYPE = unsigned>
struct ChainingMeshHelper{
  std::array<T, DIM> low;           // lower left of grid
  std::array<T, DIM> L;             // extent in each dir
  std::array<T, DIM> mesh_widths;
  std::array<IDX_TYPE, DIM> NN;     // No. cells in each dir
  IDX_TYPE ncells;                  // Total no. cells

  // TODO(Maybe): Maybe figure out if we can do this more
  // memory-efficiently with a hash
  Kokkos::View<unsigned*> cell_size;
  Kokkos::View<IDX_TYPE*> start_idx;
  

  template <class VEC>
  ChainingMeshHelper(const VEC& low_, const VEC& L_,
                     std::array<T, DIM> mesh_width){
    ncells = 1;
    // Read from vec-like elements, mesh_width
    // Undefined behavior if VEC has less than DIM elements
    for(unsigned i = 0; i < DIM; ++i){
      low[i]  = low_[i];
      L[i]    = L_[i];
      // Round to next best convenient mesh width
      // Necessary, because otherwise the mesh dpesn't fit into the domain
      const T mesh_width_i = L_[i]/std::ceil(L_[i]/mesh_width[i]);
      mesh_widths[i] = mesh_width_i;
      NN[i]   = std::ceil(L[i]/mesh_width_i);
      ncells *= NN[i];
    }

    // Allocate memory
    Kokkos::View<unsigned*> temp1("Cell sizes", ncells);
    cell_size = temp1;
     
    Kokkos::View<IDX_TYPE*> temp2("Starting indices", ncells);
    start_idx = temp2;
  }

  // Overload for "uniform" mesh width
  template <class VEC>
  ChainingMeshHelper(const VEC& low_, const VEC& L_,
                     const T& mesh_width){
    std::array<T, DIM> mesh_width_;
    for(unsigned d = 0; d < DIM; ++d) mesh_width_[d] = mesh_width;
    *this = ChainingMeshHelper(low_, L_, mesh_width_);
  }

  // Get the grid index as a "tuple" (here array)
  template <class VEC>
  KOKKOS_INLINE_FUNCTION 
  std::array<IDX_TYPE, DIM> cell_idx(const VEC& pos) const{
    std::array<IDX_TYPE, DIM> idx;
    for(unsigned i = 0; i < DIM; ++i)
      idx[i] = static_cast<IDX_TYPE>((pos[i] - low[i])/mesh_widths[i]);
    return idx;
  }

  // Turn the "tuple" into a key/index
  KOKKOS_INLINE_FUNCTION IDX_TYPE 
  idx_to_key(const std::array<IDX_TYPE, DIM>& idx) const{
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
    Kokkos::deep_copy(cell_size, 0);
    Kokkos::deep_copy(start_idx, 0);
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

  // Iterate over neighbors (indices of the mesh cells) 
  // of ``current'''
  template <int DIR, typename FUNCTOR>
  KOKKOS_INLINE_FUNCTION
  void generate_loops(
      std::array<IDX_TYPE, DIM>& current,
      const std::array<IDX_TYPE, DIM>& r,
      FUNCTOR F) const {
    // Degenerate case
    if constexpr (DIR == -1){
      const IDX_TYPE key = idx_to_key(current);
      const IDX_TYPE start_idx_base = start_idx(key);
      for(IDX_TYPE other_idx_ = 0;
          other_idx_ < cell_size(key);
          ++other_idx_){
        // Index of neighbor
        const IDX_TYPE other_idx = start_idx_base + other_idx_;
        F(other_idx);
      }
    }
    else{ // Else continue recursion 
      IDX_TYPE i;
      std::array<IDX_TYPE, DIM> temp = current;
      // +, ``right''
      for(i = 0; (temp[DIR] + 1 < NN[DIR]) && (i < r[DIR]); ++i){
        ++temp[DIR];
        generate_loops<DIR - 1, FUNCTOR>(temp, r, F);
      }
      if constexpr(PERIODIC[DIR]){
        if(i < r[DIR]){
          temp[DIR] = 0;
          // Assume ENOUGH cells (otherwise this algorithm
          // might never terminate; won't make sense that case) 
          for(; (temp[DIR] < current[DIR]) && (i < r[DIR]); ++i){
            generate_loops<DIR - 1, FUNCTOR>(temp, r, F);
            ++temp[DIR];
          }
        }
      }
      // -, ``left''
      temp = current;
      for(i = 0; (temp[DIR] > 0) && (i < r[DIR]); ++i){
        --temp[DIR];
        generate_loops<DIR - 1, decltype(F)>(temp, r, F);
      }
      if constexpr(PERIODIC[DIR]){
        if(i < r[DIR]){
          temp[DIR] = NN[DIR] - 1;
          for(; (temp[DIR] > current[DIR]) && (i < r[DIR]); ++i){
            generate_loops<DIR - 1, FUNCTOR>(temp, r, F);
            --temp[DIR];
          }
        }
      }
      // 0, add self/center
      generate_loops<DIR - 1, FUNCTOR>(current, r, F);
    }
  }

  // Iterate over cells neighboring ``current''
  template <typename FUNCTOR, typename VEC>
  KOKKOS_INLINE_FUNCTION
  void it_over_neighbors(const VEC& pos, 
                         const T& R,
                         FUNCTOR F) const {

    std::array<IDX_TYPE, DIM> current = cell_idx(pos);
    std::array<IDX_TYPE, DIM> r;
    // Compute the no. cells to iterate over
    for(unsigned d = 0; d < DIM; ++d) r[d] = std::ceil(R/mesh_widths[d]);

    // Generate the loops at compile-time
    generate_loops<DIM - 1, FUNCTOR>(current, r, F); 
  }
  
  // Re-orders the target view so we can access 
  // the neighbors efficiently
  template <typename VEC_TYPE, typename D_TYPE>
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
