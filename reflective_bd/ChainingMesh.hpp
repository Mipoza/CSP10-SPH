#pragma once
#include <Kokkos_Macros.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#define MIN(x, y) (x < y ? x : y)

// IDX_TYPE, unsigned should suffice for our purposes, as exceeding 4'294'967'295 will
// likely not happen
// If we were getting overflows, std::size_t would be the next-best choice
// shorts are unsuitable, as they cannot represent a large enough range
template <typename T, unsigned DIM, 
          const bool PERIODIC[DIM],
          typename IDX_TYPE = unsigned>
struct ChainingMesh{
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
  ChainingMesh(const VEC& low_, const VEC& L_,
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
  ChainingMesh(const VEC& low_, const VEC& L_,
               const T& mesh_width){
    std::array<T, DIM> mesh_width_;
    for(unsigned d = 0; d < DIM; ++d) mesh_width_[d] = mesh_width;
    *this = ChainingMesh(low_, L_, mesh_width_);
  }

  // Get the grid index as a "tuple" (here array)
  template <class VEC>
  KOKKOS_INLINE_FUNCTION 
  std::array<IDX_TYPE, DIM> cell_idx(const VEC& pos) const{
    std::array<IDX_TYPE, DIM> idx;
    // Multiply with 1 - eps
    // It does not like values at the boundary
    #pragma GCC unroll(DIM)
    for(unsigned i = 0; i < DIM; ++i)
      idx[i] = static_cast<IDX_TYPE>((pos[i] - low[i])*
          (static_cast<T>(1) - 16*std::numeric_limits<T>::epsilon())
                                      /mesh_widths[i]);
    return std::move(idx);
  }

  // Turn the "tuple" into a key/index
  template <int DIR = 0>
  KOKKOS_INLINE_FUNCTION IDX_TYPE 
  idx_to_key(const std::array<IDX_TYPE, DIM>& idx) const {
    if constexpr(DIR + 1 == DIM){
      return idx[DIR];
    }
    else {
      return idx[DIR] + NN[DIR] * idx_to_key<DIR + 1>(std::move(idx)); 
    }
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

  template <typename VEC, bool ABSOLUTE_POS = false, int DIR = 0>
  void KOKKOS_INLINE_FUNCTION
  __attribute__((always_inline))
  reflect_vec(
      VEC& v, 
      const std::array<std::tuple<bool, bool>, DIM>& reflect) const {
    if constexpr(!PERIODIC[DIR]){ // If it is periodic, this is a no-op
      if constexpr(!ABSOLUTE_POS){ // Reflect e.g. velocity, just flip a sign
        if(std::get<0>(reflect[DIR]) ||
           std::get<1>(reflect[DIR])){
          v[DIR] *= -1;
         }
      }
      else { // Reflect e.g. the absolute position
        if(std::get<0>(reflect[DIR])){
          if constexpr(ABSOLUTE_POS){
            v[DIR] = 2*low[DIR] - v[DIR];
          }
        } else if(std::get<1>(reflect[DIR])){ // Assumption: radius < L[DIR]
          if constexpr(ABSOLUTE_POS){
            v[DIR] = 2*(low[DIR] + L[DIR]) - v[DIR];
          }
        }
      }
    }
    //if(std::get<0>(reflect[DIR]) ||
    //   std::get<1>(reflect[DIR])){
    //  std::cerr << "WAT";
    //}
    if constexpr(DIR + 1 < DIM){
      reflect_vec<VEC, ABSOLUTE_POS, DIR + 1>(v, reflect);
    }
  }

  // Iterate over neighbors (indices of the mesh cells) 
  // of ``current'''
  template <int DIR, typename FUNCTOR,
            typename ARR_VEC,
            typename... F_ARG_ARR>
  KOKKOS_INLINE_FUNCTION constexpr
  void generate_loops(
      std::array<IDX_TYPE, DIM>& current,
      const std::array<IDX_TYPE, DIM>& r,
      FUNCTOR F, 
      std::array<std::tuple<bool, bool>, DIM> reflect,
      const ARR_VEC& pos_vec, 
      F_ARG_ARR&... fargs) const {
    // Degenerate case
    if constexpr (DIR == -1){
      const IDX_TYPE key = idx_to_key(current);
      const IDX_TYPE start_idx_base = start_idx(key);
      for(IDX_TYPE other_idx_ = 0;
          other_idx_ < cell_size(key);
          ++other_idx_){
        // Index of neighbordouble
        const IDX_TYPE other_idx = start_idx_base + other_idx_;

        // Get at index
        auto get_val = [&](auto& arr) {
          auto v = arr(other_idx);
          // Tacitly assume that the rest only needs to be reflected by sign
          reflect_vec<decltype(v), false, 0>(v, reflect);
          return v;
        };

        auto other_pos = pos_vec(other_idx);
        reflect_vec<decltype(other_pos), true, 0>(other_pos, reflect);

        F(other_idx, std::move(other_pos), std::move(get_val(fargs))...);
      }
    }
    else{ // Else continue recursion 
      IDX_TYPE i;
      std::array<IDX_TYPE, DIM> temp = current;
      // +, ``right''
      for(i = 0; (temp[DIR] + 1 < NN[DIR]) && (i < r[DIR]); ++i){
        ++temp[DIR];
        generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(temp,
            r, F, reflect, pos_vec, fargs...);
      }
      if(i < r[DIR]){
        if constexpr(PERIODIC[DIR]){
          temp[DIR] = 0;
          for(; (temp[DIR] < current[DIR]) && (i < r[DIR]); ++i){
            generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(temp,
                r, F, reflect, pos_vec, fargs...);
            ++temp[DIR];
          }
        } else {
          temp[DIR] = NN[DIR] - 1;
          std::get<1>(reflect[DIR]) = true;
          for(; (temp[DIR] > current[DIR]) && (i < r[DIR]); ++i){
            generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(temp,
                r, F, reflect, pos_vec, fargs...);
            --temp[DIR];
          }
          std::get<1>(reflect[DIR]) = false;
        }
      } 

      // -, ``left''
      temp = current;
      for(i = 0; (temp[DIR] > 0) && (i < r[DIR]); ++i){
        --temp[DIR];
        generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(temp,
            r, F, reflect, pos_vec, fargs...);
      }
      if(i < r[DIR]){
        if constexpr(PERIODIC[DIR]){
          temp[DIR] = NN[DIR] - 1;
          for(; (temp[DIR] > current[DIR]) && (i < r[DIR]); ++i){
            generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(temp,
                r, F, reflect, pos_vec, fargs...);
            --temp[DIR];
          }
        } else {
          temp[DIR] = 0;
          std::get<0>(reflect[DIR]) = true;
          for(; (temp[DIR] < current[DIR]) && (i < r[DIR]); ++i){
            generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(temp,
                r, F, reflect, pos_vec, fargs...);
            ++temp[DIR];
          }
          std::get<0>(reflect[DIR]) = false;
        }
      }

      // 0, add self/center
      generate_loops<DIR - 1, FUNCTOR, ARR_VEC, F_ARG_ARR...>(current,
          r, F, reflect, pos_vec, fargs...);
    }
  }

  // Iterate over cells neighboring ``current''
  template <typename FUNCTOR, typename VEC,
            typename ARR_VEC, typename... F_ARG_ARR>
  KOKKOS_INLINE_FUNCTION
  void it_over_neighbors(const VEC& pos, 
                         const T& R,
                         FUNCTOR F,
                         const ARR_VEC& pos_vec,
                         const F_ARG_ARR&... fargs) const {

    std::array<IDX_TYPE, DIM> current = cell_idx(pos);
    std::array<IDX_TYPE, DIM> r;
    std::array<std::tuple<bool, bool>, DIM> reflect;
    // Compute the no. cells to iterate over
    for(unsigned d = 0; d < DIM; ++d) {
      r[d] = std::ceil(R/mesh_widths[d]);
      std::get<0>(reflect[d]) = false;
      std::get<1>(reflect[d]) = false;
    }

    // Generate the loops at compile-time
    generate_loops<DIM - 1, FUNCTOR>(current, r, F, reflect, pos_vec,
                                     fargs...); 
  }
  
  /* Some helper functions */

  template <typename TYPE, typename... TYPE_PACK>
  constexpr inline __attribute__((always_inline))
  std::tuple<TYPE, TYPE_PACK...> 
  create_temps(TYPE& target, TYPE_PACK&... rest){
    // TYPE == Kokkos::View<SMTH?>
    TYPE tmp("tmp", target.size());
    if constexpr(sizeof...(rest) > 0)
      return std::tuple_cat(std::tuple<TYPE>(tmp),
                            create_temps(rest...));
    else
      return std::tuple<TYPE>(tmp);
  }

  template <unsigned I, typename... TYPE_PACK>
  constexpr inline void __attribute__((always_inline))
  reassign(const IDX_TYPE& new_idx, const IDX_TYPE& i,
           const std::tuple<TYPE_PACK...>& dest, 
           const std::tuple<TYPE_PACK...>& src){
    if constexpr(I < sizeof...(TYPE_PACK)){
      std::get<I>(dest)(new_idx) = std::get<I>(src)(i);
      reassign<I + 1, TYPE_PACK...>(new_idx, i, dest, src);
    } else return;
  }

  template <unsigned I, typename... TYPE_PACK>
  constexpr inline void __attribute__((always_inline))
  deepcopy(std::tuple<TYPE_PACK...>& dest, 
           std::tuple<TYPE_PACK...>& src){
    if constexpr(I < sizeof...(TYPE_PACK)){
      Kokkos::deep_copy(std::get<I>(dest),
                        std::get<I>(src));
      deepcopy<I + 1, TYPE_PACK...>(dest, src);
    } else return;
  }

  /* The actual sorting */

  // Re-orders the target view so we can access 
  // the neighbors efficiently
  template <typename VEC_TYPE, typename ...TARGET_T>
  void sort(const Kokkos::View<VEC_TYPE>& pos, TARGET_T&... targets_){
    static_assert(sizeof...(targets_) > 0);
    // Current index in each cell
    Kokkos::View<IDX_TYPE*> current_idx("Current Index", ncells);
    Kokkos::deep_copy(current_idx, start_idx);
    // Create temporaries
    std::tuple<TARGET_T...> temps = create_temps(targets_...);
    std::tuple<TARGET_T...> targets(targets_...);
    const std::size_t N_particles = std::get<0>(targets).size();
    // Copy in the right order
    Kokkos::parallel_for("Reorder-Loop", N_particles,
      KOKKOS_LAMBDA (const IDX_TYPE i){
        const IDX_TYPE key = idx_to_key(cell_idx(pos(i)));
        const IDX_TYPE new_idx = Kokkos::atomic_fetch_add(&current_idx(key), 1);
        reassign<0, TARGET_T...>(new_idx, i, temps, targets);
      }
    );
    // Copy it back
    deepcopy<0, TARGET_T...>(targets, temps);
  }
};
