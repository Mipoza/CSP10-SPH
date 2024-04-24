#pragma once
#include <array>
#include <iostream>
#include <vector>
#include <forward_list>
#include <cmath>

#define COMPILATION_EVAL(e) (std::integral_constant<decltype(e), e>::value)
constexpr unsigned powi(int base, unsigned exp){return std::pow(base, exp);}
#define NNCells COMPILATION_EVAL(powi(3, DIM))

template <unsigned DIM>
struct SizeListCollection{
  std::vector<std::forward_list<std::size_t>*> ll_ptrs;
  
  SizeListCollection(){
    // Most of the time, a good estimate
    ll_ptrs.reserve(NNCells);
  }

  void add_list(std::forward_list<std::size_t>* ptr){
    // Add to list
    ll_ptrs.push_back(ptr);
  }

  // Reset the list, does not change the capacity
  // of ll_ptrs, only clears its contents
  void clear(){
    ll_ptrs.clear();
  }

  struct Iterator{
    unsigned idx_ = 0;
    std::forward_list<std::size_t>::iterator it_;
    SizeListCollection<DIM>* ptr_;

    Iterator(SizeListCollection<DIM>* ptr,
        std::forward_list<std::size_t>::iterator it, unsigned idx): 
      ptr_(ptr), it_(it), idx_(idx) {
      }

    std::size_t& operator*() {
      return *it_;
    }

    Iterator& operator++(){
      ++it_;
      // Jump to next list?
      if(it_ == (ptr_->ll_ptrs[idx_]->end()) && (idx_ + 1 < ptr_->ll_ptrs.size())){
        ++idx_;
        it_ = ptr_->ll_ptrs[idx_]->begin();
      }
      return *this;
    }

    friend bool operator!=(const Iterator& a, const Iterator& b){
      return (a.it_ != b.it_);
    }
  };

  // Basic begin, end functions

  Iterator begin(){
    return Iterator(this, ll_ptrs.front()->begin(), 0);
  }
  Iterator end(){
    return Iterator(this, ll_ptrs.back()->end(), ll_ptrs.size() - 1);
  }
};


template <typename T, unsigned DIM, bool PERIODIC = false>
struct ChainingMeshHelper{
  std::array<T, DIM> low;           // lower left of grid
  std::array<T, DIM> L;             // extent in each dir
  std::array<std::size_t, DIM> NN;  // No cells in each dir
  // TODO: Maybe figure out if we can do this more
  // memory-efficiently with a hash

  // "Buckets"
  std::vector<std::forward_list<std::size_t>> cell_lists;
  // Collection of buckets as neighbors
  std::vector<SizeListCollection<DIM>> neighbor_lists;
  
  ChainingMeshHelper() = default;

  template <class VEC>
  ChainingMeshHelper(const VEC& low_, const VEC& L_, T mesh_width){
    std::size_t ncells = 1;
    // Read from vec-like elements, mesh_width
    // Undefined behavior if VEC has less than DIM elements
    for(unsigned i = 0; i < DIM; ++i){
      low[i] = low_[i];
      L[i]   = L_[i];
      NN[i]  = static_cast<std::size_t>(L[i]/mesh_width + 0.5);
      ncells *= NN[i];
    }
    // Can be quite large in some cases
    try{
      // Reserve space for all the buckets
      cell_lists.reserve(ncells);
      neighbor_lists.reserve(ncells);
      // We want to access the vector at indices, so resize to ``size'''
      cell_lists.resize(ncells);
      neighbor_lists.resize(ncells);
      // std::cout << "Allocated: " << ncells << std::endl;
    } catch(std::bad_alloc){
        std::cerr << "Alloc failed, tried to allocate "
                  << ncells << " cells" << std::endl;
        std::abort();
    }
    create_neighbor_lists();
  }

  // Get the grid index as a "tuple" (here array)
  template <class VEC>
  inline std::array<std::size_t, DIM> cell_idx(const VEC& pos) const{
    std::array<std::size_t, DIM> idx;
    for(unsigned i = 0; i < DIM; ++i){
      const T temp = (pos[i] - low[i])/L[i];
      idx[i] = static_cast<std::size_t>(NN[i]*temp);
    }
    return idx;
  }

  // Turn the "tuple" into a key
  inline std::size_t idx_to_key(const std::array<std::size_t, DIM>& idx) const{
    std::size_t key_val = 0;
    // Horner-like evaluation of the key
    for(int i = static_cast<int>(DIM) - 1; i >= 0; --i)
      key_val = idx[i] + key_val*NN[i];
    return key_val;
  }

  // Add a particle to the list
  template <class VEC>
  inline void add_particle(const VEC& pos, const std::size_t& p_idx){
    // "Supercell"-index
    auto c_idx = cell_idx(pos);
    // Add p_idx to the list in that cell
    cell_lists[idx_to_key(c_idx)].push_front(p_idx);
  }


  template <int DIR>
  void add_neighbor_cells(
      std::vector<std::array<std::size_t, DIM>>& n_list,
      std::array<std::size_t, DIM>& current){
    // Degenerate case
    if constexpr (DIR == -1){
      n_list.push_back(current);
    }
    else{
      std::array<std::size_t, DIM> temp = current;
      // +
      if(current[DIR] + 1 < NN[DIR]){
        ++temp[DIR];
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      else if constexpr(PERIODIC){
        temp[DIR] = 0;
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      // -
      temp = current;
      if(current[DIR] > 0){
        --temp[DIR];
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      else if constexpr(PERIODIC){
        temp[DIR] = NN[DIR] - 1;
        add_neighbor_cells<DIR - 1>(n_list, temp);
      }
      // 0, add self
      add_neighbor_cells<DIR - 1>(n_list, current);
    }
  }

  std::vector<std::array<std::size_t, DIM>> 
  get_cell_neighbor_idx(const std::array<std::size_t, DIM>& current_){
    std::vector<std::array<std::size_t, DIM>> res;
    // Most of the time, a good estimate
    // (boundaries might have less neighbors, depends on if it's periodic)
    res.reserve(NNCells);
    std::array<std::size_t, DIM> current = current_;
    add_neighbor_cells<DIM - 1>(res, current);  
    return res;
  }
  
  template <unsigned DIR>
  inline void add_neighbours(std::array<std::size_t, DIM>& c_idx){
    if constexpr (DIR < DIM){
      // Iterate over current dim
      for(std::size_t l = 0; l < NN[DIR]; ++l){
        c_idx[DIR] = l;
        // Next dimension
        add_neighbours<DIR + 1>(c_idx);
      }
    } else if constexpr (DIR == DIM){
        const std::size_t this_key = idx_to_key(c_idx);
        // Add all neighboring cells
        for(auto idx : get_cell_neighbor_idx(c_idx))
          neighbor_lists[this_key].add_list(&cell_lists[idx_to_key(idx)]);
    } // Else no-op
    else return;
  }

  // Merge cell lists into (potential) neighbor lists
  void create_neighbor_lists(){
    std::array<std::size_t, DIM> c_idx;
    // Set to zero
    for(unsigned d = 0; d < DIM; ++d)
      c_idx[d] = 0;
    // Iterate, DIM-independence requires
    // using some templated functions
    add_neighbours<0>(c_idx);
  }

  // Clear the cell lists
  // (e.g. in perparation of adding new particles after an update)
  // The neighbor list is cleared when create_neighbor_lists is called,
  // no need to do it here
  void clear(){
    for(auto& ll : cell_lists)
      ll.clear();
    for(auto& ll : neighbor_lists)
      ll.clear();
  }

  // Return an object giving the neighbors of something at
  // position pos
  template <class VEC>
  const SizeListCollection<DIM>& neighbors(const VEC& pos){
    // "Hash"
    const std::size_t idx = idx_to_key(cell_idx(pos));
    // std::cout << "IDX: "<< idx << std::endl;
    // Return (const reference)
    return neighbor_lists[idx];
  }
};
