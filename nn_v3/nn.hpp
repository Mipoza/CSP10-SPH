#pragma once
#include <unordered_map>

template <typename SCALAR, char DIM, class PARTICLE>
struct ChainingMesh{
  std::array<SCALAR, DIM> low;     // lower left of grid
  std::array<SCALAR, DIM> L;       // extent in each dir
  std::array<std::size_t, DIM> NN; // No cells in each dir
  std::unordered_multimap<std::size_t, PARTICLE*> particle_hash;
  // Some small number to ensure that the particles at the 
  // boundaries of the bounding box do not lie exactly on the boundary
  // Could be that this is unnessecary, but it does not hurt
  // TODO: TEST IF THIS IS ACTUALLY NEEDED
  SCALAR eps = 1e-6;

  // Constructors
  ChainingMesh() = default;
  template <class LIST_LIKE>
  ChainingMesh(LIST_LIKE& particle_list, const SCALAR mesh_width){
    update(particle_list, mesh_width);
  }
  ChainingMesh(const SCALAR _eps): eps(_eps) {}

  // Get the grid index as a "tuple" (here array)
  inline std::array<std::size_t, DIM> cell_idx(const PARTICLE& p) const{
    std::array<std::size_t, DIM> idx;
    for(char i = 0; i < DIM; ++i){
      const SCALAR temp = (p.pos[i] - low[i])/L[i];
      idx[i] = static_cast<std::size_t>(NN[i]*temp);
    }
    return idx;
  }

  // Turn the "tuple" into a key
  inline std::size_t idx_to_key(const std::array<std::size_t, DIM>& idx) const{
    std::size_t key_val = 0;
    // Horner-like evaluation of the key
    for(char i = DIM - 1; i >= 0; --i)
      key_val = idx[i] + key_val*NN[i];
    return key_val;
  }

  // Add a list/array of particles
  template <class LIST_LIKE>
  inline void add_particle_list(LIST_LIKE& particle_list){
    for(auto p_it = particle_list.begin(); p_it != particle_list.end(); ++p_it){
      // Set the local variable of the particle to the cell idx
      p_it->chaining_mesh_idx = cell_idx(*p_it);
      // Add to hash
      particle_hash.insert({idx_to_key(p_it->chaining_mesh_idx), &(*p_it)});
    }
  }

  // Update to accomodate new list of particles
  template <class LIST_LIKE>
  void update(LIST_LIKE& particle_list, const SCALAR mesh_width){
    std::array<SCALAR, DIM> max;
    // Iterate over dimensions
    for(char i = 0; i < DIM; ++i){
      // Initialize with new values
      low[i] = particle_list[0].pos[i];
      max[i] = particle_list[0].pos[i];
      // Iterate over particles to find the boundaries of the box
      for(auto p_it = particle_list.begin(); p_it != particle_list.end(); ++p_it){
        low[i] = std::min(p_it->pos[i], low[i]);
        max[i] = std::max(p_it->pos[i], max[i]);
      }
      // Perturn so that the particles don't align with the
      // boundaries of the box
      low[i] -= eps;
      max[i] += eps;
      // Set extent of box in a direction
      L[i] = max[i] - low[i];
      NN[i] = static_cast<std::size_t>(L[i]/mesh_width + 0.5);
      // + 0.5 because we want this to be rounded up
    }
    // Clear hash
    particle_hash.clear();
    // Insert new particles
    add_particle_list(particle_list);
  }

  // Retrieve list of particles in a cell of the chaining mesh
  // RETURN: pair of (begin, end), where begin nad end are iterators,
  //         Dereferencing each gives a \emph{pair}
  //         of key (std::size_t) and value (PARTICLE*)
  auto at(const std::array<std::size_t, DIM>& idx){
    const std::size_t bucket_idx = particle_hash.bucket(idx_to_key(idx));
    return std::make_pair(particle_hash.begin(bucket_idx), particle_hash.end(bucket_idx));
  }
  
};

