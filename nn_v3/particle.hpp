// Dummy class for testing stuff
#pragma once
#include <array>
#include <iostream>

// Characters can store up to 128, plenty of 
// dimensions
template <typename SCALAR, char DIM>
struct Particle{
  std::array<SCALAR, DIM> pos;
  std::array<std::size_t, DIM> chaining_mesh_idx;
  
  Particle() = default;
  
  /* 
    Some variadic template magic to
    initialize the particle coordinates
  */

  // Do nothing
  void set_pos_idx(char dim){};

  template <typename T, typename... COORDS>
  void set_pos_idx(char dim, T s, COORDS... coords){
    pos[dim] = static_cast<SCALAR>(s);
    ++dim;
    if(dim > DIM){
      std::cerr << "Number of given coordinates for the particle does not match the dimension." 
                << std::endl;
      std::abort();
    }
    set_pos_idx(dim, coords...);
  }

  template <typename T, typename... COORDS>
  Particle(T s, COORDS... coords){
    set_pos_idx(0, s, coords...);
  }


  // For debugging
  friend std::ostream& operator<<(std::ostream& out, const Particle& p){
    for(char dim = 0; dim < DIM; ++dim)
      out << p.pos[dim] << '\t';
    out << std::endl;
    return out; 
  }

  // Add more attributes ..
  // ....
};
