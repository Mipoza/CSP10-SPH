#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <cmath>

#include "std_array_overloads.hpp"
#include "ChainingMesh.hpp"
#include "kernels.hpp"
#include "secant.hpp"

#define MAX(x, y) (x > y ? x : y)

#define eps static_cast<T>(1e-4)

template <typename T, unsigned int DIM>
struct viscosity_factor{
  T alpha, beta;

  viscosity_factor() = default;

  viscosity_factor(T alpha_): alpha(alpha_), beta(2*alpha_){}

    inline T operator()(T c, T density, Vec<T, DIM> dis, Vec<T, DIM> vel, T h) const {
        T rij = Kokkos::sqrt(dis.dot(dis));
        T dot_product = dis.dot(vel);
        T mu = h*dot_product/(std::pow(rij,2) + 0.01*std::pow(h,2));

        if (dot_product < 0) {return (-(alpha * c * mu) + (beta * mu * mu)) / (density + eps);} 
        else {return 0;}
    }

};

template<typename T, unsigned int DIM, 
         const bool PERIODIC[DIM],
         bool viscous = false,
         class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHManager {
    // Parameters
    T dt, Adiabatic_index, h, mass_target = -1;
    Vec<T, DIM> L_, low_;
    // The kernel itself
    KERNEL K;
    // Helper for nearest neighbors
    ChainingMeshHelper<T, DIM, PERIODIC> CMHelper;

    // Physical quantities
    Kokkos::View<T*> mass, density, pressure, entropy, d_entropy, smoothing_kernel_sizes, gradh;
    Kokkos::View<Vec<T, DIM>*> position, velocity, accel;

    SPHManager(Vec<T, DIM>& low, Vec<T, DIM>& extent, 
               T dt_, T h_, 
               T Adiabatic_index_) : 
      dt(dt_), h(h_), 
      CMHelper(low, extent, h_), 
      Adiabatic_index(Adiabatic_index_), 
      L_(extent), low_(low) {}

    // Allocate space for the particles
    void create_particles(std::size_t N){
      // Scalar quantities
      Kokkos::View<T*> mass_("Mass", N);
      mass = mass_;
      Kokkos::View<T*> gradh_("gradh", N);
      gradh = gradh_;
      Kokkos::View<T*> density_("Density", N);
      density = density_;
      Kokkos::View<T*> pressure_("Pressure", N);
      pressure = pressure_;
      Kokkos::View<T*> entropy_("Entropy", N);
      entropy = entropy_;
      Kokkos::View<T*> d_entropy_("DEntropy", N);
      d_entropy = d_entropy_;
      Kokkos::View<T*> smoothing_kernel_sizes_("Smoothing Kernel Sizes", N);
      smoothing_kernel_sizes = smoothing_kernel_sizes_;
      // Set all smoothing kernel sizes to h
      Kokkos::deep_copy(smoothing_kernel_sizes, h);

      // Vector quantities
      Kokkos::View<Vec<T, DIM>*> position_("Position", N);
      position = position_;
      Kokkos::View<Vec<T, DIM>*> velocity_("Velocity", N);
      velocity = velocity_;
      Kokkos::View<Vec<T, DIM>*> accel_("Acceleration", N);
      accel = accel_;
    }

    // Update the ChainingMeshHelper
    void updateNeighbors(){
      CMHelper.partition(position);
      // Sort for data locality
      CMHelper.sort(position, mass);
      CMHelper.sort(position, gradh);
      CMHelper.sort(position, smoothing_kernel_sizes);
      CMHelper.sort(position, density);
      CMHelper.sort(position, pressure);
      CMHelper.sort(position, entropy);
      CMHelper.sort(position, d_entropy);
      CMHelper.sort(position, velocity);
      CMHelper.sort(position, accel);
      // AT THE END, sort position
      CMHelper.sort(position, position);
    }

    // Set initial conditions
    void init(std::vector<Vec<T, DIM>> R_part, 
              std::vector<Vec<T, DIM>> v_part,
              std::vector<T> m_part,
              std::vector<T> entropy_part,
              T mass_target_ = -1) {
      int N_new = R_part.size();
      create_particles(N_new);

      auto R_host = Kokkos::create_mirror_view(position);
      auto v_host = Kokkos::create_mirror_view(velocity);
      auto m_host = Kokkos::create_mirror_view(mass);
      auto entropy_host = Kokkos::create_mirror_view(entropy);

      T avg_m = 0;
      for (unsigned int i = 0; i < N_new; ++i) {
          R_host(i) = R_part[i];
          v_host(i) = v_part[i];
          m_host(i) = m_part[i];
          entropy_host(i) = entropy_part[i];
          avg_m += m_part[i]/N_new;
      }

      Kokkos::deep_copy(position, R_host);
      Kokkos::deep_copy(velocity, v_host);
      Kokkos::deep_copy(mass, m_host);
      Kokkos::deep_copy(entropy, entropy_host);
      // Set it to something
      if(mass_target_ == -1)
        mass_target = 12*avg_m;
    }
    
    KOKKOS_INLINE_FUNCTION
    void compute_density(const std::size_t p_idx, const T h_loc){
      auto key = CMHelper.cell_idx(position(p_idx));
      // Round to get a radius of cells in the chaining mesh
      const unsigned r = MAX(0, std::ceil(h_loc));
      // Get a vector of neighbor cell indices
      // const auto my_neighbor_cells = CMHelper.get_cell_neighbor_idx(key, r);
      // Reset 
      density(p_idx) = 0.0;
      // Loop over neighbor cells
      // for(std::size_t n_cell_idx = 0;
      //     n_cell_idx < my_neighbor_cells.size();
      //     ++n_cell_idx){
      CMHelper.it_over_neighbors(key, r, 
        [&] (const std::size_t other_idx){
          const auto& other_pos = position(other_idx);
          Vec<T, DIM> d = other_pos - position(p_idx);
          T rij = Kokkos::sqrt(d.dot(d));
          density(p_idx) += 
            mass(other_idx)*K(rij, h_loc*h);
        }
      );
      // if(r > 1) std::cout << r << " " << h_loc << " " << density(p_idx)*std::pow(h_loc*h, DIM) << " " << mass_target << std::endl;
    }

    // Perform the integration over the smoothing kernels
    void smoothen(){ 
      viscosity_factor<T, DIM> visc(1.);
      updateNeighbors();

      const std::size_t N_particles = position.size();

      // BIG TODO: ADD GRAD H TERMS
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Initial
          compute_density(p_idx, 1);
          // First: compute smoothing kernel size
          // Helper function
          auto my_mass_func = [&](const T h_) -> T {
            // If it is smaller than the chaining mesh resolution
            // just tell it to terminate
            // Otherwise continue
            compute_density(p_idx, h_);
            return std::pow(h*h_, DIM)*density(p_idx) - mass_target;
          };
          T h0, h1, h_current;
          h_current = smoothing_kernel_sizes(p_idx)/h;
          // Initial values for smoothing kernel sizes
          const T current_mass = density(p_idx)*std::pow(h_current*h, DIM);
          if(current_mass > mass_target){
            // Too much mass, give a slope towards smaller h
            h1 = h_current*0.8;
            h0 = h_current;
          } else{
            // Opposite, give a slope which goes towards bigger h
            h1 = h_current*1.2;
            h0 = h_current;
          }
          // Solve equation to get to smoothing kernel size
          // T temp = secant(my_mass_func, h0, h1);
          const auto fac = MAX(eps, secant(my_mass_func, h0, h1));
          smoothing_kernel_sizes(p_idx) = fac*h;
          // std::cout << fac << std::endl;
          // Update pressure while we're at it (TODO: add grad_h)
          pressure(p_idx) = entropy(p_idx)*
            pow(density(p_idx), Adiabatic_index);
        }
      );

      //compute gradh
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          gradh(p_idx) = 0;

          T aux_loop = 0;
          auto key = CMHelper.cell_idx(position(p_idx));
          const unsigned r = MAX(0, std::ceil(smoothing_kernel_sizes(p_idx)/h));
          // Loop over neighbor cells
          CMHelper.it_over_neighbors(key, r, 
            [&] (const std::size_t other_idx){
              if(p_idx != other_idx){
                const auto& other_pos = position(other_idx);
                Vec<T, DIM> d = other_pos - position(p_idx);
                T rij = Kokkos::sqrt(d.dot(d));
                aux_loop += mass(other_idx)*K.grad_h(rij, smoothing_kernel_sizes(p_idx));
              }
            }
          );
         gradh(p_idx) = 1.0+(1.0/DIM)*(smoothing_kernel_sizes(p_idx)/density(p_idx)) * aux_loop;
        }
      );

      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Reset
          accel(p_idx) = 0.0;
          d_entropy(p_idx) = 0.0;

          auto key = CMHelper.cell_idx(position(p_idx));
          const unsigned r = MAX(0, std::ceil(smoothing_kernel_sizes(p_idx)/h));
          // std::cout << r << std::endl;
          // const auto my_neighbor_cells = CMHelper.get_cell_neighbor_idx(key);
          // Loop over neighbor cells
          CMHelper.it_over_neighbors(key, r, 
            [&] (const std::size_t other_idx){
              if(p_idx != other_idx){
                const auto& other_pos = position(other_idx);
                const auto& other_vel = velocity(other_idx);
                Vec<T, DIM> d = other_pos - position(p_idx);
                Vec<T, DIM> vel = other_vel - velocity(p_idx);
                T rij = Kokkos::sqrt(d.dot(d));
                T vij = Kokkos::sqrt(vel.dot(vel));

                if constexpr (viscous) {

                  T aux_1 = pressure(p_idx)/(pow(density(p_idx)+ eps,2) * gradh(p_idx));

                  T aux_2 = pressure(other_idx)/(pow(density(other_idx)+ eps,2) * gradh(other_idx));

                  accel(p_idx) -= mass(other_idx) * (aux_1 * (-((d)*K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps)) +
                   aux_2 * (-((d)*K.grad_r(rij, smoothing_kernel_sizes(other_idx)))/(rij + eps)));
                  T density_mean = (density(other_idx) + density(p_idx))/2;

                  // Mean sound velocity
                  const T c_ij_bar = (Kokkos::sqrt(abs(Adiabatic_index*pressure(p_idx)/
                          (density(p_idx) + eps)))
                      + Kokkos::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                          (density(other_idx) + eps))))/2.;

                  T mean_h = 0.5*(smoothing_kernel_sizes(p_idx) + smoothing_kernel_sizes(other_idx));
                  Vec<T, DIM> sym_grad_W = 0.5 *( (-((d)*K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps))
                   + (-((d)*K.grad_r(rij, smoothing_kernel_sizes(other_idx)))/(rij + eps)));
                  accel(p_idx) -= 
                    mass(other_idx)*sym_grad_W*
                      visc(c_ij_bar, density_mean, d, vel, mean_h);

                  //std::cout << visc(c_ij_bar, density_mean, d, vel, mean_h) << std::endl;
                  d_entropy(p_idx) += 
                  ((Adiabatic_index-1.0)/std::pow(density(p_idx)+eps,
                    Adiabatic_index -1.0))*
                  (0.5*mass(other_idx)*
                   visc(c_ij_bar, density_mean, d, vel, mean_h)*
                    vel.dot( sym_grad_W ));

                } 
                else {
                  T aux = ((pressure(other_idx)/
                                pow(density(other_idx) + eps,2))
                              +((pressure(p_idx)/
                                pow(density(p_idx) + eps,2))));
                  accel(p_idx) -=
                    (-((d)*K.grad_r(rij, h))/(rij + eps))*(mass(other_idx))*aux;
                }
              }
            }
          );
        }
      );
    }

    /**
     * @brief A method that should be used to execute/advance a step of simulation.
     *
     *  In a derived class, the user must override this method to implement
     *  their time integration method for solving the considered governing equation.
     *  This function performs a step of a Kick-Drift-Kick leap-frog time integration.
     *  TODO  We might have an issue: 
     *        In the viscous case, the RHS of the second order ODE (acceleration) depends
     *        on the velocity, which is needed to compute the update for the velocity,
     *        i.e. we potentially get a non-linear system of equations for the 
     *        velocity.
     *  TODO  Determine time-step adaptively to ensure stability
     */
    void step() { 
      const std::size_t N_particles = position.size();
      smoothen();
      // Kick & Drift
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Update velocity with half of the acceleration
          velocity(p_idx) += accel(p_idx) * dt / 2.;
          // Update position
          position(p_idx) += velocity(p_idx) * dt;
          // Update entropy
          entropy(p_idx) += d_entropy(p_idx) * dt / 2.;

          // TODO: IMPLEMENT BOUNDARY CONDITIONS PROPERLY
          // Apply bc
          if(position(p_idx)[0] < low_[0] + eps)
            position(p_idx)[0] = low_[1] + L_[1] - eps; 
          if(position(p_idx)[0] > low_[1] + L_[1] - eps)
            position(p_idx)[0] = low_[0] + eps; 

          if(position(p_idx)[1] < low_[1] + eps ||
             position(p_idx)[1] > low_[1] + L_[1] - eps) {
            // Reflect the position & clamp position
            position(p_idx)[1] = std::max(low_[1] + eps, 
                std::min(low_[1] + L_[1] - eps, position(p_idx)[1])); 
            // Reverse the speed in the respective direction
            velocity(p_idx)[1] *= -1.0;
          }
        }
      );

      // TODO: Here we want the acceleration at x_{i + 1}, but
      // in the viscous case that depends on v_{i + 1}, which is 
      // yet to be found

      // Update acceleration
      smoothen();

      // Kick again
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Update entropy
          entropy(p_idx) += d_entropy(p_idx) * dt / 2.;
          // Update velocity with the other half of the acceleration
          velocity(p_idx) += accel(p_idx) * dt / 2.;
        }
      );
    }

    /**
     * @brief The main for loop fro running a simulation.
     *
     * This method performs a simulation run by calling pre_step, advance, and post_step
     * in a loop for a specified number of time steps.
     *
     * @param nt The number of time steps to run the simulation.
     */

    T density_arbitrary_pos(const Vec<T, DIM> pos) const {
         T density_final = 0.0;
         for (int i = 0; i < position.size(); i++)
         {
             Vec<T, DIM> r_vec = pos - position(i);
             T r = Kokkos::sqrt(r_vec.dot(r_vec));

             density_final += mass(i)*K(r, 2*h);
         }
         return density_final;
    }

    T pressure_arbitrary_pos(const Vec<T, DIM> pos) const {
         T pressure_final = 0;
         for (int i = 0; i < position.size(); i++)
         {
             Vec<T, DIM> r_vec = pos - position(i);
             T r = Kokkos::sqrt(r_vec.dot(r_vec));
             T volume_element = mass(i)/(density(i)+eps);

             pressure_final += pressure(i)*K(r, 2*h)*volume_element;
         }
         return pressure_final;
    }

    T velocity_arbitrary_pos(const Vec<T, DIM> pos) const {
         T velocity_final = 0.0;
         for (int i = 0; i < position.size(); i++)
         {
             Vec<T, DIM> r_vec = pos - position(i);
             T r = Kokkos::sqrt(r_vec.dot(r_vec));
             T volume_element = mass(i)/(density(i)+eps);

             velocity_final += velocity(i)[0]*K(r, 2*h)*volume_element;
         }
         return velocity_final;
    }



};
