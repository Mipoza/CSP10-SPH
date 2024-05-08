#include <Kokkos_Core.hpp>
#include <iostream>
#include <cmath>

#include "std_array_overloads.hpp"
#include "ChainingMesh.hpp"
#include "kernels.hpp"

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
         bool PERIODIC = false,
         bool viscous = false,
         class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHManager {
    // Parameters
    T dt, Adiabatic_index, h;
    Vec<T, DIM> L_, low_;
    // The kernel itself
    KERNEL K;
    // Helper for nearest neighbors
    ChainingMeshHelper<T, DIM, PERIODIC> CMHelper;

    // Physical quantities
    Kokkos::View<T*> mass, density, pressure, entropy, d_entropy;
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
      Kokkos::View<T*> density_("Density", N);
      density = density_;
      Kokkos::View<T*> pressure_("Pressure", N);
      pressure = pressure_;
      Kokkos::View<T*> entropy_("Entropy", N);
      entropy = entropy_;
      Kokkos::View<T*> d_entropy_("DEntropy", N);
      d_entropy = d_entropy_;
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
              std::vector<T> entropy_part) {
      int N_new = R_part.size();
      create_particles(N_new);

      auto R_host = Kokkos::create_mirror_view(position);
      auto v_host = Kokkos::create_mirror_view(velocity);
      auto m_host = Kokkos::create_mirror_view(mass);
      auto entropy_host = Kokkos::create_mirror_view(entropy);

      for (unsigned int i = 0; i < N_new; ++i) {
          R_host(i) = R_part[i];
          v_host(i) = v_part[i];
          m_host(i) = m_part[i];
          entropy_host(i) = entropy_part[i];
      }

      Kokkos::deep_copy(position, R_host);
      Kokkos::deep_copy(velocity, v_host);
      Kokkos::deep_copy(mass, m_host);
      Kokkos::deep_copy(entropy, entropy_host);
    }

    // Perform the integration over the smoothing kernels
    void smoothen(){ 
      viscosity_factor<T, DIM> visc(1.);
      updateNeighbors();

      const std::size_t N_particles = position.size();

      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA  (const std::size_t p_idx){
          auto key = CMHelper.cell_idx(position(p_idx));
          const auto my_neighbor_cells = CMHelper.get_cell_neighbor_idx(key);
          // Reset 
          density(p_idx) = 0.0;
          // Loop over neighbor cells
          for(std::size_t n_cell_idx = 0;
              n_cell_idx < my_neighbor_cells.size();
              ++n_cell_idx){
            // Loop over particles in neighbor cells
            const std::size_t start_idx_base = 
              CMHelper.start_idx(my_neighbor_cells[n_cell_idx]);
            for(std::size_t other_idx_ = 0;
                other_idx_ < CMHelper.cell_size(my_neighbor_cells[n_cell_idx]);
                ++other_idx_){
              const std::size_t other_idx = start_idx_base + other_idx_;
              const auto& other_pos = position(other_idx);
              Vec<T, DIM> d = other_pos - position(p_idx);
              T rij = Kokkos::sqrt(d.dot(d));
              density(p_idx) += 
                mass(other_idx)*K(rij, h);
            }
          }
          pressure(p_idx) = entropy(p_idx)*
            pow(density(p_idx), Adiabatic_index);

        }
      );

      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Reset
          accel(p_idx) = 0.0;
          d_entropy(p_idx) = 0.0;

          auto key = CMHelper.cell_idx(position(p_idx));
          const auto my_neighbor_cells = CMHelper.get_cell_neighbor_idx(key);
          // Loop over neighbor cells
          for(std::size_t n_cell_idx = 0;
              n_cell_idx < my_neighbor_cells.size();
              ++n_cell_idx){
            // Loop over particles in neighbor cells
            const std::size_t start_idx_base = 
              CMHelper.start_idx(my_neighbor_cells[n_cell_idx]);
            for(std::size_t other_idx_ = 0;
                other_idx_ < CMHelper.cell_size(my_neighbor_cells[n_cell_idx]);
                ++other_idx_){
              const std::size_t other_idx = start_idx_base + other_idx_;
              if(p_idx != other_idx){
                const auto& other_pos = position(other_idx);
                const auto& other_vel = velocity(other_idx);
                Vec<T, DIM> d = other_pos - position(p_idx);
                Vec<T, DIM> vel = other_vel - velocity(p_idx);
                T rij = Kokkos::sqrt(d.dot(d));
                T vij = Kokkos::sqrt(vel.dot(vel));

                if constexpr (viscous) {
                  T aux = ((pressure(other_idx)/pow(density(other_idx)
                          + eps,2)) + 
                      ((pressure(p_idx)/pow(density(p_idx) + eps,2))));
                  T aux_1 = (((pressure(p_idx)/pow(density(p_idx) + eps,2))));

                  accel(p_idx) -=
                      (-((d)*K.grad_r(rij, h))/(rij + eps))*
                      (mass(other_idx))*aux;
                  T density_mean = (density(other_idx) + density(p_idx))/2;

                  // Mean sound velocity
                  const T c_ij_bar = (Kokkos::sqrt(abs(Adiabatic_index*pressure(p_idx)/
                          (density(p_idx) + eps)))
                      + Kokkos::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                          (density(other_idx) + eps))))/2.;


                  accel(p_idx) -=
                    mass(other_idx)*(-((d)*K.grad_r(rij, h))/(rij + eps))*
                      visc(c_ij_bar, density_mean, d, vel, h);


                  d_entropy(p_idx) += 
                  ((Adiabatic_index-1.0)/std::pow(density(p_idx)+eps,
                    Adiabatic_index -1.0))*
                  (0.5*mass(other_idx)*
                   visc(c_ij_bar, density_mean, d, vel, h)*
                    vel.dot((-(d)*K.grad_r(rij, h))/(rij + eps)));

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
          }
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
