#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <iostream>
#include <cmath>

#include "std_array_overloads.hpp"
#include "ChainingMesh.hpp"
#include "kernels.hpp"
// #include "solve.hpp"
#include "regula_falsi.hpp"
#include <limits>

#define MAX(x, y) (x > y ? x : y)
#define eps (std::numeric_limits<T>::epsilon())

template <unsigned DIM>
constexpr auto SPHERE_VOL_FAC(){
    return std::pow(M_PI, DIM/2.)/std::tgamma(DIM/2. + 1);
}

template <typename T, unsigned int DIM>
struct viscosity_factor{
  T alpha, beta;

  viscosity_factor() = default;

  viscosity_factor(T alpha_): alpha(alpha_), beta(2*alpha_){}

    inline T operator()(T c, T density, Vec<T, DIM> dis, Vec<T, DIM> vel, T h) const {
        T rij = Kokkos::sqrt(dis.dot(dis));
        T dot_product = dis.dot(vel);
        T mu = h*dot_product/(std::pow(rij, 2) + 0.01*std::pow(h, 2));

        if (dot_product < 0) 
          return (-(alpha * c * mu) + (beta * mu * mu)) / (density + eps);
        else 
          return 0;
    }
};

template<typename T, unsigned int DIM, 
         const bool PERIODIC[DIM],
         bool viscous = false,
         class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHManager {
    // Parameters
    T dt, Adiabatic_index, h, n_target = -1;
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
      CMHelper(low, extent, 2*h_), 
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

    template <unsigned DIR = 0>
    KOKKOS_INLINE_FUNCTION
    T dist(const Vec<T, DIM>& x, const Vec<T, DIM>& y, T res = 0){
      if constexpr(DIR < DIM){
        T d = Kokkos::abs(y[DIR] - x[DIR]);
        if constexpr(PERIODIC[DIR]){
          d = MIN(d, MIN(
                    Kokkos::abs(y[d] - (x[d] + static_cast<T>(L_[DIR]))),
                    Kokkos::abs((y[d] + static_cast<T>(L_[DIR])) - x[d])
                  )
              );
        }
        return dist<DIR + 1>(x, y, res + d*d);
      }
      return Kokkos::sqrt(res);
    }

    template <unsigned DIR = 0>
    KOKKOS_INLINE_FUNCTION
    void dist_vec_dir(const Vec<T, DIM>& x, const Vec<T, DIM>& y,
                             Vec<T, DIM>& res){
      if constexpr(DIR < DIM){
        res[DIR] = y[DIR] - x[DIR];
        if constexpr(PERIODIC[DIR]){
          T d = Kokkos::abs(res[DIR]);
          if(Kokkos::abs(res[DIR] - static_cast<T>(L_[DIR])) < d)
            res[DIR] = res[DIR] - static_cast<T>(L_[DIR]);
          else if(Kokkos::abs(res[DIR] + static_cast<T>(L_[DIR])) < d)
            res[DIR] = res[DIR] + static_cast<T>(L_[DIR]);
        }
        dist_vec_dir<DIR + 1>(x, y, res);
      }
    }

    KOKKOS_INLINE_FUNCTION
    Vec<T, DIM> dist_vec(const Vec<T, DIM>& x, const Vec<T, DIM>& y){
      Vec<T, DIM> res;
      for(unsigned d = 0; d < DIM; ++d) res[d] = 0; //y[d] - x[d];
      dist_vec_dir(x, y, res);
      return res;
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
              T n_target_ = -1) {
      unsigned N_new = R_part.size();
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
      if(n_target_< 0)
        n_target = K(0., h) * 
                      std::pow(h*K.supp_radius, DIM)*SPHERE_VOL_FAC<DIM>() 
                      * (-n_target_);
      else 
        n_target = n_target_;

      // Compute kernels right away, otherwise it would have to be done
      // before the first step
      updateNeighbors();
      compute_kernels();
    }

    // For debugging
    std::size_t count_neighbors(const std::size_t p_idx){
      std::size_t count = 0;
      // Loop over neighbor cells
      CMHelper.it_over_neighbors(position(p_idx), 
         K.supp_radius*smoothing_kernel_sizes(p_idx), 
        [&] (const std::size_t other_idx){
          const auto& other_pos = position(other_idx);
          // Vec<T, DIM> d = other_pos - position(p_idx);
          // T rij = Kokkos::sqrt(d.dot(d));
          T rij = dist(position(p_idx), other_pos);
          if(rij < K.supp_radius * smoothing_kernel_sizes(p_idx)) ++count;
        }
      );
      return count;
    }
    
    KOKKOS_INLINE_FUNCTION
    T compute_n_density(const std::size_t p_idx, const T h_loc){
      T n = 0;
      // Loop over neighbor cells
      CMHelper.it_over_neighbors(position(p_idx), 
         K.supp_radius*h_loc*h, 
        [&] (const std::size_t other_idx){
          const auto& other_pos = position(other_idx);
          // Vec<T, DIM> d = other_pos - position(p_idx);
          // T rij = Kokkos::sqrt(d.dot(d));
          const T rij = dist(position(p_idx), other_pos);
          n += K(rij, h_loc*h);
        }
      );
      return n * std::pow(h_loc*h*K.supp_radius, DIM)*SPHERE_VOL_FAC<DIM>(); 
    }

    void compute_kernels(){
      const std::size_t N_particles = position.size();
#ifdef _DEBUG
      T minh = L_[0], maxh = 0, 
        min_mass = n_target*N_particles, max_mass = 0,
        minerr = min_mass, maxerr = 0;
      T* minh_ptr = &minh;
      T* maxh_ptr = &maxh;
      T* min_mass_ptr = &min_mass;
      T* max_mass_ptr = &max_mass;
      T* minerr_ptr = &minerr;
      T* maxerr_ptr = &maxerr;
#endif
      // Find the smoothing kernel sizes
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // First: compute smoothing kernel size
          // Helper function
          auto my_mass_func = [&](const T h_) -> T {
            // compute_density(p_idx, h_);
            return compute_n_density(p_idx, h_) - n_target;
          };

          const T h_current = smoothing_kernel_sizes(p_idx)/h;
          // Initial values for smoothing kernel sizes
          const T current_n = compute_n_density(p_idx, h_current);
          T h0, h1;
          T fac = 1;
          if(current_n > n_target){
            // Too much mass, give a slope towards smaller h
            h1 = h_current;
            h0 = 0;
            fac = illinois_lower_bound(h0, h1, my_mass_func);
          } else if(current_n < n_target){
            // Opposite, give a slope which goes towards bigger h
            h1 = h_current*2;
            h0 = h_current;
            fac = illinois_lower_bound(h0, h1, my_mass_func);
          }
          // Else keep h_current

          smoothing_kernel_sizes(p_idx) = fac*h;
#ifdef _DEBUG
          Kokkos::atomic_max(maxh_ptr, fac*h);
          Kokkos::atomic_min(minh_ptr, fac*h);
          Kokkos::atomic_max(max_mass_ptr, 
              density(p_idx)*std::pow(fac*h, DIM)*SPHERE_VOL_FAC<DIM>());
          Kokkos::atomic_min(min_mass_ptr, 
              density(p_idx)*std::pow(fac*h, DIM)*SPHERE_VOL_FAC<DIM>());
          Kokkos::atomic_max(maxerr_ptr, std::abs(my_mass_func(fac)));
          Kokkos::atomic_min(minerr_ptr, std::abs(my_mass_func(fac)));
#endif
        }
      );
#ifdef _DEBUG
      std::cout << std::endl;
      std::cout << "Min/Max h/h0: " << minh/h << "/" << maxh/h << std::endl;
      std::cout << "Min/Max mass: " << min_mass << "/" << max_mass << std::endl;
      std::cout << "Mass Target: " << n_target << std::endl;
      std::cout << "Min/Max relerr: " << minerr/n_target << "/"
                                      << maxerr/n_target << std::endl;
      std::cout << std::endl;
#endif

    }

    // Perform the integration over the smoothing kernels
    void smoothen(){ 
      viscosity_factor<T, DIM> visc(0.6);

      const std::size_t N_particles = position.size();
      //compute gradh
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          density(p_idx) = 0.0;
          gradh(p_idx) = 0;

          T aux_loop = 0;
          // Loop over neighbor cells
          CMHelper.it_over_neighbors(position(p_idx),
             K.supp_radius*smoothing_kernel_sizes(p_idx), 
            [&] (const std::size_t other_idx){
              const auto& other_pos = position(other_idx);
              T rij = dist(position(p_idx), other_pos);
              density(p_idx) += mass(other_idx) * K(rij, smoothing_kernel_sizes(p_idx));

              if(p_idx != other_idx){
                // Vec<T, DIM> d = other_pos - position(p_idx);
                // T rij = Kokkos::sqrt(d.dot(d));
                aux_loop += mass(other_idx)*K.grad_h(rij, smoothing_kernel_sizes(p_idx));
              }
              // Also compute density alongside it
            }
          );
          // Have to compute pressure sooner or later
          pressure(p_idx) = entropy(p_idx)*
            std::pow(density(p_idx), Adiabatic_index);

          gradh(p_idx) = 1.0 + (1.0/DIM)*(smoothing_kernel_sizes(p_idx)/
                                          (density(p_idx) + eps))
                        * aux_loop;
        }
      );

      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Reset
          accel(p_idx) = 0.0;
          d_entropy(p_idx) = 0.0;
          // Loop over neighbor cells
          CMHelper.it_over_neighbors(position(p_idx), 
             K.supp_radius*smoothing_kernel_sizes(p_idx), 
            [&] (const std::size_t other_idx){
              if(p_idx != other_idx){
                const auto& other_pos = position(other_idx);
                // Vec<T, DIM> d = other_pos - position(p_idx);
                const Vec<T, DIM> d = dist_vec(position(p_idx), other_pos);
                const T rij = Kokkos::sqrt(d.dot(d));

                // Symmetric terms
                T aux_1 = pressure(p_idx)/(std::pow(density(p_idx) + eps, 2) *
                                  gradh(p_idx));

                T aux_2 = pressure(other_idx)/(std::pow(density(other_idx) + eps, 2) *
                                  gradh(other_idx));

                // Viscous or non-viscous, these terms are present
                accel(p_idx) -= mass(other_idx) * (
                    aux_1 * (-((d)*
                        K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps)) +
                    aux_2 * (-((d)*
                        K.grad_r(rij, smoothing_kernel_sizes(other_idx)))/(rij + eps)));

                if constexpr (viscous) {
                  const auto& other_vel = velocity(other_idx);
                  Vec<T, DIM> vel = other_vel - velocity(p_idx);
                  T density_mean = (density(other_idx) + density(p_idx))/2.;

                  // Mean sound velocity
                  const T c_ij_bar = (Kokkos::sqrt(abs(Adiabatic_index*pressure(p_idx)/
                          (density(p_idx) + eps)))
                      + Kokkos::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                          (density(other_idx) + eps))))/2.;

                  T mean_h = 0.5*(smoothing_kernel_sizes(p_idx) +
                                  smoothing_kernel_sizes(other_idx));
                  Vec<T, DIM> sym_grad_W = 0.5 *( (-((d)*
                          K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps))
                   + (-((d)*K.grad_r(rij, 
                         smoothing_kernel_sizes(other_idx)))/(rij + eps)));
                  //Vec<T, DIM> sym_grad_W = -((d)*
                  //        K.grad_r(rij, mean_h)/(rij + eps));
                  accel(p_idx) -= 
                    mass(other_idx)*sym_grad_W*
                      visc(c_ij_bar, density_mean, d, vel, mean_h);

                  //std::cout << visc(c_ij_bar, density_mean, d, vel, mean_h) << std::endl;
                  d_entropy(p_idx) += 
                  -((Adiabatic_index-1.0)/std::pow(density(p_idx)+eps,
                    Adiabatic_index -1.0))*
                  (0.5*mass(other_idx)*
                   visc(c_ij_bar, density_mean, d, vel, mean_h)*
                    vel.dot( sym_grad_W ));
                } 
              }
            }
          );
        }
      );
    }

    template <unsigned DIR = 0>
    KOKKOS_INLINE_FUNCTION
    void apply_bcs(const std::size_t p_idx){
      if constexpr(DIR < DIM){
        // Periodic
        if constexpr(PERIODIC[DIR]){
          if(position(p_idx)[DIR] < low_[DIR] + eps)
            position(p_idx)[DIR] += L_[DIR];
          else if(position(p_idx)[DIR] > low_[DIR] + L_[DIR] - eps)
            position(p_idx)[DIR] -= L_[DIR]; 
        }
        // Else reflective
        else {
          if(position(p_idx)[DIR] < low_[DIR] + eps) {
           // Reflect the position
           position(p_idx)[DIR] = 2. * low_[DIR] - position(p_idx)[DIR];
           // Reverse the speed in the respective direction
           velocity(p_idx)[DIR] *= -1.0;
          }
          else if(position(p_idx)[DIR] > low_[DIR] + L_[DIR] - eps) {
           // Reflect the position
           position(p_idx)[DIR] = 2. * (low_[DIR] + L_[DIR]) - position(p_idx)[DIR];
           // Reverse the speed in the respective direction
           velocity(p_idx)[DIR] *= -1.0;
          }
        }
        apply_bcs<DIR + 1>(p_idx);
      }
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
          // DEBUG
          // velocity(p_idx) = accel(p_idx) * dt;

          // Update position
          // Has to be the ONLY place in the simulation where it is changed
          // It shold always be followed by compute_kernels() before the next
          // smoothen()
          position(p_idx) += velocity(p_idx) * dt;
          // Update entropy
          entropy(p_idx) += d_entropy(p_idx) * dt / 2.;
          // Apply boundary conditions
          apply_bcs(p_idx);
        }
      );

      // TODO: Here we want the acceleration at x_{i + 1}, but
      // in the viscous case that depends on v_{i + 1}, which is 
      // yet to be found

      // Update acceleration
      updateNeighbors();
      compute_kernels();
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
             // Vec<T, DIM> r_vec = pos - position(i);
             // T r = Kokkos::sqrt(r_vec.dot(r_vec));
             T r = dist(pos, position(i));

             density_final += mass(i)*K(r, 2*h);
         }
         return density_final;
    }

    T pressure_arbitrary_pos(const Vec<T, DIM> pos) const {
         T pressure_final = 0;
         for (int i = 0; i < position.size(); i++)
         {
             //Vec<T, DIM> r_vec = pos - position(i);
             //T r = Kokkos::sqrt(r_vec.dot(r_vec));
             T r = dist(pos, position(i));
             T volume_element = mass(i)/(density(i)+eps);

             pressure_final += pressure(i)*K(r, 2*h)*volume_element;
         }
         return pressure_final;
    }

    T velocity_arbitrary_pos(const Vec<T, DIM> pos) const {
         T velocity_final = 0.0;
         for (int i = 0; i < position.size(); i++)
         {
             // Vec<T, DIM> r_vec = pos - position(i);
             // T r = Kokkos::sqrt(r_vec.dot(r_vec));
             T r = dist(pos, position(i));
             T volume_element = mass(i)/(density(i)+eps);

             velocity_final += velocity(i)[0]*K(r, 2*h)*volume_element;
         }
         return velocity_final;
    }



};
