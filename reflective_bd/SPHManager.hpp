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
#define eps (16*std::numeric_limits<T>::epsilon())

template <unsigned DIM>
constexpr auto SPHERE_VOL_FAC(){
    return std::pow(M_PI, DIM/2.)/std::tgamma(DIM/2. + 1);
}

template<typename T, unsigned int DIM, 
         const bool PERIODIC[DIM],
         bool viscous = false,
         bool balsara = false,
         class KERNEL = CubicSplineKernel<T, DIM>>
struct SPHManager {
    // Parameters
    T dt, Adiabatic_index, h, n_target = -1, CFL, dt_max;
    // For deciding whether to re-mesh
    T avgh;
    Vec<T, DIM> L_, low_;
    // The kernel itself
    KERNEL K;
    // Helper for nearest neighbors
    ChainingMesh<T, DIM, PERIODIC> CM;
    // Viscosity parameters
    T alpha, beta;

    // Physical quantities
    Kokkos::View<T*> mass, density, pressure, 
                     entropy, d_entropy, smoothing_kernel_sizes, gradh,
                     div_v;
    Kokkos::View<Vec<T, DIM>*> position, velocity, accel, curl_v;

    // Constructor
    SPHManager(Vec<T, DIM>& low, Vec<T, DIM>& extent, 
               T CFL_, T h_, 
               T Adiabatic_index_,
               T dt_max_,
               T alpha_ = 0.6,
               T beta_ = 1.2) : 
      h(h_), dt_max(dt_max_),
      CM(low, extent, K.supp_radius*h_), 
      Adiabatic_index(Adiabatic_index_), 
      L_(extent), low_(low),
      alpha(alpha_), beta(beta_),
      CFL(CFL_), avgh(h_) {}

/* Helper functions */

    // (Neumann and Richtmeyer) Viscosity factor
    KOKKOS_INLINE_FUNCTION
    T visc(const T& c, const T& density, 
           const Vec<T, DIM>& dis, const Vec<T, DIM>& vel,
           const T& h) const {
      const T rij = Kokkos::sqrt(dis.dot(dis));
      const T dot_product = dis.dot(vel);
      const T mu = h*dot_product/(std::pow(rij, 2) + 0.01*std::pow(h, 2));

      if (dot_product < 0) 
        return (-(alpha * c * mu) + (beta * mu * mu)) / (density + eps);
      else 
        return 0;
    }

    // Distance between two points, possibly with 
    // periodic boundary conditions
    template <unsigned DIR = 0>
    __attribute__((always_inline))
    T dist2(const Vec<T, DIM>& x, const Vec<T, DIM>& y, T res = 0){
      if constexpr(DIR < DIM){
        T d = y[DIR] - x[DIR];
        if constexpr(PERIODIC[DIR]){
          if(d > L_[DIR]/2)
            d -= L_[DIR];
          if(d <= -L_[DIR]/2)
            d += L_[DIR];
        }
        return dist2<DIR + 1>(x, y, res + d*d);
      }
      return res;
    }

    KOKKOS_INLINE_FUNCTION
    T dist(const Vec<T, DIM>& x, const Vec<T, DIM>& y){
      return std::sqrt(dist2(x, y));
    }

    template <unsigned DIR = 0>
    KOKKOS_INLINE_FUNCTION
    void dist_vec_dir(const Vec<T, DIM>& x, const Vec<T, DIM>& y,
                      Vec<T, DIM>& res){
      if constexpr(DIR < DIM){
        res[DIR] = y[DIR] - x[DIR];
        if constexpr(PERIODIC[DIR]){
          if(res[DIR] > L_[DIR]/2)
            res[DIR] -= L_[DIR];
          if(res[DIR] <= -L_[DIR]/2)
            res[DIR] += L_[DIR];
        }
        dist_vec_dir<DIR + 1>(x, y, res);
      }
    }

    KOKKOS_INLINE_FUNCTION
    Vec<T, DIM> dist_vec(const Vec<T, DIM>& x, const Vec<T, DIM>& y){
      Vec<T, DIM> res;
      res = 0;
      dist_vec_dir(x, y, res);
      return res;
    }

    // For debugging
    std::size_t count_neighbors(const std::size_t p_idx){
      std::size_t count = 0;
      // Loop over neighbor cells
      CM.it_over_neighbors(position(p_idx), 
         K.supp_radius*smoothing_kernel_sizes(p_idx), 
        [&] (const std::size_t other_idx,
             const Vec<T, DIM>& other_pos){
          const T rij = dist(position(p_idx), other_pos);
          if(rij < K.supp_radius * smoothing_kernel_sizes(p_idx)) ++count;
        },
      position);
      return count;
    }

/* The actual SPH */

    // Allocate space for the particles
    void create_particles(std::size_t N){
      // Scalar quantities
      Kokkos::View<T*> mass_("Mass", N);
      mass = mass_;
      Kokkos::View<T*> gradh_("gradh", N);
      gradh = gradh_;
      Kokkos::deep_copy(gradh, 0);
      Kokkos::View<T*> density_("Density", N);
      density = density_;
      Kokkos::View<T*> pressure_("Pressure", N);
      pressure = pressure_;
      Kokkos::View<T*> entropy_("Entropy", N);
      entropy = entropy_;
      Kokkos::View<T*> d_entropy_("DEntropy", N);
      d_entropy = d_entropy_;
      if constexpr(viscous && balsara){
        Kokkos::View<T*> div_v_("div_v", N);
        div_v = div_v_;
      }
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
      Kokkos::View<Vec<T, DIM>*> curl_v_("curl_v", N);
      curl_v = curl_v_;
      if constexpr(viscous && balsara){
        static_assert(DIM > 1, "Balsara correcetion only does something in DIM > 1!");
      }
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
      smoothen<false, true>();
      compute_kernels();
    }


    // Update the ChainingMesh
    template <typename... ARGS>
    void updateNeighbors(ARGS&... optional_sort){
      CM.partition(position);
      // Sort for data locality
      if constexpr(viscous && balsara){
        CM.sort(position, mass, gradh, smoothing_kernel_sizes, 
                                density, pressure, entropy, d_entropy, 
                                velocity, accel, div_v, curl_v,
                                optional_sort...);
      }
      else { 
        CM.sort(position, mass, gradh, smoothing_kernel_sizes, 
                                density, pressure, entropy, d_entropy, 
                                velocity, accel,
                                optional_sort...);
      }
      
      // AT THE END, sort position
      CM.sort(position, position);
    }

    // Compute the "number density" and the gradient
    // Useful for newton's method, but newton's method is not rly
    // better than the illinois algorithm (illinois also has guaranteed convergence!)
    KOKKOS_INLINE_FUNCTION std::pair<T, T>
    compute_n_density_grad(const std::size_t p_idx, const T h_loc){
      T n = 0,
        dn = 0;
      // Loop over neighbor cells
      CM.it_over_neighbors(position(p_idx), 
         K.supp_radius*h_loc*h, 
        [&] (const std::size_t other_idx, 
             const Vec<T, DIM>& other_pos){
          const T rij = dist(position(p_idx), other_pos);
          n += K(rij, h_loc*h);
          dn += K.grad_h(rij, h_loc*h);
        },
      position);
      dn *= std::pow(h*h_loc*K.supp_radius, DIM) * h;
      dn += DIM*std::pow(h_loc*h, (int)DIM - 1)*h*n*std::pow(K.supp_radius, DIM);
      return std::move(std::make_pair(
          n * std::pow(h_loc*h*K.supp_radius, DIM)*SPHERE_VOL_FAC<DIM>(),
          dn * SPHERE_VOL_FAC<DIM>()
          )); 
    }

    // Compute the "number density"
    KOKKOS_INLINE_FUNCTION T
    compute_n_density(const std::size_t p_idx, const T h_loc){
      T n = 0;
      // Loop over neighbor cells
      CM.it_over_neighbors(position(p_idx), 
         K.supp_radius*h_loc*h, 
        [&] (const std::size_t other_idx, 
             const Vec<T, DIM>& other_pos){
          const T rij = dist(position(p_idx), other_pos);
          n += K(rij, h_loc*h);
        },
      position);
      return std::move(n * std::pow(h_loc*h*K.supp_radius, DIM)*SPHERE_VOL_FAC<DIM>());
    }

    void compute_kernels(){
      const std::size_t N_particles = position.size();
      // Find the smoothing kernel sizes
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // First: compute smoothing kernel size
          // Helper function
          auto my_mass_func = [&](const T h_) {
            return std::move(compute_n_density(p_idx, h_) - n_target);
          };
          // Initial values for smoothing kernel sizes
          const T h_current = smoothing_kernel_sizes(p_idx)/h;
          auto current_n = compute_n_density(p_idx, h_current);
          T h0, h1;
          T fac = 1;
          if(current_n > n_target){
            // Too much mass, give a slope towards smaller h
            h1 = h_current;
            h0 = h_current * n_target/current_n;
            fac = illinois(h0, h1, my_mass_func);
          } else if(current_n < n_target){
            // Opposite, give a slope which goes towards bigger h
            h1 = h_current * n_target/current_n;
            h0 = h_current;
            fac = illinois(h0, h1, my_mass_func);
          } // Else keep h_current
          smoothing_kernel_sizes(p_idx) = fac*h;
        }
      );
    }

    // Perform the integration over the smoothing kernels
    template <bool VISC = viscous, bool ONLY_VISC = false>
    void smoothen(){ 
      const std::size_t N_particles = position.size();

      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          density(p_idx) = 0.0;
          gradh(p_idx) = 0;
          if constexpr(VISC && balsara){
            div_v(p_idx) = 0;
            curl_v(p_idx) = 0;
          }
          T aux_loop = 0;
          // Loop over neighbor cells
          CM.it_over_neighbors(position(p_idx),
             K.supp_radius*smoothing_kernel_sizes(p_idx), 
            [&] (const std::size_t other_idx, 
                 const Vec<T, DIM>& other_pos,
                 const Vec<T, DIM>& other_vel){
              const Vec<T, DIM> d = dist_vec(position(p_idx), other_pos);
              const T rij = std::sqrt(d.dot(d));

              density(p_idx) += mass(other_idx) * K(rij, smoothing_kernel_sizes(p_idx));
              if(p_idx != other_idx)
                aux_loop += mass(other_idx) * K.grad_h(rij, smoothing_kernel_sizes(p_idx));

              if constexpr(VISC && balsara){
                div_v(p_idx) += mass(other_idx)* 
                                (other_vel - velocity(p_idx)).dot(
                                  -(d/(rij + eps)
                                  * K.grad_r(rij, smoothing_kernel_sizes(p_idx)))
                                 );
                // Dimension 2: Rotate v by 90 deg and dot with grad W
                if constexpr(DIM == 2){
                  Vec<T, DIM> vij_rot;
                  vij_rot[0] =  d[1];
                  vij_rot[1] = -d[0];

                  // Choose one component
                  // NOT ACTUALLY the vector curl, but it's only one number anyway
                  curl_v(p_idx)[0] += mass(other_idx) *
                                        vij_rot.dot(
                                          -(d/(rij + eps)
                                          * K.grad_r(rij, smoothing_kernel_sizes(p_idx)))
                                        );
                } 
              }
            },
          position, velocity);
          if constexpr(VISC && balsara){
            div_v(p_idx) /= density(p_idx) + eps;
            curl_v(p_idx) /= density(p_idx) + eps;
          }
          // Have to compute pressure sooner or later
          pressure(p_idx) = entropy(p_idx)*
            std::pow(density(p_idx), Adiabatic_index);

          gradh(p_idx) = 1.0 + (1.0/DIM)*(smoothing_kernel_sizes(p_idx)/
                                          (density(p_idx) + eps))
                        * aux_loop;
        }
      );

      // For the Balsara correction
      if constexpr(VISC && balsara && DIM > 2){
        static_assert(false, "ERROR: CURL FOR DIMENSION > 2 NOT IMPLEMENTED");
      }

      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Reset
          accel(p_idx) = 0.0;
          d_entropy(p_idx) = 0.0;
          // Loop over neighbor cells
          CM.it_over_neighbors(position(p_idx), 
             K.supp_radius*smoothing_kernel_sizes(p_idx), 
            [&] (const std::size_t other_idx,
                 const Vec<T, DIM>& other_pos,
                 const Vec<T, DIM>& other_vel,
                 const Vec<T, DIM>& other_curl){
              if(p_idx != other_idx){
                const Vec<T, DIM> d = dist_vec(position(p_idx), other_pos);
                const T rij = d.norm();

                // Symmetric terms
                T aux_1 = pressure(p_idx)/(std::pow(density(p_idx) + eps, 2) *
                                  gradh(p_idx));

                T aux_2 = pressure(other_idx)/(std::pow(density(other_idx) + eps, 2) *
                                  gradh(other_idx));

                // We have an option to ignore the pressure-acceleration,
                // since we want to apply a split-step method
                if constexpr(!ONLY_VISC){
                  // Viscous or non-viscous, these terms are present
                  accel(p_idx) -= mass(other_idx) * (
                      aux_1 * (-((d)*
                          K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps)) +
                      aux_2 * (-((d)*
                          K.grad_r(rij, smoothing_kernel_sizes(other_idx)))/(rij + eps)));
                }

                if constexpr (VISC) {
                  const Vec<T, DIM> vel = other_vel - velocity(p_idx);

                  const T density_mean = (density(other_idx) + density(p_idx))/2.;
                  // Mean sound velocity
                  const T c_ij_bar = (Kokkos::sqrt(abs(Adiabatic_index*pressure(p_idx)/
                          (density(p_idx) + eps)))
                      + Kokkos::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                          (density(other_idx) + eps))))/2.;

                  const T mean_h = 0.5*(smoothing_kernel_sizes(p_idx) +
                                        smoothing_kernel_sizes(other_idx));
                  const Vec<T, DIM> sym_grad_W = 0.5 *( (-((d)*
                              K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps))
                     + (-((d)*K.grad_r(rij, 
                         smoothing_kernel_sizes(other_idx)))/(rij + eps)));

                  T fij = 1;
                  if constexpr(balsara){
                    // Balsara factor
                    const T fi = std::abs(div_v(p_idx))/(
                        std::abs(div_v(p_idx)) + curl_v(p_idx).norm() + // c_i/h_i 
                        1e-4 * std::sqrt(abs(Adiabatic_index*pressure(p_idx)/
                            (density(p_idx) + eps)))/
                          smoothing_kernel_sizes(p_idx)
                        );
                    const T fj = std::abs(div_v(other_idx))/(
                        std::abs(div_v(other_idx)) + other_curl.norm() + // c_j/h_j 
                        1e-4 * std::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                            (density(other_idx) + eps)))/
                          smoothing_kernel_sizes(other_idx)
                        );
                    fij = (fi + fj)/2.;
                  }

                  accel(p_idx) -= mass(other_idx) * sym_grad_W *
                            visc(c_ij_bar, density_mean, d, vel, mean_h) * fij;

                  d_entropy(p_idx) += -((Adiabatic_index-1.0)/std::pow(density(p_idx)+eps,
                              Adiabatic_index -1.0)) * (0.5*mass(other_idx)*
                      visc(c_ij_bar, density_mean, d, vel, mean_h) * fij *
                        vel.dot(sym_grad_W));
                } 
              }
            },
          position, velocity, curl_v);
        }
      );
    }

    template <unsigned DIR = 0>
    KOKKOS_INLINE_FUNCTION
    void apply_bcs(const std::size_t p_idx){
      if constexpr(DIR < DIM){
        // Periodic
        if constexpr(PERIODIC[DIR]){
          if(position(p_idx)[DIR] < low_[DIR] + eps || 
             position(p_idx)[DIR] > low_[DIR] + L_[DIR] - eps){
            position(p_idx)[DIR] = low_[DIR] + std::fmod(
                std::fmod(position(p_idx)[DIR], 
                          L_[DIR]) + L_[DIR],
                L_[DIR]) + eps;
          }
        }
        // Else reflective
        else {
          if(position(p_idx)[DIR] < low_[DIR] + eps) {
           // Reflect the position
           position(p_idx)[DIR] = low_[DIR] - position(p_idx)[DIR];
           position(p_idx)[DIR] = low_[DIR] + std::fmod(
               std::fmod(position(p_idx)[DIR], 
                         L_[DIR]) + L_[DIR],
               L_[DIR]) + eps;
           // Reverse the speed in the respective direction
           velocity(p_idx)[DIR] *= -1;
          }
          else if(position(p_idx)[DIR] > low_[DIR] + L_[DIR] - eps) {
           // Reflect the position
           position(p_idx)[DIR] = low_[DIR] + L_[DIR] - position(p_idx)[DIR];
           position(p_idx)[DIR] = low_[DIR] + std::fmod(
               std::fmod(position(p_idx)[DIR], 
                         L_[DIR]) + L_[DIR],
               L_[DIR]) + eps;
           // Reverse the speed in the respective direction
           velocity(p_idx)[DIR] *= -1;
          }
        }
        apply_bcs<DIR + 1>(p_idx);
      }
    }


    inline void compute_dt_h(){
      const std::size_t N_particles = position.size();
      dt = dt_max;
      avgh = 0;

      T* dt_ptr = &dt;
      Kokkos::parallel_reduce(N_particles, 
        KOKKOS_LAMBDA (const std::size_t& p_idx, T& sum){
          const T loc_dt = CFL * smoothing_kernel_sizes(p_idx)/
                Kokkos::sqrt(Adiabatic_index*pressure(p_idx)/
                              (density(p_idx) + eps));
          Kokkos::atomic_min(dt_ptr, loc_dt);
          sum += smoothing_kernel_sizes(p_idx);
        },
      avgh);

      avgh /= N_particles;
    }

    void step(){
      // Compute time-step size
      compute_dt_h();
      // Adjust h if necessary and adjust the ChainingMesh
      if(avgh > 2 * h){
        h = avgh*1.01;
        CM = ChainingMesh<T, DIM, PERIODIC>(low_, L_, K.supp_radius*h);
        updateNeighbors();
        compute_kernels();
      } else if(avgh < h/2){
        h = avgh*1.01;
        CM = ChainingMesh<T, DIM, PERIODIC>(low_, L_, K.supp_radius*h);
        updateNeighbors();
        compute_kernels();
      }
      // Split-step approach with Strang-Splitting
      if constexpr(viscous){
        dt /= 2;
        rk2_step<true, true>();

        dt *= 2;
        verlet_step<false, false>();

        dt /= 2;
        rk2_step<true, true>();

        dt *= 2;
      } // If no viscosity is needed, just use verlet
      else verlet_step();
    }

    template <bool VISC = viscous, bool ONLY_VISC = true>
    inline void rk2_step(){
      const std::size_t N_particles = position.size();

      Kokkos::View<Vec<T, DIM>*> x_n("x", N_particles);
      Kokkos::View<T*> s_n("s", N_particles);
      Kokkos::View<Vec<T, DIM>*> v_n("v", N_particles);
      if constexpr(!ONLY_VISC){
        Kokkos::deep_copy(x_n, position);
      }
      Kokkos::deep_copy(s_n, entropy);
      Kokkos::deep_copy(v_n, velocity);

      smoothen<VISC, ONLY_VISC>();
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Push
          entropy(p_idx) += dt*d_entropy(p_idx);
          velocity(p_idx) += dt*accel(p_idx);
          if constexpr(!ONLY_VISC){
            position(p_idx) += dt*v_n(p_idx);
            // Apply BCs
            apply_bcs(p_idx);
          }
        }
      );
      // Update acceleration
      if constexpr(!ONLY_VISC){
        updateNeighbors(x_n, v_n, s_n);
        compute_kernels();
      }
      smoothen<VISC, ONLY_VISC>();

      // 2nd RK stage
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Update 
          if constexpr(!ONLY_VISC){
            position(p_idx) = 0.5 * x_n(p_idx) + 
                              0.5 * (position(p_idx) + dt*velocity(p_idx));
          }
          entropy(p_idx)  = 0.5 * s_n(p_idx) + 
                            0.5 * (entropy(p_idx) + dt*d_entropy(p_idx));
          velocity(p_idx) = 0.5 * v_n(p_idx) + 
                            0.5 * (velocity(p_idx) + dt*accel(p_idx));
          // Apply BCs
          if constexpr(!ONLY_VISC){
            apply_bcs(p_idx);
          }
        }
      );
      // Position has been changed, update!
      if constexpr(!ONLY_VISC){
        updateNeighbors();
        compute_kernels();
      }
      // smoothen() comes in the next step
    }

    template <bool VISC = viscous, bool ONLY_VISC = false>
    inline void verlet_step() { 
      const std::size_t N_particles = position.size();
      smoothen<VISC, ONLY_VISC>();
      // Kick & Drift
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Update velocity with half of the acceleration
          velocity(p_idx) += accel(p_idx) * dt / 2.;
          // Update position
          position(p_idx) += velocity(p_idx) * dt;
          // Update entropy, only changes if we have viscosity
          if constexpr(VISC)
            entropy(p_idx) += d_entropy(p_idx) * dt / 2.;
          // Apply boundary conditions
          apply_bcs(p_idx);
        }
      );

      // Here we want the acceleration at x_{i + 1}, but
      // in the viscous case that depends on v_{i + 1}, which is 
      // yet to be found

      // Update acceleration
      updateNeighbors();
      compute_kernels();
      smoothen<VISC, ONLY_VISC>();

      // Kick again
      Kokkos::parallel_for(N_particles, 
        KOKKOS_LAMBDA (const std::size_t p_idx){
          // Update entropy, only changes if we have viscosity
          if constexpr(VISC)
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