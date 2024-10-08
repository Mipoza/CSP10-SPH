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
#define MIN(x, y) (x < y ? x : y)
#define eps (8*std::numeric_limits<T>::epsilon())

template <unsigned DIM>
constexpr auto SPHERE_VOL_FAC(){
    return Kokkos::pow(M_PI, DIM/2.)/std::tgamma(DIM/2. + 1);
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
  Kokkos::View<T*> mass, density, pressure, pressure_dh, n, n_dh, 
                   uint, du, smoothing_kernel_sizes,
                   div_v, dt_loc;
  Kokkos::View<Vec<T, DIM>*> position, velocity, accel, curl_v;

  // Constructor
  SPHManager(const Vec<T, DIM>& low, 
             const Vec<T, DIM>& extent, 
             T CFL_, T h_, 
             T Adiabatic_index_,
             T dt_max_,
             T alpha_ = 0,
             T beta_ = 0) : 
    h(h_), dt_max(dt_max_),
    CM(low, extent, K.supp_radius*h_), 
    Adiabatic_index(Adiabatic_index_), 
    L_(extent), low_(low),
    alpha(alpha_), beta(beta_),
    CFL(CFL_), avgh(h_) {}

  /* Helper functions */

  // (Neumann and Richtmeyer) Viscosity factor
  KOKKOS_INLINE_FUNCTION
  std::pair<T, T> visc(const T& c, const T& density, 
                       const Vec<T, DIM>& dis, const Vec<T, DIM>& vel,
                       const T& h) const {
    const T rij = Kokkos::sqrt(dis.dot(dis));
    const T dot_product = dis.dot(vel);
    const T mu = h*dot_product/(Kokkos::pow(rij, 2) + 0.01*Kokkos::pow(h, 2));

    return std::make_pair(mu, dot_product < 0 ? 
        (-(alpha * c * mu) + (beta * mu * mu)) / (density + eps) : 0);
  }

  // Distance between two points, possibly with 
  // periodic boundary conditions
  template <unsigned DIR = 0>
  __attribute__((always_inline))
  T dist2(const Vec<T, DIM>& x, const Vec<T, DIM>& y, T&& res = 0) const {
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
    return std::move(res);
  }

  KOKKOS_INLINE_FUNCTION
  T dist(const Vec<T, DIM>& x, const Vec<T, DIM>& y) const {
    return Kokkos::sqrt(dist2(x, y));
  }

  template <unsigned DIR = 0>
  KOKKOS_INLINE_FUNCTION
  void dist_vec_dir(const Vec<T, DIM>& x, const Vec<T, DIM>& y,
                    Vec<T, DIM>& res) const {
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
  Vec<T, DIM> dist_vec(const Vec<T, DIM>& x, const Vec<T, DIM>& y) const {
    Vec<T, DIM> res;
    res = 0;
    dist_vec_dir(x, y, res);
    return std::move(res);
  }

  // For debugging
  std::size_t count_neighbors(const std::size_t p_idx) const {
    std::size_t count = 0;
    CM.it_over_neighbors(position(p_idx), 
       K.supp_radius*smoothing_kernel_sizes(p_idx), 
      [&] (const std::size_t other_idx,
           const Vec<T, DIM>& other_pos){
        //duplicate += (p_idx == other_idx);
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
    Kokkos::View<T*> density_("Density", N);
    density = density_;
    Kokkos::View<T*> pressure_("Pressure", N);
    pressure = pressure_;
    Kokkos::View<T*> pressure_dh_("Pressure_dh", N);
    pressure_dh = pressure_dh_;
    Kokkos::View<T*> uint_("u", N);
    uint = uint_;
    Kokkos::View<T*> du_("DUint", N);
    du  = du_;
    Kokkos::View<T*> n_("n", N);
    n  = n_;
    Kokkos::View<T*> n_dh_("n_dh", N);
    n_dh  = n_dh_;
    Kokkos::View<T*> div_v_("div_v", N);
    div_v = div_v_;
    Kokkos::View<T*> smoothing_kernel_sizes_("Smoothing Kernel Sizes", N);
    smoothing_kernel_sizes = smoothing_kernel_sizes_;
    // Set all smoothing kernel sizes to h
    Kokkos::deep_copy(smoothing_kernel_sizes, h);
    Kokkos::View<T*> dt_loc_("dt_loc", N);
    dt_loc = dt_loc_;
    Kokkos::deep_copy(dt_loc, dt_max);

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
            std::vector<T> uint_part,
            T n_target_ = -1) {
    unsigned N_new = R_part.size();
    create_particles(N_new);

    auto R_host = Kokkos::create_mirror_view(position);
    auto v_host = Kokkos::create_mirror_view(velocity);
    auto m_host = Kokkos::create_mirror_view(mass);
    auto uint_host = Kokkos::create_mirror_view(uint);

    for (unsigned int i = 0; i < N_new; ++i) {
        R_host(i) = R_part[i];
        v_host(i) = v_part[i];
        m_host(i) = m_part[i];
        uint_host(i) = uint_part[i];
    }

    Kokkos::deep_copy(position, R_host);
    Kokkos::deep_copy(velocity, v_host);
    Kokkos::deep_copy(mass, m_host);
    Kokkos::deep_copy(uint, uint_host);
    // Set it to something
    if(n_target_< 0)
      n_target = K(0., h) * 
                    Kokkos::pow(h*K.supp_radius, DIM)
                    * (-n_target_);
    else 
      n_target = n_target_;

    // Compute kernels right away, otherwise it would have to be done
    // before the first step
    updateNeighbors();
    compute_kernels();
  }


  // Update the ChainingMesh
  template <bool SORT_ONLY_NON_SMOOTHED = false, typename... ARGS>
  void updateNeighbors(ARGS&... optional_sort){
    CM.partition(position);
    // Sort for data locality
    // NOTE: position is overwritten at the end
    // no other way, bc we would get race-conditions otherwise
    if constexpr(SORT_ONLY_NON_SMOOTHED){
      CM.sort(position, mass, smoothing_kernel_sizes, 
                        uint, velocity, position,
                        optional_sort...);
    }
    else {
      if constexpr(viscous && balsara){
        CM.sort(position, mass, n, n_dh, smoothing_kernel_sizes, 
                          density, pressure, pressure_dh, dt_loc,
                          uint, du, velocity, accel, div_v, curl_v,
                          position,
                          optional_sort...);
      }
      else { 
        CM.sort(position, mass, n, n_dh, smoothing_kernel_sizes, 
                          density, pressure, pressure_dh, dt_loc,
                          uint, du, velocity, accel,
                          position,
                          optional_sort...);
      }
    }
  }

  // Compute the "number density"
  KOKKOS_INLINE_FUNCTION T __attribute__((always_inline))
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
    return std::move(n * Kokkos::pow(h_loc*h*K.supp_radius, DIM));
  }

  void compute_kernels(){
    const std::size_t N_particles = position.size();
    // Find the smoothing kernel sizes
    Kokkos::parallel_for(N_particles, 
      KOKKOS_LAMBDA (const std::size_t p_idx){
        // First: compute smoothing kernel size
        // Helper function
        auto my_mass_func = [&](const T h_) {
          return compute_n_density(p_idx, h_) - n_target;
        };
        // Initial values for smoothing kernel sizes
        const T h_current = smoothing_kernel_sizes(p_idx)/h;
        const T current_n = compute_n_density(p_idx, h_current);
        T h0, h1;
        T fac = 1;
        if(current_n > n_target){
          // Too much mass, give a slope towards smaller h
          h1 = h_current;
          h0 = h_current * n_target/(current_n + eps);
          fac = illinois(h0, h1, my_mass_func);
        } else if(current_n < n_target){
          // Opposite, give a slope which goes towards bigger h
          h1 = MIN(h_current * n_target/(current_n + eps),
                        2 * h_current);
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
        pressure(p_idx) = 0;
        if constexpr(!ONLY_VISC){
          pressure_dh(p_idx) = 0;
          n(p_idx) = 0;
          n_dh(p_idx) = 0;
        }
        T volume_element = 0;
        if constexpr(VISC && balsara){
          div_v(p_idx) = 0;
          curl_v(p_idx) = 0;
        }
        // Loop over neighbor cells
        CM.it_over_neighbors(position(p_idx),
           K.supp_radius*smoothing_kernel_sizes(p_idx), 
          [&] (const std::size_t other_idx, 
               const Vec<T, DIM>& other_pos,
               const Vec<T, DIM>& other_vel){
            const Vec<T, DIM> d = dist_vec(position(p_idx), other_pos);
            const T rij = Kokkos::sqrt(d.dot(d));
            const T W_ij = K(rij, smoothing_kernel_sizes(p_idx));
            const T W_ij_dh = K.grad_h(rij, smoothing_kernel_sizes(p_idx));

            pressure(p_idx) += (Adiabatic_index - 1)*mass(other_idx) * 
                               uint(other_idx)*W_ij;

            if constexpr(!ONLY_VISC){
              pressure_dh(p_idx) += (Adiabatic_index - 1)*mass(other_idx) * 
                                    uint(other_idx)*W_ij_dh;
              n(p_idx) += W_ij;
              n_dh(p_idx) += W_ij_dh;
            }

            volume_element += (Adiabatic_index - 1) * 
                              mass(other_idx) * uint(other_idx) * W_ij;

            if constexpr(VISC && balsara){
              if(rij > eps){
                const Vec<T, DIM> vji = other_vel - velocity(p_idx);
                div_v(p_idx) += mass(other_idx)* 
                                vji.dot(
                                  (d/(rij + eps)
                                  * K.grad_r(rij, smoothing_kernel_sizes(p_idx)))
                                 );
                // Dimension 2: Rotate v by 90 deg and dot with grad W
                if constexpr(DIM == 2){
                  Vec<T, DIM> vij_rot;
                  vij_rot[0] =  vji[1];
                  vij_rot[1] = -vji[0];

                  // Choose one component
                  // NOT ACTUALLY the vector curl, but it's only one number anyway
                  curl_v(p_idx)[0] += mass(other_idx) *
                                        vij_rot.dot(
                                          (d/(rij + eps)
                                          * K.grad_r(rij, smoothing_kernel_sizes(p_idx)))
                                        );
                }
              } 
            }
          },
        position, velocity);

        volume_element = (Adiabatic_index - 1) * 
                         mass(p_idx) * uint(p_idx) / (volume_element + eps);
        density(p_idx) = mass(p_idx)/volume_element;

        if constexpr(VISC && balsara){
          div_v(p_idx) /= (density(p_idx) + eps);
          curl_v(p_idx) /= (density(p_idx) + eps);
        }
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
        du(p_idx) = 0.0;
        const T fij_fac = 1./(1 + 
            smoothing_kernel_sizes(p_idx)*n_dh(p_idx)/(DIM * n(p_idx)));
        const T ci = Kokkos::sqrt(abs(Adiabatic_index*pressure(p_idx)/
                      (density(p_idx) + eps)));
        dt_loc(p_idx) = dt_max;
        // Loop over neighbor cells
        CM.it_over_neighbors(position(p_idx), 
           K.supp_radius*smoothing_kernel_sizes(p_idx), 
          [&] (const std::size_t other_idx,
               const Vec<T, DIM>& other_pos,
               const Vec<T, DIM>& other_vel,
               const Vec<T, DIM>& other_curl){
            const Vec<T, DIM> d = dist_vec(other_pos, position(p_idx));
            const T rij = d.norm();
            if(rij > eps){
              const Vec<T, DIM> dv = velocity(p_idx) - other_vel;

              // We have an option to ignore the pressure-acceleration,
              // since we want to apply a split-step method
              if constexpr(!ONLY_VISC){
                const T fij = (1 - (smoothing_kernel_sizes(p_idx) * pressure_dh(p_idx)/
                    ( eps + 
                      DIM * (Adiabatic_index - 1) * n(p_idx) 
                      * mass(other_idx) * uint(other_idx)
                    ))*fij_fac)/(pressure(p_idx) + eps);

                const T fji_fac = 1./(1 + 
                    smoothing_kernel_sizes(other_idx)*n_dh(other_idx)/(
                      DIM * n(other_idx) + eps));
                const T fji = (1 - (smoothing_kernel_sizes(other_idx) * 
                              pressure_dh(other_idx)/
                    ( eps + 
                      DIM * (Adiabatic_index - 1) * n(other_idx) 
                      * mass(p_idx) * uint(p_idx)
                    ))*fji_fac)/(pressure(other_idx) + eps);

                accel(p_idx) -= Kokkos::pow(Adiabatic_index - 1, 2) * mass(other_idx) *
                                uint(p_idx) * uint(other_idx) * (
                    fij * (((d)*
                        K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps)) +
                    fji * (((d)*
                        K.grad_r(rij, smoothing_kernel_sizes(other_idx)))/(rij + eps))
                    );

                du(p_idx) += Kokkos::pow(Adiabatic_index - 1, 2) * mass(other_idx) *
                                uint(p_idx) * uint(other_idx) * (
                    fij * dv.dot(((d)*
                        K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps))
                    );
              }
              // We need mu_ij for the CFL condition
              T mu_ij = 0;

              if constexpr (VISC) {
                const T density_mean = (density(other_idx) + density(p_idx))/2.;
                // Mean sound velocity
                const T c_ij_bar = (ci +
                    Kokkos::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                        (density(other_idx) + eps))))/2.;

                const T mean_h = 0.5*(smoothing_kernel_sizes(p_idx) +
                                      smoothing_kernel_sizes(other_idx));
                const Vec<T, DIM> sym_grad_W = 0.5 *(
                    (((d)*K.grad_r(rij, smoothing_kernel_sizes(p_idx)))/(rij + eps))
                   + (((d)*K.grad_r(rij, 
                       smoothing_kernel_sizes(other_idx)))/(rij + eps)));

                T fij_balsara = 1;
                if constexpr(balsara){
                  // Balsara factor
                  const T fi = Kokkos::abs(div_v(p_idx))/(
                      Kokkos::abs(div_v(p_idx)) + curl_v(p_idx).norm() + // c_i/h_i 
                      1e-4 * ci/smoothing_kernel_sizes(p_idx)
                      );
                  const T fj = Kokkos::abs(div_v(other_idx))/(
                      Kokkos::abs(div_v(other_idx)) + other_curl.norm() + // c_j/h_j 
                      1e-4 * Kokkos::sqrt(abs(Adiabatic_index*pressure(other_idx)/
                          (density(other_idx) + eps)))/
                        smoothing_kernel_sizes(other_idx)
                      );
                  fij_balsara = (fi + fj)/2.;
                }
                const auto v_fac = visc(c_ij_bar, density_mean, d, dv, mean_h);
                mu_ij = v_fac.first;
                const T pi_ij = v_fac.second * fij_balsara;
                accel(p_idx) -= mass(other_idx) * sym_grad_W * pi_ij;

                du(p_idx) += 0.5 * mass(other_idx) * pi_ij *
                            dv.dot(((d)*K.grad_r(rij, 
                              smoothing_kernel_sizes(other_idx)))/(rij + eps));
              }
              // Compute maximal time-step
              dt_loc(p_idx) = MIN(dt_loc(p_idx),
                  static_cast<T>(
                    (0.5 * smoothing_kernel_sizes(p_idx) * K.supp_radius /(
                    eps + (1 + 0.6 * alpha) * ci + 0.6 * beta * Kokkos::abs(MIN(
                                                          static_cast<T>(0), mu_ij)))
                  )
                )
              );
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
    dt = CFL * (*Kokkos::Experimental::min_element(Kokkos::DefaultExecutionSpace(),
                                                   dt_loc));
    avgh = 0;

    Kokkos::parallel_reduce(N_particles, 
      KOKKOS_LAMBDA (const std::size_t& p_idx, T& sum){
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
    Kokkos::View<T*> u_n("u", N_particles);
    Kokkos::View<Vec<T, DIM>*> v_n("v", N_particles);
    if constexpr(!ONLY_VISC){
      Kokkos::deep_copy(x_n, position);
    }
    Kokkos::deep_copy(u_n, uint);
    Kokkos::deep_copy(v_n, velocity);

    smoothen<VISC, ONLY_VISC>();
    Kokkos::parallel_for(N_particles, 
      KOKKOS_LAMBDA (const std::size_t p_idx){
        // Push
        uint(p_idx) += dt*du(p_idx);
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
      updateNeighbors(x_n, v_n, u_n);
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
        uint(p_idx)  = 0.5 * u_n(p_idx) + 
                       0.5 * (uint(p_idx) + dt*du(p_idx));
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

  template <bool VISC = viscous, bool ONLY_VISC_SMOOTH = false>
  inline void verlet_step() { 
    const std::size_t N_particles = position.size();
    smoothen<VISC, ONLY_VISC_SMOOTH>();
    // Kick & Drift
    Kokkos::parallel_for(N_particles, 
      KOKKOS_LAMBDA (const std::size_t p_idx){
        // Update velocity with half of the acceleration
        velocity(p_idx) += accel(p_idx) * dt / 2.;
        // Update position
        position(p_idx) += velocity(p_idx) * dt;
        // Update uint, only changes if we have viscosity
        uint(p_idx) += du(p_idx) * dt / 2.;
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
    smoothen<VISC, ONLY_VISC_SMOOTH>();

    // Kick again
    Kokkos::parallel_for(N_particles, 
      KOKKOS_LAMBDA (const std::size_t p_idx){
        // Update uint, only changes if we have viscosity
        uint(p_idx) += du(p_idx) * dt / 2.;
        // Update velocity with the other half of the acceleration
        velocity(p_idx) += accel(p_idx) * dt / 2.;
      }
    );
  }
};
