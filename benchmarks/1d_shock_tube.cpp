#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

#include "Ippl.h"
#include "Expression/IpplExpressions.h"
#include "Types/Vector.h"
#include "Particle/ParticleLayout.h"
#include "Particle/ParticleSpatialLayout.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleAttribBase.h"
#include "Particle/ParticleAttrib.h"
#include "FieldLayout/FieldLayout.h"
#include "Field/Field.h"
#include "Field/BareField.h"
#include "Meshes/CartesianCentering.h"
#include "Particle/ParticleBC.h"
#include "Manager/BaseManager.h"

#include "SPHParticle.hpp"
#include "SPHManager.h"
#include "utils.hpp"

using namespace std;

// Call this function once at the beginning of your program
void initialize_random_number_generator()
{
    srand((unsigned)time(NULL));
}

int main(int argc, char *argv[])
{
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 1;
        array<int, dim> points = {8};
        int pt = 8;
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I); // what is this
        array<bool, dim> isParallel;
        isParallel.fill(true); // Specifies SERIAL, PARALLEL dims

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);
        double dx = 1.0 / 8.0;

        ippl::Vector<double, dim> hx = {dx};
        ippl::Vector<double, dim> origin = {-10.0};
        ippl::Vector<double, dim> extent = {20.0};
        ippl::UniformCartesian<double, dim> mesh(owned, hx, origin);
        ippl::ParticleSpatialLayout<double, dim> myparticlelayout(layout, mesh);

        double h = 0.05;
        double dt = 1e-3;

        Manager<dim> manager(myparticlelayout, origin, extent, dt, h, 1.4);
        // Initial values
        std::vector<ippl::Vector<double, dim>> R_0;
        std::vector<ippl::Vector<double, dim>> v_0;
        std::vector<double> m_0;
        std::vector<double> E_0;
        std::vector<double> S_0; // Entropy

        const double mass = 1.0;
        double temp = 0.0;

        // Initializing random particle positions and velocities within Manager object
        std::random_device rd;                           // Non-deterministic random number generator
        std::mt19937 gen(rd());                          // Mersenne Twister pseudo-random generator, seeded with rd()
        std::uniform_real_distribution<> dist(0.0, 1.0); // Uniform distribution between 0 and 1

        const unsigned N_particles = 1000;
        const unsigned fac = 5;
        for (unsigned i = 0; i < fac * N_particles; ++i)
        {
            double x = dist(gen) / 2.0;
            double v = utils::maxwellBoltzmann(temp, mass);

            R_0.push_back(ippl::Vector<double, 1>(origin[0] + extent[0] * i / ((double)N_particles * fac) * 0.5));
            v_0.push_back(ippl::Vector<double, 1>(v));
            m_0.push_back(mass);
            E_0.push_back(10.0);
            S_0.push_back(1.0);
        }

        double T2 = 0.0;
        std::random_device rd_2;                           // Non-deterministic random number generator
        std::mt19937 gen_2(rd_2());                        // Mersenne Twister pseudo-random generator, seeded with rd()
        std::uniform_real_distribution<> dist_2(0.0, 1.0); // Uniform distribution between 0 and 1

        for (unsigned i = 0; i < N_particles; i++)
        {
            double x = dist_2(gen_2) / 2.0;
            double v = utils::maxwellBoltzmann(T2, mass);
            R_0.push_back(ippl::Vector<double, dim>(extent[0] * i / ((double)N_particles) * 0.5));
            v_0.push_back(ippl::Vector<double, dim>(v));
            m_0.push_back(mass);
            E_0.push_back(10.0);
            S_0.push_back(1.25);
        }

        const bool visc = true;
        manager.pre_run(R_0, v_0, E_0, m_0, S_0);

        std::vector<double> position;
        std::vector<double> pressure;
        std::vector<double> density;
        std::vector<double> velocity;

        // integration loop of the time eovlution
        const unsigned int N_times = 100;
        for (unsigned int i = 0; i < N_times; i++)
        {
            manager.pre_step(visc);

            position.clear();
            pressure.clear();
            density.clear();
            velocity.clear();

            double x_length;
            for (unsigned int k = 0; k < 5000; k++)
            {
                x_length = -5.0 + k * 0.002;
                position.push_back(x_length);
                pressure.push_back(manager.get_pressure(x_length));
                density.push_back(manager.get_density(x_length));
                velocity.push_back(manager.get_velocity(x_length));
            }

            if (i == (N_times - 1))
            {
                std::ofstream file_p, file_rho, file_v;
                if (visc)
                {
                    file_p.open("../results/pressures/pressures_pos_neww_f" + std::to_string(i) + ".dat");
                    file_rho.open("../results/densities/densities_pos_neww_f" + std::to_string(i) + ".dat");
                    file_v.open("../results/velocities/velocities_pos_neww_f" + std::to_string(i) + ".dat");
                }
                else
                {
                    file_p.open("../results/pressures/pressures_pos_no_visc" + std::to_string(i) + ".dat");
                    file_rho.open("../results/densities/densities_pos_no_visc" + std::to_string(i) + ".dat");
                    file_v.open("../results/velocities/velocities_pos_no_visc" + std::to_string(i) + ".dat");
                }

                // Write the current time and positions to the file
                for (unsigned int j = 0; j < position.size(); j++)
                {
                    file_p << ' ' << position[j] << ' ' << pressure[j] << '\n';
                    file_rho << ' ' << position[j] << ' ' << density[j] << '\n';
                    file_v << ' ' << position[j] << ' ' << velocity[j] << '\n';
                }
                file_p.close();
                file_rho.close();
                file_v.close();
            }

            manager.advance();
            cout << "Time step " << i << endl;
        }
    }
    ippl::finalize();

    return 0;
}