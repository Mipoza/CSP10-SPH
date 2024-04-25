#include <algorithm>
const char* TestName = "IPPL_Test";
 
#include "../include/Ippl.h"
#include "SPHManager.hpp"
// #include "SPHParticle.hpp"
 
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);
 
        msg << "Hello World" << endl;
        using p_type = SPHParticle<double, 2>; // default kernel: cubic spline
        typedef ippl::Vector<double, 2> Vector_t;
        typedef ippl::UniformCartesian<double, 2> Mesh_t;
        typedef ippl::FieldLayout<2> FieldLayout_t;
        typedef ippl::ParticleSpatialLayout<double, 2> PLayout_t;

        ippl::Vector<int, 2> nr;
        nr[0] = 8;
        nr[1] = 8;

        Vector_t rmin(0.0);
        Vector_t rmax(1.0);
        // create mesh and layout objects for this problem domain
        Vector_t hr = rmax / nr;

        ippl::NDIndex<2> domain;
        for (unsigned d = 0; d < 2; d++) {
            domain[d] = ippl::Index(nr[d]);
        }

        std::array<bool, 2> isParallel;
        isParallel.fill(true);

        Vector_t origin = rmin;

        const double dt = 0.5 * hr[0];  // size of timestep

        const bool isAllPeriodic = true;
        Mesh_t mesh(domain, hr, origin);
        FieldLayout_t FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t PL(FL, mesh);

        Vector_t L;
        L[0] = 1;
        L[1] = 1;
        double h_ = hr[0];
        std::shared_ptr<p_type> P;
        P = std::make_shared<p_type>(PL, rmin, L, h_);

        P->create(10000);

        SPHManager<double, 2> M(P, dt);
        M.run(2);

        msg << P->position(1) << endl;
    }
    ippl::finalize();
 
    return 0;
}
