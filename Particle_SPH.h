#include <iostream>
#include <cmath>

#include "include/Ippl.h"

#include "include/Expression/IpplExpressions.h" 
//#include "include/Expression/IpplOperators.h" 

#include "include/Types/Vector.h"
#include "include/Particle/ParticleLayout.h"
#include "include/Particle/ParticleSpatialLayout.h"
#include "include/Particle/ParticleBase.h"
#include "include/Particle/ParticleAttribBase.h"
#include "include/Particle/ParticleAttrib.h"
#include "include/FieldLayout/FieldLayout.h"
#include "include/Field/Field.h"
#include "include/Field/BareField.h"
#include "include/Meshes/CartesianCentering.h"

#ifndef PARTICLE_SPH_H
#define PARTICLE_SPH_H


using namespace ippl;
using namespace std;



template<unsigned int Dim> //WARNING: I am not using Dim in ParticleSpatialLayout. If I do, particle_position_type does not work!!!!
class SPH_Particle: public ParticleBase<ParticleSpatialLayout<double,2> > {
public:
  // attributes for this class
  particle_position_type vel;  // velocity, same storage type as R
  particle_position_type acceleration;
  ParticleAttrib<double> mass;
  ParticleAttrib<double> density;
  ParticleAttrib<double> pressure;


  // constructor: add attributes to base class
  SPH_Particle(ParticleSpatialLayout<double,Dim>& L) : ParticleBase(L) 
  {
    this->addAttribute(vel);
    this->addAttribute(mass);
    this->addAttribute(density);
    this->addAttribute(pressure);
    this->addAttribute(acceleration);
  }


  void acceleration_function_external(Vector<double, Dim> (*func)(Vector<double, Dim>))
  {
    //Needs to be changed to getTotalNum() when we have the function?
    int N = this->getLocalNum();

    typename particle_position_type::HostMirror accel_host = (this->acceleration).getHostMirror();
    Vector<double, Dim> R_aux;

    for (unsigned int i = 0; i < N; ++i) 
    {
      R_aux = this->R(i);
      accel_host(i) = func(R_aux);
    }
    
    Kokkos::deep_copy((this->acceleration).getView(), accel_host);

    this->update();
  }

};

#endif