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

template<unsigned int Dim>
class SPH_Particle: public ParticleBase<ParticleSpatialLayout<double,3> > {
public:
  // attributes for this class
  particle_position_type vel;  // velocity, same storage type as R

  // constructor: add attributes to base class
  SPH_Particle(ParticleSpatialLayout<double,Dim>& L) : ParticleBase(L) 
  {
    addAttribute(vel);
  }

};

#endif