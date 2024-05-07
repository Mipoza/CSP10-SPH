# Smoothed Particle Hydrodynamics with IPPL

## Project Branches
We used several branches to work on the project more comfortably, but it is in the `main` branch where the final codes are located within the test folder. The `Test_cases` folder are old codes and are not intended to be used. In particular, we created four branches for each contributor as follows:

- `atamargo`: Andres Tamargo. Github alias: andrest2000
- `rdabetic`: Radovan Dabeti\'c. Github alias: Rado
- `vincent`: Vincent Boulard. Github alias: Mipoza
- `marc`: Marc de Miguel. Github alias: Marc deMiguelComella

## IPPL

This library must be used with IPPL. The CMake configuration assumes that the `test` folder is in the same directory as the IPPL installation, and that Kokkos has been installed in serial mode (alongside IPPL). When compiling with OpenMP, change the CMake configuration to use `Kokkos_openmp` instead of the serial version.

## SFML

For visualizing the particles, a graphics library named SFML is used. Make sure that it is installed on your system.

## Contributions

It's important to note that we all helped each other and worked together as a team. For example, sometimes we all worked simultaneously on the same feature, so it's really important for us that this is recognized as a collective contribution and not a sum of individual ones.

### Radovan 
I was in charge of handling the nearest-neighbor problem. I wrote several versions using the standard library and tried using Kokkos in the latest version, however, we have discovered some bugs in the that one. Other than that, I contributed by debugging in general.

### Andrés
I wrote the `Manager.h` class and also wrote the 1D Sod Shock tube test code and analyse the results. I also contributed with debugging in general in all codes.

### Vincent
First, I wrote the prototype of our SPH implementation in native C++ (available in `vincent/vanilla_sph`), then I implemented the boundary conditions and the particle drawing. I also wrote the Kelvin-Helmholtz instability test. More broadly, I did a lot of testing and debugging, especially in `Manager.h` and `SPHParticle_radovan.hpp`.

### Marc
Add your contributions here...