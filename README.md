# Smoothed Particle Hyrdodynamics with IPPL
## IPPL
This library must be used with IPPL. The CMake assumes that the ```Test_cases``` folder is in the same directory as the IPPL installation 
and that Kokkos has been installed in serial (alongsied IPPL). When compiling with OpenMP, change the CMake to use ``Kokkos_openmp`` instead of the serial version.
## SFML
For visualizing the particles, a graphics library named SFML is used. Make sure that it is installed on your system.
