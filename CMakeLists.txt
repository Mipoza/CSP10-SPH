cmake_minimum_required(VERSION 3.28)

project(HelloIPPL CXX)

# Compiler options
add_compile_options(-O3 -fopenmp)
set(CMAKE_CXX_STANDARD 20)

# Include IPPL directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ippl/build/include)

# Add IPPL library path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ippl/build/lib)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)

# Recursively find all header files and add their directories to the include path
file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.h")
foreach(HEADER ${HEADER_FILES})
    get_filename_component(HEADER_PATH ${HEADER} PATH)
    include_directories(${HEADER_PATH})
endforeach()

# Add the executable target
add_executable(1d_shock_tube benchmarks/1d_shock_tube.cpp)
add_executable(Kelvin_Helmholtz benchmarks/KH_instability.cpp)

set(REQUIRED_LIBS
    OpenMP::OpenMP_CXX 
    MPI::MPI_CXX 
    ippl 
    Kokkos::kokkos
)

set(SFML_LIBS
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-network 
    sfml-audio
)

# Linking libraries
target_link_libraries(1d_shock_tube ${REQUIRED_LIBS})
target_link_libraries(Kelvin_Helmholtz ${REQUIRED_LIBS} ${SFML_LIBS})
