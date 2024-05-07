project(SPH_IPPL CXX)

add_compile_options(-O3)

set (CMAKE_CXX_STANDARD 20)
add_compile_options (-Wno-deprecated-declarations -w)

# TODO: If using the OpenMP version of IPPL, adjust
# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos/Kokkos_serial/include/)

# Finding libraries
file(GLOB KOKOS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos/Kokkos_serial/lib/*.a)
link_directories(ippl ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

# Executable
add_executable(KHInstability KH_instability.cpp)
add_executable(ShockTube 1d_shock_tube.cpp)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)

# Recursively find all header files and add their directories to the include path
file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.h")
foreach(HEADER ${HEADER_FILES})
    get_filename_component(HEADER_PATH ${HEADER} PATH)
    include_directories(${HEADER_PATH})
endforeach()

# Linking libraries
target_link_libraries(KHInstability 
    OpenMP::OpenMP_CXX 
    ${KOKOS_LIB}
    MPI::MPI_CXX 
    ippl 
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-network 
    sfml-audio
)

target_link_libraries(ShockTube 
    OpenMP::OpenMP_CXX 
    ${KOKOS_LIB}
    MPI::MPI_CXX 
    ippl 
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-network 
    sfml-audio
)
