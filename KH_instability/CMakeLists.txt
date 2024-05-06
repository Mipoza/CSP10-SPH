project(HelloIPPL CXX)

add_compile_options(-fopenmp -ggdb)
add_compile_options(-O3)
add_compile_options(-Ofast)

set (CMAKE_CXX_STANDARD 20)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos/Kokkos_serial/include/)

# Finding libraries
file(GLOB KOKOS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos/Kokkos_serial/lib/*.a)
link_directories(ippl ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

# Executable
add_executable(hello KH_instability.cpp)

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
target_link_libraries(hello 
    OpenMP::OpenMP_CXX 
    MPI::MPI_CXX 
    ippl 
    ${KOKOS_LIB}
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-network 
    sfml-audio
)
