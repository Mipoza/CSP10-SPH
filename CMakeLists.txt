cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED N)
set(CMAKE_BUILD_TYPE Debug)

set (IPPL_VERSION_MAJOR 3)
set (IPPL_VERSION_MINOR 0.1)

set (IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")

set (IPPL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

project(IPPL_Project)

# Set the source files for your project
set(SOURCES Manager_implementation.cpp)


# Set the path to MPI
set(MPI_PATH "/opt/homebrew/Cellar/open-mpi/5.0.3/include")  # Replace with your actual MPI path

# Add the include directory for MPI
include_directories(${MPI_PATH})

# Find MPI package
#find_package(MPI REQUIRED)

# Set the directory containing Kokkos_Core.hpp
set(KOKKOS_INCLUDE_DIR "/Users/andrestamargo/Desktop/SPH_repository/CSP10-SPH/IPPL_files/kokkos/Kokkos_serial/include")

file(GLOB KOKKOS_LIBRARIES "/Users/andrestamargo/Desktop/SPH_repository/CSP10-SPH/IPPL_files/kokkos/Kokkos_serial/lib/*.a")

# Set the directory containing IPPL libraries
set(IPPL_LIBRARY_DIR "/Users/andrestamargo/Desktop/SPH_repository/CSP10-SPH/IPPL_files/lib/libippl.a")

# Add the directory containing Kokkos_Core.hpp to the include path
include_directories(${KOKKOS_INCLUDE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_definitions(-DIPPL_ENABLE_TIMER_FENCES=true)

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