# cmake_minimum_required(VERSION 3.0)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED N)
# set(CMAKE_BUILD_TYPE Debug)

# set (IPPL_VERSION_MAJOR 3)
# set (IPPL_VERSION_MINOR 0.1)

# set (IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")

# set (IPPL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# project(IPPL_Project)

# if(CMAKE_COMPILER_IS_GNUCXX)
#     add_compile_options(-w)
# elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     add_compile_options(-Wno-everything)
# endif()

# # Set the source files for your project
# set(SOURCES gravity_ippl.cpp)

# # Find MPI package
# find_package(MPI REQUIRED)


# # Set the directory containing Kokkos_Core.hpp
# set(KOKKOS_INCLUDE_DIR "/cluster/home/atamargo/repeat_ippl_2/Kokkos/Kokkos_serial/include")

# file(GLOB KOKKOS_LIBRARIES "/cluster/home/atamargo/repeat_ippl_2/Kokkos/Kokkos_serial/lib64/*.a")

# # Set the directory containing IPPL libraries
# set(IPPL_LIBRARY_DIR "/cluster/home/atamargo/ippl_Multi_build/lib/libippl.a")

# # Add the directory containing Kokkos_Core.hpp to the include path
# include_directories(${KOKKOS_INCLUDE_DIR})

# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()


# add_definitions(-DIPPL_ENABLE_TIMER_FENCES=true)

# # Recursively find all header files (.hpp and .h) and add their directories to the include path
# file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.h")
# foreach(HEADER ${HEADER_FILES})
#     get_filename_component(HEADER_PATH ${HEADER} PATH)
#     include_directories(${HEADER_PATH})
# endforeach()

# # Create the executable target
# add_executable(myprogram_IPPL_TRY ${SOURCES})
# target_link_libraries(myprogram_IPPL_TRY ${MPI_CXX_LIBRARIES} ${IPPL_LIBRARY_DIR})
# target_link_libraries(myprogram_IPPL_TRY ${MPI_CXX_LIBRARIES} ${KOKKOS_LIBRARIES} dl)


cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED N)
set(CMAKE_BUILD_TYPE Debug)

set (IPPL_VERSION_MAJOR 3)
set (IPPL_VERSION_MINOR 0.1)

set (IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")

set (IPPL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

project(IPPL_Project)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-w)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-everything)
endif()

# Set the source files for your project
set(SOURCES 1d_shock_tube.cpp)

# Find MPI package
find_package(MPI REQUIRED)

add_compile_options(-Ofast)

# Set the directory containing Kokkos_Core.hpp
set(KOKKOS_INCLUDE_DIR "/cluster/home/atamargo/repeat_ippl_2/Kokkos/Kokkos_serial/include")

file(GLOB KOKKOS_LIBRARIES "/cluster/home/atamargo/repeat_ippl_2/Kokkos/Kokkos_serial/lib64/*.a")

# Set the directory containing IPPL libraries
set(IPPL_LIBRARY_DIR "/cluster/home/atamargo/ippl_Multi_build/lib/libippl.a")

# Add the directory containing Kokkos_Core.hpp to the include path
include_directories(${KOKKOS_INCLUDE_DIR})

# Set the Heffte directory
set(Heffte_DIR "/cluster/home/atamargo/repeat_ippl_2/Heffte/Heffte_serial/lib/cmake/Heffte")

# Find the HeFFTe package
find_package(Heffte REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_definitions(-DIPPL_ENABLE_TIMER_FENCES=true)

include_directories("/cluster/home/atamargo/repeat_ippl_2/include")

# Recursively find all header files (.hpp and .h) and add their directories to the include path
file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.h")
foreach(HEADER ${HEADER_FILES})
    get_filename_component(HEADER_PATH ${HEADER} PATH)
    include_directories(${HEADER_PATH})
endforeach()

# Create the executable target
add_executable(myprogram_IPPL_TRY ${SOURCES})
target_link_libraries(myprogram_IPPL_TRY ${MPI_CXX_LIBRARIES} ${IPPL_LIBRARY_DIR})
target_link_libraries(myprogram_IPPL_TRY ${MPI_CXX_LIBRARIES} ${KOKKOS_LIBRARIES} dl)

# Link against HeFFTe
target_link_libraries(myprogram_IPPL_TRY Heffte::Heffte)


