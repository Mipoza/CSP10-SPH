project(HelloIPPL CXX)

add_compile_options(-fopenmp -ggdb)
set (CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos/Kokkos_serial/include/)
file(GLOB KOKOS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos/Kokkos_serial/lib/*.a)
link_directories(ippl ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_executable(hello HelloIPPL.cpp)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Recursively find all header files (.hpp and .h) and add their directories to the include path
file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.h")
foreach(HEADER ${HEADER_FILES})
    get_filename_component(HEADER_PATH ${HEADER} PATH)
    include_directories(${HEADER_PATH})
endforeach()

target_link_libraries(hello OpenMP::OpenMP_CXX MPI::MPI_CXX ippl ${KOKOS_LIB})
