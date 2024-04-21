project(HelloIPPL CXX)

add_compile_options(-fopenmp -ggdb)
set (CMAKE_CXX_STANDARD 20)

include_directories(inc ${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_directories(ippl ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_executable(hello HelloIPPL.cpp)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)

target_link_libraries(hello OpenMP::OpenMP_CXX MPI::MPI_CXX Kokkos::kokkos ippl)
