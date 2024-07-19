project(SharedMemoryNearestNeighbor CXX)

set(CXX_STANARD 23)

set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)

# add_compile_options(-ggdb)
add_compile_options(-ggdb -Wall)
add_executable(NN_test test.cpp)
# add_executable(RuntimeTest runtime_test.cpp)

target_link_libraries(NN_test OpenMP::OpenMP_CXX Kokkos::kokkos)
# target_link_libraries(RuntimeTest OpenMP::OpenMP_CXX Kokkos::kokkos)
