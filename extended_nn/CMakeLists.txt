project(SPH CXX)

add_compile_options(-Wall)

set (CMAKE_CXX_STANDARD 23)

# Executable
add_executable(KHInstability KH_instability.cpp)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)
find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)

# Linking libraries
target_link_libraries(KHInstability 
    OpenMP::OpenMP_CXX 
    Kokkos::kokkos
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-network 
    sfml-audio
)
