project(SharedMemoryNearestNeighbor CXX)

find_package(OpenMP REQUIRED)

add_compile_options(-Ofast)
add_executable(SharedMemoryNearestNeighbor test.cpp)
target_link_libraries(SharedMemoryNearestNeighbor OpenMP::OpenMP_CXX)
