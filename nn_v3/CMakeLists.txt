project(SharedMemoryNearestNeighbor CXX)

find_package(OpenMP REQUIRED)

add_compile_options(-O3)
add_executable(SharedMemoryNearestNeighbor test.cpp)
add_executable(RuntimeTest runtime_test.cpp)

target_link_libraries(SharedMemoryNearestNeighbor OpenMP::OpenMP_CXX)
target_link_libraries(RuntimeTest OpenMP::OpenMP_CXX)
