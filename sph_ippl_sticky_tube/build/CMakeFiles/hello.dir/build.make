# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build

# Include any dependencies generated for this target.
include CMakeFiles/hello.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hello.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hello.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hello.dir/flags.make

CMakeFiles/hello.dir/Manager_implementation.o: CMakeFiles/hello.dir/flags.make
CMakeFiles/hello.dir/Manager_implementation.o: /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/Manager_implementation.cpp
CMakeFiles/hello.dir/Manager_implementation.o: CMakeFiles/hello.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hello.dir/Manager_implementation.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hello.dir/Manager_implementation.o -MF CMakeFiles/hello.dir/Manager_implementation.o.d -o CMakeFiles/hello.dir/Manager_implementation.o -c /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/Manager_implementation.cpp

CMakeFiles/hello.dir/Manager_implementation.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/hello.dir/Manager_implementation.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/Manager_implementation.cpp > CMakeFiles/hello.dir/Manager_implementation.i

CMakeFiles/hello.dir/Manager_implementation.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/hello.dir/Manager_implementation.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/Manager_implementation.cpp -o CMakeFiles/hello.dir/Manager_implementation.s

# Object files for target hello
hello_OBJECTS = \
"CMakeFiles/hello.dir/Manager_implementation.o"

# External object files for target hello
hello_EXTERNAL_OBJECTS =

hello: CMakeFiles/hello.dir/Manager_implementation.o
hello: CMakeFiles/hello.dir/build.make
hello: /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/../Kokkos/Kokkos_serial/lib/libkokkoscontainers.a
hello: /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/../Kokkos/Kokkos_serial/lib/libkokkoscore.a
hello: /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/../Kokkos/Kokkos_serial/lib/libkokkossimd.a
hello: /usr/lib/libsfml-graphics.so.2.6.1
hello: /usr/lib/libsfml-network.so.2.6.1
hello: /usr/lib/libsfml-audio.so.2.6.1
hello: /usr/lib/libgomp.so
hello: /usr/lib/libpthread.a
hello: /usr/lib/libmpi.so
hello: /usr/lib/libsfml-window.so.2.6.1
hello: /usr/lib/libsfml-system.so.2.6.1
hello: CMakeFiles/hello.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hello"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hello.dir/build: hello
.PHONY : CMakeFiles/hello.dir/build

CMakeFiles/hello.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hello.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hello.dir/clean

CMakeFiles/hello.dir/depend:
	cd /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build /home/rdabetic/Documents/IPPL/sph_ippl_sticky_grid/build/CMakeFiles/hello.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/hello.dir/depend

