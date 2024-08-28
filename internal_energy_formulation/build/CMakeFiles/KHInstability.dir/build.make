# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_SOURCE_DIR = /home/me/Documents/CSP10-SPH/internal_energy_formulation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/me/Documents/CSP10-SPH/internal_energy_formulation/build

# Include any dependencies generated for this target.
include CMakeFiles/KHInstability.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/KHInstability.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/KHInstability.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/KHInstability.dir/flags.make

CMakeFiles/KHInstability.dir/KH_instability.o: CMakeFiles/KHInstability.dir/flags.make
CMakeFiles/KHInstability.dir/KH_instability.o: /home/me/Documents/CSP10-SPH/internal_energy_formulation/KH_instability.cpp
CMakeFiles/KHInstability.dir/KH_instability.o: CMakeFiles/KHInstability.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/me/Documents/CSP10-SPH/internal_energy_formulation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/KHInstability.dir/KH_instability.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/KHInstability.dir/KH_instability.o -MF CMakeFiles/KHInstability.dir/KH_instability.o.d -o CMakeFiles/KHInstability.dir/KH_instability.o -c /home/me/Documents/CSP10-SPH/internal_energy_formulation/KH_instability.cpp

CMakeFiles/KHInstability.dir/KH_instability.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/KHInstability.dir/KH_instability.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/me/Documents/CSP10-SPH/internal_energy_formulation/KH_instability.cpp > CMakeFiles/KHInstability.dir/KH_instability.i

CMakeFiles/KHInstability.dir/KH_instability.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/KHInstability.dir/KH_instability.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/me/Documents/CSP10-SPH/internal_energy_formulation/KH_instability.cpp -o CMakeFiles/KHInstability.dir/KH_instability.s

# Object files for target KHInstability
KHInstability_OBJECTS = \
"CMakeFiles/KHInstability.dir/KH_instability.o"

# External object files for target KHInstability
KHInstability_EXTERNAL_OBJECTS =

KHInstability: CMakeFiles/KHInstability.dir/KH_instability.o
KHInstability: CMakeFiles/KHInstability.dir/build.make
KHInstability: /usr/lib64/libsfml-graphics.so.2.6.0
KHInstability: /usr/lib64/libsfml-network.so.2.6.0
KHInstability: /usr/lib64/libsfml-audio.so.2.6.0
KHInstability: /usr/lib64/libkokkoscontainers.so.4.0.1
KHInstability: /usr/lib64/libkokkoscore.so.4.0.1
KHInstability: /usr/lib/gcc/x86_64-pc-linux-gnu/13/libgomp.so
KHInstability: /usr/lib64/libpthread.a
KHInstability: /usr/lib64/libhwloc.so
KHInstability: /usr/lib64/libkokkossimd.so.4.0.1
KHInstability: /usr/lib64/libsfml-window.so.2.6.0
KHInstability: /usr/lib64/libsfml-system.so.2.6.0
KHInstability: CMakeFiles/KHInstability.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/me/Documents/CSP10-SPH/internal_energy_formulation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable KHInstability"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KHInstability.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/KHInstability.dir/build: KHInstability
.PHONY : CMakeFiles/KHInstability.dir/build

CMakeFiles/KHInstability.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/KHInstability.dir/cmake_clean.cmake
.PHONY : CMakeFiles/KHInstability.dir/clean

CMakeFiles/KHInstability.dir/depend:
	cd /home/me/Documents/CSP10-SPH/internal_energy_formulation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/me/Documents/CSP10-SPH/internal_energy_formulation /home/me/Documents/CSP10-SPH/internal_energy_formulation /home/me/Documents/CSP10-SPH/internal_energy_formulation/build /home/me/Documents/CSP10-SPH/internal_energy_formulation/build /home/me/Documents/CSP10-SPH/internal_energy_formulation/build/CMakeFiles/KHInstability.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/KHInstability.dir/depend

