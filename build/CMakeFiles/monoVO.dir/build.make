# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shubham/workspace/TAPL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shubham/workspace/TAPL/build

# Include any dependencies generated for this target.
include CMakeFiles/monoVO.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/monoVO.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/monoVO.dir/flags.make

CMakeFiles/monoVO.dir/src/matching2D.cpp.o: CMakeFiles/monoVO.dir/flags.make
CMakeFiles/monoVO.dir/src/matching2D.cpp.o: ../src/matching2D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shubham/workspace/TAPL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/monoVO.dir/src/matching2D.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/monoVO.dir/src/matching2D.cpp.o -c /home/shubham/workspace/TAPL/src/matching2D.cpp

CMakeFiles/monoVO.dir/src/matching2D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/monoVO.dir/src/matching2D.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shubham/workspace/TAPL/src/matching2D.cpp > CMakeFiles/monoVO.dir/src/matching2D.cpp.i

CMakeFiles/monoVO.dir/src/matching2D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/monoVO.dir/src/matching2D.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shubham/workspace/TAPL/src/matching2D.cpp -o CMakeFiles/monoVO.dir/src/matching2D.cpp.s

CMakeFiles/monoVO.dir/src/matching2D.cpp.o.requires:

.PHONY : CMakeFiles/monoVO.dir/src/matching2D.cpp.o.requires

CMakeFiles/monoVO.dir/src/matching2D.cpp.o.provides: CMakeFiles/monoVO.dir/src/matching2D.cpp.o.requires
	$(MAKE) -f CMakeFiles/monoVO.dir/build.make CMakeFiles/monoVO.dir/src/matching2D.cpp.o.provides.build
.PHONY : CMakeFiles/monoVO.dir/src/matching2D.cpp.o.provides

CMakeFiles/monoVO.dir/src/matching2D.cpp.o.provides.build: CMakeFiles/monoVO.dir/src/matching2D.cpp.o


CMakeFiles/monoVO.dir/src/cvEngine.cpp.o: CMakeFiles/monoVO.dir/flags.make
CMakeFiles/monoVO.dir/src/cvEngine.cpp.o: ../src/cvEngine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shubham/workspace/TAPL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/monoVO.dir/src/cvEngine.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/monoVO.dir/src/cvEngine.cpp.o -c /home/shubham/workspace/TAPL/src/cvEngine.cpp

CMakeFiles/monoVO.dir/src/cvEngine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/monoVO.dir/src/cvEngine.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shubham/workspace/TAPL/src/cvEngine.cpp > CMakeFiles/monoVO.dir/src/cvEngine.cpp.i

CMakeFiles/monoVO.dir/src/cvEngine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/monoVO.dir/src/cvEngine.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shubham/workspace/TAPL/src/cvEngine.cpp -o CMakeFiles/monoVO.dir/src/cvEngine.cpp.s

CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.requires:

.PHONY : CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.requires

CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.provides: CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.requires
	$(MAKE) -f CMakeFiles/monoVO.dir/build.make CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.provides.build
.PHONY : CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.provides

CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.provides.build: CMakeFiles/monoVO.dir/src/cvEngine.cpp.o


CMakeFiles/monoVO.dir/examples/monoVO.cpp.o: CMakeFiles/monoVO.dir/flags.make
CMakeFiles/monoVO.dir/examples/monoVO.cpp.o: ../examples/monoVO.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shubham/workspace/TAPL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/monoVO.dir/examples/monoVO.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/monoVO.dir/examples/monoVO.cpp.o -c /home/shubham/workspace/TAPL/examples/monoVO.cpp

CMakeFiles/monoVO.dir/examples/monoVO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/monoVO.dir/examples/monoVO.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shubham/workspace/TAPL/examples/monoVO.cpp > CMakeFiles/monoVO.dir/examples/monoVO.cpp.i

CMakeFiles/monoVO.dir/examples/monoVO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/monoVO.dir/examples/monoVO.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shubham/workspace/TAPL/examples/monoVO.cpp -o CMakeFiles/monoVO.dir/examples/monoVO.cpp.s

CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.requires:

.PHONY : CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.requires

CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.provides: CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.requires
	$(MAKE) -f CMakeFiles/monoVO.dir/build.make CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.provides.build
.PHONY : CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.provides

CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.provides.build: CMakeFiles/monoVO.dir/examples/monoVO.cpp.o


# Object files for target monoVO
monoVO_OBJECTS = \
"CMakeFiles/monoVO.dir/src/matching2D.cpp.o" \
"CMakeFiles/monoVO.dir/src/cvEngine.cpp.o" \
"CMakeFiles/monoVO.dir/examples/monoVO.cpp.o"

# External object files for target monoVO
monoVO_EXTERNAL_OBJECTS =

monoVO: CMakeFiles/monoVO.dir/src/matching2D.cpp.o
monoVO: CMakeFiles/monoVO.dir/src/cvEngine.cpp.o
monoVO: CMakeFiles/monoVO.dir/examples/monoVO.cpp.o
monoVO: CMakeFiles/monoVO.dir/build.make
monoVO: /usr/local/lib/libopencv_gapi.so.4.1.1
monoVO: /usr/local/lib/libopencv_stitching.so.4.1.1
monoVO: /usr/local/lib/libopencv_stereo.so.4.1.1
monoVO: /usr/local/lib/libopencv_xobjdetect.so.4.1.1
monoVO: /usr/local/lib/libopencv_hfs.so.4.1.1
monoVO: /usr/local/lib/libopencv_bgsegm.so.4.1.1
monoVO: /usr/local/lib/libopencv_bioinspired.so.4.1.1
monoVO: /usr/local/lib/libopencv_xfeatures2d.so.4.1.1
monoVO: /usr/local/lib/libopencv_quality.so.4.1.1
monoVO: /usr/local/lib/libopencv_superres.so.4.1.1
monoVO: /usr/local/lib/libopencv_surface_matching.so.4.1.1
monoVO: /usr/local/lib/libopencv_dpm.so.4.1.1
monoVO: /usr/local/lib/libopencv_aruco.so.4.1.1
monoVO: /usr/local/lib/libopencv_saliency.so.4.1.1
monoVO: /usr/local/lib/libopencv_optflow.so.4.1.1
monoVO: /usr/local/lib/libopencv_freetype.so.4.1.1
monoVO: /usr/local/lib/libopencv_img_hash.so.4.1.1
monoVO: /usr/local/lib/libopencv_shape.so.4.1.1
monoVO: /usr/local/lib/libopencv_fuzzy.so.4.1.1
monoVO: /usr/local/lib/libopencv_line_descriptor.so.4.1.1
monoVO: /usr/local/lib/libopencv_tracking.so.4.1.1
monoVO: /usr/local/lib/libopencv_plot.so.4.1.1
monoVO: /usr/local/lib/libopencv_ximgproc.so.4.1.1
monoVO: /usr/local/lib/libopencv_dnn_objdetect.so.4.1.1
monoVO: /usr/local/lib/libopencv_ccalib.so.4.1.1
monoVO: /usr/local/lib/libopencv_datasets.so.4.1.1
monoVO: /usr/local/lib/libopencv_reg.so.4.1.1
monoVO: /usr/local/lib/libopencv_face.so.4.1.1
monoVO: /usr/local/lib/libopencv_videostab.so.4.1.1
monoVO: /usr/local/lib/libopencv_structured_light.so.4.1.1
monoVO: /usr/local/lib/libopencv_hdf.so.4.1.1
monoVO: /usr/local/lib/libopencv_xphoto.so.4.1.1
monoVO: /usr/local/lib/libopencv_cvv.so.4.1.1
monoVO: /usr/local/lib/libopencv_rgbd.so.4.1.1
monoVO: /usr/local/lib/libopencv_text.so.4.1.1
monoVO: /usr/lib/x86_64-linux-gnu/libboost_system.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_thread.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_regex.so
monoVO: /usr/lib/x86_64-linux-gnu/libpthread.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_common.so
monoVO: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_search.so
monoVO: /usr/lib/libOpenNI.so
monoVO: /usr/lib/libOpenNI2.so
monoVO: /usr/lib/x86_64-linux-gnu/libz.so
monoVO: /usr/lib/x86_64-linux-gnu/libjpeg.so
monoVO: /usr/lib/x86_64-linux-gnu/libpng.so
monoVO: /usr/lib/x86_64-linux-gnu/libtiff.so
monoVO: /usr/lib/x86_64-linux-gnu/libfreetype.so
monoVO: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
monoVO: /usr/lib/x86_64-linux-gnu/libnetcdf.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
monoVO: /usr/lib/x86_64-linux-gnu/libsz.so
monoVO: /usr/lib/x86_64-linux-gnu/libdl.so
monoVO: /usr/lib/x86_64-linux-gnu/libm.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
monoVO: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
monoVO: /usr/lib/x86_64-linux-gnu/libexpat.so
monoVO: /usr/lib/x86_64-linux-gnu/libpython2.7.so
monoVO: /usr/lib/libgl2ps.so
monoVO: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
monoVO: /usr/lib/x86_64-linux-gnu/libtheoradec.so
monoVO: /usr/lib/x86_64-linux-gnu/libogg.so
monoVO: /usr/lib/x86_64-linux-gnu/libxml2.so
monoVO: /usr/lib/libvtkWrappingTools-6.2.a
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_io.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_features.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
monoVO: /usr/lib/x86_64-linux-gnu/libqhull.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_people.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_system.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_thread.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
monoVO: /usr/lib/x86_64-linux-gnu/libboost_regex.so
monoVO: /usr/lib/x86_64-linux-gnu/libpthread.so
monoVO: /usr/lib/x86_64-linux-gnu/libqhull.so
monoVO: /usr/lib/libOpenNI.so
monoVO: /usr/lib/libOpenNI2.so
monoVO: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingStencil-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libz.so
monoVO: /usr/lib/x86_64-linux-gnu/libjpeg.so
monoVO: /usr/lib/x86_64-linux-gnu/libpng.so
monoVO: /usr/lib/x86_64-linux-gnu/libtiff.so
monoVO: /usr/lib/x86_64-linux-gnu/libfreetype.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOParallelNetCDF-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
monoVO: /usr/lib/x86_64-linux-gnu/libnetcdf.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
monoVO: /usr/lib/x86_64-linux-gnu/libpthread.so
monoVO: /usr/lib/x86_64-linux-gnu/libsz.so
monoVO: /usr/lib/x86_64-linux-gnu/libdl.so
monoVO: /usr/lib/x86_64-linux-gnu/libm.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
monoVO: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
monoVO: /usr/lib/x86_64-linux-gnu/libexpat.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkLocalExample-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libpython2.7.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkTestingGenericBridge-6.2.so.6.2.0
monoVO: /usr/lib/libgl2ps.so
monoVO: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
monoVO: /usr/lib/x86_64-linux-gnu/libtheoradec.so
monoVO: /usr/lib/x86_64-linux-gnu/libogg.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOMINC-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingImage-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libxml2.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersReebGraph-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOXdmf2-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOAMR-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingStatistics-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOParallel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIONetCDF-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtOpenGL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOParallelLSDyna-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelGeometry-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtWebkit-6.2.so.6.2.0
monoVO: /usr/lib/libvtkWrappingTools-6.2.a
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersHyperTree-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeOpenGL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOPostgreSQL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkWrappingJava-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelFlowPaths-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelStatistics-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersProgrammable-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelImaging-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallelLIC-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingLIC-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersPython-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOParallelExodus-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneric-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOVideo-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOInfovis-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtSQL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeOpenGL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkInfovisBoostGraphAlgorithms-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOGeoJSON-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersVerdict-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkViewsGeovis-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOImport-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkTestingIOSQL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOODBC-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOEnSight-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOMySQL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingMatplotlib-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkDomainsChemistry-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelMPI-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOParallelXML-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkTestingRendering-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOMPIParallel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI4Py-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersSMP-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersSelection-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOVPIC-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkVPIC-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingMath-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingMorphological-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOFFMPEG-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOMPIImage-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOGDAL-6.2.so.6.2.0
monoVO: /usr/local/lib/libopencv_ml.so.4.1.1
monoVO: /usr/local/lib/libopencv_dnn.so.4.1.1
monoVO: /usr/local/lib/libopencv_highgui.so.4.1.1
monoVO: /usr/local/lib/libopencv_objdetect.so.4.1.1
monoVO: /usr/local/lib/libopencv_video.so.4.1.1
monoVO: /usr/local/lib/libopencv_videoio.so.4.1.1
monoVO: /usr/local/lib/libopencv_imgcodecs.so.4.1.1
monoVO: /usr/local/lib/libopencv_phase_unwrapping.so.4.1.1
monoVO: /usr/local/lib/libopencv_photo.so.4.1.1
monoVO: /usr/local/lib/libopencv_calib3d.so.4.1.1
monoVO: /usr/local/lib/libopencv_features2d.so.4.1.1
monoVO: /usr/local/lib/libopencv_imgproc.so.4.1.1
monoVO: /usr/local/lib/libopencv_flann.so.4.1.1
monoVO: /usr/local/lib/libopencv_viz.so.4.1.1
monoVO: /usr/local/lib/libopencv_core.so.4.1.1
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOExport-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingGL2PS-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_common.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_search.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_io.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_features.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_people.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
monoVO: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkxdmf2-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libxml2.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.so
monoVO: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.so
monoVO: /usr/lib/x86_64-linux-gnu/libsz.so
monoVO: /usr/lib/x86_64-linux-gnu/libdl.so
monoVO: /usr/lib/x86_64-linux-gnu/libm.so
monoVO: /usr/lib/x86_64-linux-gnu/libsz.so
monoVO: /usr/lib/x86_64-linux-gnu/libdl.so
monoVO: /usr/lib/x86_64-linux-gnu/libm.so
monoVO: /usr/lib/openmpi/lib/libmpi.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOLSDyna-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkViewsQt-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersAMR-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersFlowPaths-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOExodus-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkexoIIc-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
monoVO: /usr/lib/x86_64-linux-gnu/libnetcdf.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.5.1
monoVO: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.5.1
monoVO: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.5.1
monoVO: /usr/lib/x86_64-linux-gnu/libvtkverdict-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkGeovisCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkproj4-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkViewsInfovis-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkInfovisLayout-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersImaging-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOSQL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkWrappingPython27Core-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkPythonInterpreter-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOXML-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libpython2.7.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libGLU.so
monoVO: /usr/lib/x86_64-linux-gnu/libSM.so
monoVO: /usr/lib/x86_64-linux-gnu/libICE.so
monoVO: /usr/lib/x86_64-linux-gnu/libX11.so
monoVO: /usr/lib/x86_64-linux-gnu/libXext.so
monoVO: /usr/lib/x86_64-linux-gnu/libXt.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libfreetype.so
monoVO: /usr/lib/x86_64-linux-gnu/libGL.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOMovie-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
monoVO: /usr/lib/x86_64-linux-gnu/libtheoradec.so
monoVO: /usr/lib/x86_64-linux-gnu/libogg.so
monoVO: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkParallelCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtksys-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.2.so.6.2.0
monoVO: /usr/lib/x86_64-linux-gnu/libz.so
monoVO: CMakeFiles/monoVO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shubham/workspace/TAPL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable monoVO"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/monoVO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/monoVO.dir/build: monoVO

.PHONY : CMakeFiles/monoVO.dir/build

CMakeFiles/monoVO.dir/requires: CMakeFiles/monoVO.dir/src/matching2D.cpp.o.requires
CMakeFiles/monoVO.dir/requires: CMakeFiles/monoVO.dir/src/cvEngine.cpp.o.requires
CMakeFiles/monoVO.dir/requires: CMakeFiles/monoVO.dir/examples/monoVO.cpp.o.requires

.PHONY : CMakeFiles/monoVO.dir/requires

CMakeFiles/monoVO.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/monoVO.dir/cmake_clean.cmake
.PHONY : CMakeFiles/monoVO.dir/clean

CMakeFiles/monoVO.dir/depend:
	cd /home/shubham/workspace/TAPL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shubham/workspace/TAPL /home/shubham/workspace/TAPL /home/shubham/workspace/TAPL/build /home/shubham/workspace/TAPL/build /home/shubham/workspace/TAPL/build/CMakeFiles/monoVO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/monoVO.dir/depend
