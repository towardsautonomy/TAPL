cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
add_definitions("-lstdc++fs")

project(TAPL_examples)
# Find OvenCV
find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find PCL
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# Find TAPL
find_package(tapl REQUIRED)

# Add all the libraries
set(DEP_LIBRARIES tapl ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} "-lstdc++fs")

##==============================================##
##================Build Examples================##
##==============================================##
# list executables
add_executable (kptsDetectionAndTracking src/kptsDetectionAndTracking.cpp )
add_executable (lidarObjectDetection src/lidarObjectDetection.cpp)
add_executable (panoramaicStitching src/panoramaicStitching.cpp)
add_executable (structureFromMotion src/structureFromMotion.cpp)

# link library
target_link_libraries(kptsDetectionAndTracking ${DEP_LIBRARIES})
target_link_libraries(lidarObjectDetection ${DEP_LIBRARIES})
target_link_libraries(panoramaicStitching ${DEP_LIBRARIES})
target_link_libraries(structureFromMotion ${DEP_LIBRARIES})