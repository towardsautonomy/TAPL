cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
add_definitions("-lstdc++fs")

project(TAPL)
include_directories(include)

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

## List Executables
# feature tracking
add_executable (kpts_detect_match src/matching2D.cpp src/cvEngine.cpp examples/feature_tracking.cpp )
target_include_directories(kpts_detect_match PUBLIC ./include)
target_link_libraries (kpts_detect_match ${OpenCV_LIBRARIES} "-lstdc++fs")

# compute fundamental matrix
add_executable (compute_f_matrix src/matching2D.cpp src/cvEngine.cpp examples/compute_f_matrix.cpp)
target_include_directories(compute_f_matrix PUBLIC ./include)
target_link_libraries (compute_f_matrix ${OpenCV_LIBRARIES})
#target_link_libraries (compute_f_matrix ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

# compute essential matrix
add_executable (compute_e_matrix src/matching2D.cpp src/cvEngine.cpp examples/compute_e_matrix.cpp)
target_include_directories(compute_e_matrix PUBLIC ./include)
target_link_libraries (compute_e_matrix ${OpenCV_LIBRARIES})

# compute pose
add_executable (compute_pose src/matching2D.cpp src/cvEngine.cpp src/render.cpp examples/compute_pose.cpp)
target_include_directories(compute_pose PUBLIC ./include)
target_link_libraries (compute_pose ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} "-lstdc++fs")