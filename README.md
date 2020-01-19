
# Towards Autonomy Perception Library (TAPL)

Goal of this library is to provide an easy and quick way of implementing perception pipelines. 

![](media/tapl_architecture.png)

## [**Find on GitHub**](https://github.com/towardsautonomy/TAPL)

## Examples of Perception Task  

#### Visual Odometry for a sequence of Monocular camera images  

This example is provided at ```examples/src/monoVO.cpp```. It uses a sequence of monocular images to perform visual odometry and build sparse point-cloud. This functionality is provided as an API and can be accessed using the function: ```tapl::cve::computeRelativePose()```.  

**Pipeline**  

  - Read images and push into a ring buffer.  
  - If more than one image is available in the buffer then:  
    - Perform keypoint detection and matching.  
    - Compute essential matrix. 
    - Compute relative pose (R, t)  
    - Triangulate good keypoints for which a match is found.  
  - Compute global pose from this relative pose.  


![](media/mono_vo.gif)

#### LiDAR Object Detection  

This example is provided at ```examples/src/lidarObjectDetection.cpp```. It reads in PCD point-cloud files, performs downsampling, ground-plane segmentation, and clustering, and then some post-processing filtering to get 3D Bounding-Box of objects.  

**Pipeline**  

  - Load point-cloud data.  
  - Downsample point-cloud (voxelization).  
  - Crop the point-cloud based on a region of interest.  
  - Segment out ground-plane using RANSAC.  
    - For *n* iterations:  
      - Choose 3 random points.  
      - Fit a plane using least-squares.  
      - Count number of inliers within a certain distance threshold between each point and plane.  
    - Choose the plane that resulted in maximum number of inliers.  
    - Implemented as part of ```class tapl::pte::Plane()```.  
  - Perform Euclidian Clustering within the segmented point-cloud.  
    - Store the point-cloud as a **KdTree**. Implemented as ```struct tapl::pte::KdTree```.  
    - Perform euclidean clustering on the KdTree. Implemented as ```class tapl::pte::EuclideanCluster()```.  
  - Filter and Extract the bounding-boxes.  
  
![](media/lidar_object_detection.gif)  

![](media/clustering.gif)  


#### Image Feature Detection and Tracking  

This example is provided at ```examples/src/kptsDetectionAndTracking.cpp``` and this functionality is implemented as ```tapl::cve::detectAndMatchKpts()```.

![](media/matching_points.png)

#### RANSAC for line and plane fitting  

 - C++ implementation of RANSAC for line and plane fitting using both SVD and least-square methods are provided as part of ```class tapl::pte::Line()``` and ```class tapl::pte::Plane()```.  

<!-- <p float="left">
  <img src="media/line_fitting.png" width="200" height="200" />
  <img src="media/plane_fitting.png" width="400" height="200"/> 
</p> -->

Line Fitting using RANSAC     |  Plane Fitting using RANSAC
:----------------------------:|:-------------------------:
![ ](media/line_fitting.png)  |  ![ ](media/plane_fitting.png)
  
## Prerequisites  

 - CMake >= 3.5
 - OpenCV >= 4.1
 - PCL >= 1.2  
 - Eigen >= 3.2

 ## Installation Instructions  

 - Download the library.  

   ```
   git clone https://github.com/towardsautonomy/TAPL.git
   ```

 - Build and install the library as follows.  
 
   ```
   mkdir build  
   cd build
   cmake ..
   make
   sudo make install
   ```

 - Build the examples as follows.  

   ```
   cd examples
   mkdir build
   cd build
   cmake ..
   make
   ```