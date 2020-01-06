# Towards Autonomy Perception Library (TAPL)

Goal of this library is to provide an easy and quick way of implementing several well known algorithms like Visual Odometry, Visual SLAM, RANSAC, PnP, Bundle Adjustment etc.

TODO Notes:

 - one header file 'tapl.h' at the parent location which includes everythin under 'include/'. This should be the only file that needs to be included by any other code.  
 - modify the data structure *DataFrame* to use new *structs* *CameraFrame*. Another struct *StereoCamFrame*  uses two instances of it for left and right camera frames. 
 - For each data, use a boolean flag to specify if it exists.  
 - Use a struct constructor to initialize the struct and set false flags.  
  
