#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include "dataStructures.hpp"
#include "matching2D.hpp"
#include "ringBuffer.hpp"
#include "cvEngine.hpp"
#include "ptEngine.hpp"
#include "render.hpp"

using namespace std;

int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../data/living_room";

    // number of images
    int nImages = -1;   // last file index to load

    // misc
    int dataBufferSize = 200;       // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false;            // visualize results
    RingBuffer<tapl::DataFrame> dataBuffer(dataBufferSize);

    // Read camera calibration
    cv::FileStorage opencv_file("../scripts/camera_model.yaml", cv::FileStorage::READ);
    cv::Mat camera_matrix;
    opencv_file["camera_matrix"] >> camera_matrix;
    cv::Mat dist_coeff;
    opencv_file["dist_coeff"] >> dist_coeff;
    opencv_file.release();
    std::cout << "Camera Matrix:" << std::endl;
    std::cout << camera_matrix << std::endl;
    std::cout << "Distortion Coefficients:" << std::endl;
    std::cout << dist_coeff << std::endl;

    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    int imgIndex = 0;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto fname : fnames) {
        /* Load image into buffer */
        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(fname);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        tapl::DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push(frame);
        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    } 

    if(tapl::cve::getPose(dataBuffer, camera_matrix) == 0) {
        // prepare a viewer
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (255, 255, 255);
        viewer->addCoordinateSystem (0.5); // origin
        viewer->initCameraParameters ();
        
        CameraAngle setAngle = TopDown;
        int distance = 50;
        switch(setAngle)
        {
            case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
            case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
            case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
            case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
        }
        
        cv::Mat R_total = cv::Mat::eye(3, 3, CV_32FC1);
        cv::Mat t_total = cv::Mat::zeros(3, 1, CV_32FC1);
        cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_32FC1);
        pcl::PointXYZ prev_point = pcl::PointXYZ(0.0,0.0,0.0);
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            cv::Mat R = dataBuffer.get_ptr(imgIndex)->pose.R;
            cv::Mat t = dataBuffer.get_ptr(imgIndex)->pose.t;
            cv::Mat euler = dataBuffer.get_ptr(imgIndex)->pose.euler;

            cout << "R: " << endl;
            cout << R << endl;
            cout << "t:" << endl;
            cout << t << endl;
            cout << "[roll,pitch,yaw]: [" << euler.at<float>(0,0) << "," << euler.at<float>(0,1) << "," << euler.at<float>(0,2) << "]" << std::endl;

            // add the second camera pose 
            Eigen::Matrix4f eig_mat;
            Eigen::Affine3f cam_pose;

            R.convertTo(R, CV_32F);
            t.convertTo(t, CV_32F);

            cv::Mat transformation = cv::Mat::eye(4, 4, CV_32FC1);
            transformation(cv::Rect(0, 0, 3, 3)) = R * 1.0;
            transformation.col(3).rowRange(0, 3) = t * 1.0;
            camera_pose = camera_pose * transformation;
            cout << "pose: " << camera_pose.at<double>(0, 3) << "," << camera_pose.at<double>(1, 3) << "," <<  camera_pose.at<double>(2, 3) << endl;

            cv::Mat point3d_homo = dataBuffer.get_ptr(imgIndex)->triangulated_pts;
            
            // create point cloud
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
            cloud->points.resize (point3d_homo.cols);

            for(int i = 0; i < point3d_homo.cols; i++) {
                pcl::PointXYZRGB &point = cloud->points[i];
                cv::Mat p3d;
                cv::Mat _p3h = point3d_homo.col(i);
                convertPointsFromHomogeneous(_p3h.t(), p3d);
                point.x = p3d.at<double>(0);
                point.y = p3d.at<double>(1);
                point.z = p3d.at<double>(2);
                point.r = 0;
                point.g = 0;
                point.b = 255;
                tapl::pte::world2CamCoordinate<pcl::PointXYZRGB>(point);
            }

            for(int i = 0; i < 4; i++) 
            for(int j = 0; j < 4; j++) 
                cam_pose(i,j) = camera_pose.at<float>(i,j);

            // pcl::transformPointCloud (*cloud, *cloud, cam_pose);
            // // renderPointCloud(viewer, cloud, "cloud_"+std::to_string(imgIndex));
            // viewer->addPointCloud(cloud,"cloud_"+std::to_string(imgIndex));
            // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
            //                                             0.5,
            //                                             "cloud_"+std::to_string(imgIndex));
            pcl::PointXYZ cam_pose_pt = pcl::PointXYZ(cam_pose(0, 3), cam_pose(1, 3), cam_pose(2, 3));
            tapl::pte::world2CamCoordinate<pcl::PointXYZ>(cam_pose_pt);
            viewer->addSphere (cam_pose_pt, 0.2, "sphere_"+std::to_string(imgIndex));
            viewer->addLine(prev_point, cam_pose_pt, "line_"+std::to_string(imgIndex));
            prev_point = cam_pose_pt;

            // cam_pose should be Affine3f, Affine3d cannot be used
            // viewer->addCoordinateSystem(0.5, cam_pose, "pt_"+std::to_string(imgIndex)); //TODO: Rotate this coordinate system into camera coordinate

            string windowName = "Camera Frame";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, dataBuffer.get_ptr(imgIndex)->cameraImg);
            // cout << "Press key to continue to next image" << endl;
            cv::waitKey(1); 

            viewer->spinOnce(500);
            // std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        while (!viewer->wasStopped ()) {
            viewer->spin();
        }
    }

    return 0;
}