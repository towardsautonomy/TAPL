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
#include <opencv2/core/eigen.hpp>

#include "dataStructures.hpp"
#include "matching2D.hpp"
#include "ringBuffer.hpp"
#include "cvEngine.hpp"
#include "ptEngine.hpp"
#include "render.hpp"
#include "visualization.hpp"

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

using namespace std;

int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../data/living_room";

    // number of images
    int nImages = -1;   // last file index to load

    // misc
    int dataBufferSize = 250;       // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false;            // visualize results
    RingBuffer<tapl::DataFrame> dataBuffer(dataBufferSize);

    // Read camera calibration
    cv::FileStorage opencv_file("../data/calib/camera_model.yaml", cv::FileStorage::READ);
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

    // add reference pose at the origin
    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    // rotate from world to camera coordinate
    double Rx, Ry, Rz;
    Rx = deg2rad(-90);
    Ry = deg2rad(0);
    Rz = deg2rad(-90);
    Eigen::Matrix4f complete_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R = (Eigen::AngleAxis<float>(Rz, Eigen::Vector3f::UnitZ()) *
                        Eigen::AngleAxis<float>(Ry, Eigen::Vector3f::UnitY()) *
                        Eigen::AngleAxis<float>(Rx, Eigen::Vector3f::UnitX()))
                            .toRotationMatrix();
    Eigen::Matrix<float, 3, 1> t(0.0, 0.0, 0.0);
    complete_pose.block(0, 0, 3, 3) = R;
    complete_pose.block(0, 3, 3, 1) = t;
    pose = complete_pose.matrix();

    // prepare a viewer
    tapl::viz::Visualizer * visualizer = new tapl::viz::Visualizer();
    visualizer->renderPose(0.5, pose, "ref");

    string windowName = "Camera Frame";
    cv::namedWindow(windowName, 7);

    if(tapl::cve::getPose(dataBuffer, camera_matrix) == 0) {

        cv::Mat R_total = cv::Mat::eye(3, 3, CV_32FC1);
        cv::Mat t_total = cv::Mat::zeros(3, 1, CV_32FC1);
        cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_32FC1);
        pcl::PointXYZ prev_point = pcl::PointXYZ(0.0,0.0,0.0);
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            // transformation matrix
            Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
            // rotation
            cv2eigen(dataBuffer.get_ptr(imgIndex)->pose.R, R);
            P.block(0, 0, 3, 3) = R;
            // translation
            cv2eigen(dataBuffer.get_ptr(imgIndex)->pose.t, t);
            P.block(0, 3, 3, 1) = t;
            
            // compute pose from the camera origin
            // complete transformation matrix
            complete_pose = complete_pose * P;
            pose = complete_pose.matrix();
            visualizer->renderPose(0.5, pose, "pose_" + std::to_string(imgIndex));

            cv::Mat euler = dataBuffer.get_ptr(imgIndex)->pose.euler;

            // create point cloud from triangulated points
            cv::Mat point3d_homo = dataBuffer.get_ptr(imgIndex)->triangulated_pts;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
            cloud->points.resize (point3d_homo.cols);

            for(int i = 0; i < point3d_homo.cols; i++) {
                pcl::PointXYZ &point = cloud->points[i];
                cv::Mat p3d;
                cv::Mat _p3h = point3d_homo.col(i);
                convertPointsFromHomogeneous(_p3h.t(), p3d);
                point.x = p3d.at<double>(0);
                point.y = p3d.at<double>(1);
                point.z = p3d.at<double>(2);
            }

            pcl::transformPointCloud (*cloud, *cloud, pose);
            visualizer->renderPointCloud(cloud, 1.0, 1.0, 0.0, 0.0, "cloud_"+std::to_string(imgIndex));
            pcl::PointXYZ cam_pose_pt = pcl::PointXYZ(pose(0, 3), pose(1, 3), pose(2, 3));
            std::cout << "Camera position : \n" << dataBuffer.get_ptr(imgIndex)->pose.t << std::endl;
            visualizer->renderSphere(cam_pose_pt, 0.2, 1.0, 0.0, 0.0, "sphere_"+std::to_string(imgIndex));
            visualizer->renderLine(prev_point, cam_pose_pt, 0.0, 0.0, 1.0, "line_"+std::to_string(imgIndex));
            prev_point = cam_pose_pt;

            // display image
            cv::imshow(windowName, dataBuffer.get_ptr(imgIndex)->cameraImg);
            cv::waitKey(1); 

            visualizer->renderScene(200);
            // std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        visualizer->renderSceneAndHold();
    }

    return 0;
}