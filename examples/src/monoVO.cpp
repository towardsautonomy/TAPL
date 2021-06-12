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

#include <opencv2/opencv.hpp>

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

#include "tapl.hpp"

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

using namespace std;

int main(int argc, const char *argv[])
{
    // paths
    string calibPath = "../data/calib/camera_model.yaml";
    string dataPath = "../data/living_room";

    // number of images
    int nImages = -1;   // last file index to load

    // misc
    int dataBufferSize = 4;       // no. of images which are held in memory (ring buffer) at the same time
    tapl::RingBuffer<tapl::CameraPairs> dataBuffer(dataBufferSize);

    // Read camera calibration
    cv::FileStorage opencv_file(calibPath, cv::FileStorage::READ);
    cv::Mat camera_matrix;
    opencv_file["camera_matrix"] >> camera_matrix;
    cv::Mat dist_coeff;
    opencv_file["dist_coeff"] >> dist_coeff;
    opencv_file.release();

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

    cv::Mat R_total = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat t_total = cv::Mat::zeros(3, 1, CV_32FC1);
    pcl::PointXYZ prev_point = pcl::PointXYZ(0.0,0.0,0.0);
    
    // prepare a viewer
    tapl::viz::Visualizer * visualizer = new tapl::viz::Visualizer();
    visualizer->renderPose(0.5, pose, "ref");

    string windowName = "Camera Frame";
    cv::namedWindow(windowName, 7);

    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    int imgIndex = 0;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto it=fnames.begin(); it!=fnames.end(); ++it) {
        // wait for at least 2 images
        if (fnames.begin() == it) {
            // load image from file and convert to grayscale
            cv::Mat img, img_undistorted, img_gray;
            img = cv::imread(*it);
            cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);
            cv::cvtColor(img_undistorted, img_gray, cv::COLOR_BGR2GRAY);

            // push image into data frame buffer
            tapl::CameraPairs camPairs;
            tapl::CameraFrame frame(img_gray);
            *camPairs.second = frame;
            camPairs.second->pushIntrinsicMatrix(camera_matrix);
            dataBuffer.push(camPairs);
        }
        else {
            /* Load image into buffer */
            // load image from file and convert to grayscale
            cv::Mat img, img_undistorted, img_gray;
            img = cv::imread(*it);
            cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);
            cv::cvtColor(img_undistorted, img_gray, cv::COLOR_BGR2GRAY);

            // push image into data frame buffer
            cv::Mat prev_img;
            if (dataBuffer.get(dataBuffer.getSize()-1).second->getImage(prev_img) != tapl::SUCCESS) {
                TLOG_ERROR << "could not retrieve previous frame";
                exit(1);
            }
            tapl::CameraPairs camPairs(prev_img, img_gray);
            camPairs.first->pushIntrinsicMatrix(camera_matrix);
            camPairs.second->pushIntrinsicMatrix(camera_matrix);
            dataBuffer.push(camPairs);

            TLOG_INFO << "----------------------------------------";
            TLOG_INFO << "Image [" << imgIndex << "] loaded into the ring buffer";

            // transformation matrix
            Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
            // get relative pose
            if(tapl::cve::computeRelativePose(camPairs) == tapl::SUCCESS) {
                tapl::Pose6dof relative_pose;
                if(camPairs.getPose(relative_pose) == tapl::SUCCESS) {
                    // rotation
                    cv2eigen(relative_pose.R, R);
                    P.block(0, 0, 3, 3) = R;
                    // translation
                    cv2eigen(relative_pose.t, t);
                    P.block(0, 3, 3, 1) = t;
                }
                else {
                    TLOG_INFO << "ERROR: could not retrieve relative pose";
                    exit(EXIT_FAILURE);
                }
            }
            else {
                TLOG_INFO << "ERROR: could not compute relative pose";
                exit(EXIT_FAILURE);
            }
            
            // compute pose from the camera origin
            // complete transformation matrix
            complete_pose = complete_pose * P;
            pose = complete_pose.matrix();
            visualizer->renderPose(0.5, pose, "pose_" + std::to_string(imgIndex));

            // create point cloud from triangulated points
            cv::Mat point3d_homo;
            if(camPairs.getTriangulatedPoints(point3d_homo) != tapl::SUCCESS) {
                TLOG_INFO << "ERROR: could not retrieve triangulated 3D points";
                exit(EXIT_FAILURE);
            }
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
            visualizer->renderPointCloud<pcl::PointXYZ> (cloud, 1.0, 1.0, 0.0, 0.0, "cloud_"+std::to_string(imgIndex));
            pcl::PointXYZ cam_pose_pt = pcl::PointXYZ(pose(0, 3), pose(1, 3), pose(2, 3));
            visualizer->renderSphere(cam_pose_pt, 0.2, 1.0, 0.0, 0.0, "sphere_"+std::to_string(imgIndex));
            visualizer->renderLine(prev_point, cam_pose_pt, 0.0, 0.0, 1.0, "line_"+std::to_string(imgIndex));
            prev_point = cam_pose_pt;

            // display image
            cv::Mat img_disp;
            if(camPairs.first->getImage(img_disp) != tapl::SUCCESS) {
                TLOG_INFO << "ERROR: could not retrieve image frame";
                exit(EXIT_FAILURE);
            }
            cv::imshow(windowName, img_disp);
            cv::waitKey(1); 

            visualizer->renderScene(10);
            // std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    }

    // render the scene and hold
    visualizer->renderSceneAndHold();

    return 0;
}