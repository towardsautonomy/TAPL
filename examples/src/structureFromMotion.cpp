#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <filesystem>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <opencv2/core/eigen.hpp>

#include "tapl.hpp"

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

int main(int argc, const char *argv[])
{
    ///////////////////////////////////////
    // Load Images
    ///////////////////////////////////////
    // paths
    std::string calibPath = "../data/calib/statue_camera_model.yaml";
    std::string dataPath = "../data/statue";

    // number of images
    int nImages = -1;   // last file index to load

    // images
    std::vector<cv::Mat> imgs;

    // Read camera calibration
    cv::FileStorage opencv_file(calibPath, cv::FileStorage::READ);
    cv::Mat camera_matrix;
    opencv_file["camera_matrix"] >> camera_matrix;
    cv::Mat dist_coeff;
    opencv_file["dist_coeff"] >> dist_coeff;
    opencv_file.release();

    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());

    int imgIndex = 0;
    for (auto it=fnames.begin(); it!=fnames.end(); ++it) {
        // load image from file, undistort, and convert to rgb
        cv::Mat img, img_undistorted, img_gray;
        img = cv::imread(*it);
        cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);
        cv::cvtColor(img_undistorted, img_undistorted, cv::COLOR_BGR2RGB);

        // push image into data frame buffer
        imgs.push_back(img_undistorted);
        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    }

    ///////////////////////////////////////
    // Structure-from-Motion
    ///////////////////////////////////////
    // create an sfm object
    auto sfm = tapl::cve::StructureFromMotion(imgs, camera_matrix);
    // compute structure-from-motion
    std::vector<tapl::Point3dColor> points;
    std::vector<tapl::Pose6dof> poses;
    std::vector<tapl::CameraPairs> framePairs;
    if (sfm.process(points, poses, framePairs) != tapl::SUCCESS) {
        TLOG_ERROR << "could not compute structure-from-motion";
        exit(1);
    }
            
    ///////////////////////////////////////
    // Visualization
    ///////////////////////////////////////
    // prepare a viewer
    tapl::viz::Visualizer * visualizer = new tapl::viz::Visualizer(0,0,0,tapl::viz::FPS,2);
    // add reference pose at the origin
    Eigen::Affine3f affinePose = Eigen::Affine3f::Identity();
    Eigen::Matrix4f globalPose = Eigen::Matrix4f::Identity();
    // rotate from world to camera coordinate
    double Rx, Ry, Rz;
    Rx = deg2rad(-90);
    Ry = deg2rad(0);
    Rz = deg2rad(-90);
    Eigen::Matrix3f R = ( Eigen::AngleAxis<float>(Rz, Eigen::Vector3f::UnitZ()) *
                          Eigen::AngleAxis<float>(Ry, Eigen::Vector3f::UnitY()) *
                          Eigen::AngleAxis<float>(Rx, Eigen::Vector3f::UnitX()))
                            .toRotationMatrix();
    Eigen::Matrix<float, 3, 1> t(0.0, 0.0, 0.0);
    globalPose.block(0, 0, 3, 3) = R;
    globalPose.block(0, 3, 3, 1) = t;
    Eigen::Matrix4f cam2PclT = globalPose.matrix();
    
    // render reference pose
    affinePose = cam2PclT * affinePose.matrix();
    visualizer->renderPose(0.25, affinePose, "ref_pose");
    pcl::PointXYZ origin_pt = pcl::PointXYZ(0,0,0);
    visualizer->renderSphere(origin_pt, 0.1, 1.0, 0.0, 0.0, "ref_sphere");
    pcl::PointXYZ prev_point = pcl::PointXYZ(0.0,0.0,0.0);

    // build point-cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->points.resize (points.size());

    for(int i = 0; i < points.size(); i++) {
        pcl::PointXYZRGB &point = cloud->points[i];
        point.x = points[i].x;
        point.y = points[i].y;
        point.z = points[i].z;
        point.r = points[i].r;
        point.g = points[i].g;
        point.b = points[i].b;
    }

    // convert to PCL's coordinate frame
    pcl::transformPointCloud (*cloud, *cloud, cam2PclT);
    visualizer->renderPointCloud<pcl::PointXYZRGB> (cloud, 5.0, -1.0, -1.0, -1.0, "cloud");

    // go through each pose and render
    uint16_t frameIdx = 0;
    for (auto &pose : poses) {
        // Eigen::Matrix4f eigenPose;
        cv::cv2eigen(pose.T, globalPose);
        affinePose = cam2PclT * globalPose.matrix();

        // render pose
        visualizer->renderPose(0.25, affinePose, "pose_" + std::to_string(frameIdx));
        pcl::PointXYZ cam_pose_pt = pcl::PointXYZ(affinePose(0, 3), affinePose(1, 3), affinePose(2, 3));
        visualizer->renderSphere(cam_pose_pt, 0.1, 1.0, 0.0, 0.0, "sphere_"+std::to_string(frameIdx));
        visualizer->renderLine(prev_point, cam_pose_pt, 0.0, 0.0, 1.0, "line_"+std::to_string(frameIdx));
        prev_point = cam_pose_pt;
        visualizer->renderScene(10);

        // render 2D keypoints and triangulated 3D keypoints
        std::vector<tapl::Point2d> trackedKpts;
        if (framePairs[frameIdx].getTrackedKpts(trackedKpts) != tapl::SUCCESS) {
            TLOG_ERROR << "could not find tracked keypoints";
            exit(1);
        }
        std::vector<tapl::Point3dColor> triangulatedPts;
        if (framePairs[frameIdx].getTriangulatedPoints(triangulatedPts) != tapl::SUCCESS) {
            TLOG_ERROR << "could not find triangulated 3D points";
            exit(1);
        }
        cv::Mat imgViz;
        if (framePairs[frameIdx].first->getImage(imgViz) != tapl::SUCCESS) {
            TLOG_ERROR << "could not retrieve image";
            exit(1);
        }
        // blue points are 2D, red points are 2D projections of triangulated 3D points
        cv::cvtColor(imgViz, imgViz, cv::COLOR_RGB2BGR);
        imgViz = tapl::cve::drawKeypoints(imgViz, trackedKpts, cv::Scalar(255,0,0));
        imgViz = tapl::cve::drawKeypoints3D(imgViz, triangulatedPts, camera_matrix, cv::Scalar(0,0,255));
        cv::imshow("frame_"+std::to_string(frameIdx), imgViz);
        cv::waitKey(1); 
        frameIdx ++;

    }
    // // write to file
    // pcl::io::savePLYFileBinary("sfm.ply", *cloud);
    cv::waitKey(0); 
    visualizer->renderSceneAndHold();

    // return
    return 0;
}