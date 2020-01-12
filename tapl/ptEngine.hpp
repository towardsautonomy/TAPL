/**
 * @file    ptEngine.hpp
 * @brief   This file provides APIs for all point related functions.
 * 				This includes point-cloud processing, point transformations,
 * .
 * @author  Shubham Shrivastava
 */

#ifndef PT_ENGINE_H_
#define PT_ENGINE_H_

/* Helper functions for processing point-cloud data */
#include <unordered_set>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>

#include "taplTypes.hpp"

inline float degreesToRadians(float angleDegrees) { return ((angleDegrees) * M_PI / 180.0); }
inline float radiansToDegrees(float angleRadians) { return ((angleRadians) * 180.0 / M_PI); }

namespace tapl {
    namespace pte {
        template <typename PointT>
        class Line {
            public:
                // constructor 
                Line();

                // de-constructor 
                ~Line();

                // line fitting using SVD method
                std::vector<float> fitSVD(std::vector<float> &x, std::vector<float> &y);

                // line fitting using least-squares method
                std::vector<float> fitLS(std::vector<float> &x, std::vector<float> &y);

                // line to point distance
                float distToPoint(std::vector<float> line_coeffs, PointT point);

                // RANSAC for 2D Points
                std::unordered_set<int> Ransac(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol);
        };

        template <typename PointT>
        class Plane {
            public:
                // constructor 
                Plane();

                // de-constructor 
                ~Plane();

                // plane fitting using SVD method
                std::vector<float> fitSVD(std::vector<float> &x, std::vector<float> &y, std::vector<float> &z);

                // plane fitting using least-squares method
                std::vector<float> fitLS(std::vector<float> &x, std::vector<float> &y, std::vector<float> &z);

                // plane to point distance
                float distToPoint(std::vector<float> plane_coeffs, PointT point);

                // RANSAC for 3D Points
                std::unordered_set<int> Ransac(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceToPlane);
        };

        // Structure to represent node of kd tree
        struct Node {
            std::vector<float> point;
            int id;
            Node* left;
            Node* right;

            Node(std::vector<float> arr, int setId)
            :	point(arr), id(setId), left(NULL), right(NULL)
            {}
        };

        // k-d tree struct
        struct KdTree {
            // root of the tree
            Node* root;

            // default constructor
            KdTree() : root(NULL) {}

            // insert method
            void insert(std::vector<float> point, int id);

            // search function
            std::vector<int> search(std::vector<float> target, float distTolerance);

            // distance between two points
            float dist(std::vector<float> point_a, std::vector<float> point_b);

            // helper function for insert
            void insertHelper(Node ** node, unsigned depth, std::vector<float> point, int id);

            // helper function for search
            void searchHelper(Node * node, std::vector<float> target, float distTolerance, int depth, std::vector<int>& ids);
        };

        class EuclideanCluster {
            public:
                // constructor
                EuclideanCluster(const std::vector<std::vector<float>>& points, KdTree * tree)
                :   points(points), tree(tree)
                {}

                // clustering function
                std::vector<std::vector<int>> clustering(float distTolerance);

            private:
                // vector of 3d points
                const std::vector<std::vector<float>> points;

                // k-d tree
                KdTree * tree;

                // find points in the proximity of the point indexed 'pointIndex'
                void proximityPoints( int pointIndex,
                                        std::vector<bool>& checked,
                                        float distTolerance, 
                                        std::vector<int>& cluster);
        };

        /** 
         * returns world to camera rotation matrix 
         * 
         *   Camera Coordinate System:
         *       X -> To the right
         *       Y -> Down
         *       Z -> Forward - Direction where the camera is pointing
         *
         *   World Coordinate System:
         *       X -> Forward - Direction where the camera is pointing
         *       Y -> To the left
         *       Z -> Up
         */
        cv::Mat world2CamRotation() {
            // camera coordinate to world coordinate rotation matrix
            cv::Mat R = cv::Mat::zeros(3, 3, CV_32F);
            // Camera rotation
            float Rx = degreesToRadians(-90);
            float Ry = degreesToRadians(0);
            float Rz = degreesToRadians(-90);
            
            // Rz
            cv::Mat R_z = cv::Mat::eye(3, 3, CV_32F);
            R_z.at<float>(0, 0) = cos(Rz);
            R_z.at<float>(0, 1) = -sin(Rz);
            R_z.at<float>(1, 0) = sin(Rz);
            R_z.at<float>(1, 1) = cos(Rz);
            // Ry
            cv::Mat R_y = cv::Mat::eye(3, 3, CV_32F);
            R_y.at<float>(0, 0) = cos(Ry);
            R_y.at<float>(0, 2) = sin(Ry);
            R_y.at<float>(2, 0) = -sin(Ry);
            R_y.at<float>(2, 2) = cos(Ry);
            // Rx
            cv::Mat R_x = cv::Mat::eye(3, 3, CV_32F);
            R_y.at<float>(1, 1) = cos(Rx);
            R_y.at<float>(1, 2) = -sin(Rx);
            R_y.at<float>(2, 1) = sin(Rx);
            R_y.at<float>(2, 2) = cos(Rx);

                            
            // Camera Rotation Correction Matrix
            R = R_z * R_y * R_x;
            
            return R;
        }

        /** 
         * affine transform on a point 
         * 
         * Apply affine transforms on point given in world coordinate
         *
         *
         *   Camera Coordinate System:
         *       X -> To the right
         *       Y -> Down
         *       Z -> Forward - Direction where the camera is pointing
         *
         *   World Coordinate System:
         *       X -> Forward - Direction where the camera is pointing
         *       Y -> To the left
         *       Z -> Up
         */
        template <typename PointT>
        void world2CamCoordinate(PointT &point) {      
            // Camera Rotation Correction Matrix
            cv::Mat R = world2CamRotation();
            
            cv::Mat xyz = cv::Mat(3, 1, CV_32F);
            xyz.at<float>(0, 0) = point.x;
            xyz.at<float>(1, 0) = point.y;
            xyz.at<float>(2, 0) = point.z;

            cv::Mat xyz_w = cv::Mat(3, 1, CV_32F);

            xyz_w = R * xyz;
            point.x = xyz_w.at<float>(0, 0);
            point.y = xyz_w.at<float>(1, 0);
            point.z = xyz_w.at<float>(2, 0);
        }
    };
};

#endif /* PT_ENGINE_H_ */
