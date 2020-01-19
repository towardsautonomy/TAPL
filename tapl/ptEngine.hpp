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
         * @brief returns world to camera rotation matrix 
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
         * @return rotation matrix
         */
        cv::Mat world2CamRotation()
        {
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
            
			// return rotation matrix
            return R;
        }

        /** 
         * @brief affine transform on a point 
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
         * 
         * @param[in] point point in world coordinate
         * 
         * @return point in camera coordinate
         */
        template <typename PointT>
        void world2CamCoordinate(PointT &point)
        {
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

        /**
         * @brief Downsample point-cloud
         * 
         * @param[in,out] cloud Point-Cloud 
         * @param[in] resolution Target resolution for downsampling the cloud
         */
        template<typename PointT>
		void downsampleCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float resolution) 
        {
            // Create the filtering object
			pcl::VoxelGrid<PointT> vxl_grid;
			vxl_grid.setInputCloud (cloud);
			vxl_grid.setLeafSize (resolution, resolution, resolution);
			vxl_grid.filter (*cloud);
        }

        /**
         * @brief Crop point-cloud given a region of interest
         * 
         * @param[in,out] cloud Point-Cloud 
         * @param[in] x_min minimum point x-coordinate
         * @param[in] x_max maximum point x-coordinate
         * @param[in] y_min minimum point y-coordinate
         * @param[in] y_max maximum point y-coordinate
         * @param[in] z_min minimum point z-coordinate
         * @param[in] z_max maximum point z-coordinate
         */
        template<typename PointT>
		void cropCloud(typename pcl::PointCloud<PointT>::Ptr cloud, 
                                        float x_min, float x_max, 
                                        float y_min, float y_max, 
                                        float z_min, float z_max)
        {
            // define minimum and maximum points
			Eigen::Vector4f minPoint(x_min, y_min, z_min, 1.0);
			Eigen::Vector4f maxPoint(x_max, y_max, z_max, 1.0);

			// crop region of interest
			pcl::CropBox<PointT> cropCloud(true);
			cropCloud.setInputCloud(cloud);
			cropCloud.setMin(minPoint);
			cropCloud.setMax(maxPoint);
			cropCloud.filter(*cloud);
        }

        /**
         * @brief This function segments a plane within a point-cloud. Points within the 
         *              plane are inliers and other points are outliers. It populates 
         *              two separate point-clouds corresponding to inliers and outliers.
         * 
         * @param[in] cloud point-cloud to segment
         * @param[out] segResult segmentation result
         *                          Inliers: segResult.first
         *                          Outliers: segResult.second
         * @param[in] maxIteration Maximum number of iterations for plane fitting with RANSAC
         * @param[in] distanceThreshold Distance threshold between point and plane for 
         *                              classifying a point as an inlier.
         * @param[in] usePCL Whether to use PCL of TAPL's implementation of RANSAC
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        template<typename PointT>
		tapl::ResultCode segmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, 
										std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>& segResult,
										int maxIterations, float distanceThreshold, bool usePCL=false)
        {
            // if PCL is to be used
			if(usePCL)
			{
				// find inliers for the cloud.
				pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
				pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
				// Create the segmentation object
				pcl::SACSegmentation<PointT> seg;
				// Optional
				seg.setOptimizeCoefficients (true);
				// Mandatory
				seg.setModelType (pcl::SACMODEL_PLANE);
				seg.setMethodType (pcl::SAC_RANSAC);
				seg.setMaxIterations (maxIterations);
				seg.setDistanceThreshold (distanceThreshold);

				// Find inliers indices
				seg.setInputCloud (cloud);
				seg.segment (*inliers, *coefficients);

				if (inliers->indices.size () == 0)
				{
					segResult = std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>(cloud, cloud);
					return tapl::FAILURE;
				}

				// Create two new point clouds, one cloud with inliers and other with outliers
				// positive and negative planes
				typename pcl::PointCloud<PointT>::Ptr cloudInliers (new pcl::PointCloud<PointT>), cloudOutliers (new pcl::PointCloud<PointT>);

				// Create the filtering object
				pcl::ExtractIndices<PointT> extract;
				// Extract the inliers
				extract.setInputCloud (cloud);
				extract.setIndices (inliers);
				extract.setNegative (false);
				extract.filter (*cloudInliers);

				// Create the filtering object
				extract.setNegative (true);
				extract.filter (*cloudOutliers);

				segResult = std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>(cloudInliers, cloudOutliers);
			}

			// use TAPL
			else {
				tapl::pte::Plane<PointT> * plane = new tapl::pte::Plane<PointT>();
				std::unordered_set<int> inliers = plane->Ransac(cloud, maxIterations, distanceThreshold);

				typename pcl::PointCloud<PointT>::Ptr  cloudInliers(new pcl::PointCloud<PointT>());
				typename pcl::PointCloud<PointT>::Ptr cloudOutliers(new pcl::PointCloud<PointT>());

				for(int index = 0; index < cloud->points.size(); index++)
				{
					PointT point = cloud->points[index];
					if(inliers.count(index))
						cloudInliers->points.push_back(point);
					else
						cloudOutliers->points.push_back(point);
				}

				segResult = std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr>(cloudInliers, cloudOutliers);
			}

			return tapl::SUCCESS;
        }
        
        /**
         * @brief Perform euclidean clustering within a point-cloud. Returns a vector of cluster clouds.
         * 
         * @param[in] cloud point-cloud to segment
         * @param[in] clusterTolerance distance tolerance for considering a point
         *                              as part of the cluster.
         * @param[in] minNumPoints minimum number of points in a neighborhood to be considered
         *                      as a cluster
         * @param[in] maxNumPoints maximum number of points in a neighborhood to be considered
         *                      as a cluster
         * @param[in] usePCL Whether to use PCL of TAPL's implementation of RANSAC
         * 
         * @return Vector of cluster point-clouds
         */
        template<typename PointT>
		std::vector<typename pcl::PointCloud<PointT>::Ptr> euclideanClustering(typename pcl::PointCloud<PointT>::Ptr cloud, 
																				float clusterTolerance, 
																				int minNumPoints, 
																				int maxNumPoints, 
																				bool usePCL=false)
        {
            // clusters point-cloud to return
			std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

			// use PCL	
			if(usePCL) {
				// Creating the KdTree object for the search method of the extraction
				typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
				tree->setInputCloud (cloud);

				std::vector<pcl::PointIndices> clusterIndices;
				pcl::EuclideanClusterExtraction<PointT> ec;
				ec.setClusterTolerance (clusterTolerance);
				ec.setMinClusterSize (minNumPoints);
				ec.setMaxClusterSize (maxNumPoints);
				ec.setSearchMethod (tree);
				ec.setInputCloud (cloud);
				ec.extract (clusterIndices);

				for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin (); it != clusterIndices.end (); ++it)
				{
					typename pcl::PointCloud<PointT>::Ptr clusterCloud (new pcl::PointCloud<PointT>);
					for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) 
					{
						clusterCloud->points.push_back (cloud->points[*pit]);
					}
					clusterCloud->width = clusterCloud->points.size ();
					clusterCloud->height = 1;
					clusterCloud->is_dense = true;

					// store in the vector of clusters
					clusters.push_back(clusterCloud);
				}
			}

			// use TAPL
			else {
				// create k-d tree
				std::vector<std::vector<float>> points;
				tapl::pte::KdTree * tree = new tapl::pte::KdTree;
				for (int i=0; i<cloud->size(); i++)  {
					std::vector<float> point = {cloud->at(i).x, 
												cloud->at(i).y, 
												cloud->at(i).z};
					points.push_back(point);
					tree->insert(point,i); 
				}

				// Euclidean Clustering class
				tapl::pte::EuclideanCluster * ec = new tapl::pte::EuclideanCluster(points, tree);
				std::vector<std::vector<int>> clusterIDs = ec->clustering(clusterTolerance);

				for(std::vector<int> cluster : clusterIDs) {
					// check if the cluster size is within range
					if((cluster.size() >= minNumPoints) && (cluster.size() <= maxNumPoints)) {
						typename pcl::PointCloud<PointT>::Ptr clusterCloud (new pcl::PointCloud<PointT>);
						for(auto it = cluster.begin(); it != cluster.end(); ++it) {
							PointT p;
							p.x = points[*it][0];
							p.y = points[*it][1];
							p.z = points[*it][2];

							clusterCloud->push_back(p);
						}

						// store in the vector of clusters
						clusters.push_back(clusterCloud);
					}
				}
			}


			// return clusters
			return clusters;
        }
        
        /**
         * @brief This function is used to obtain bounding-box for a cluster of points.
         * 
         * @param[in] cloudCluster a cluster of point-cloud
         * 
         * @return 3D Bounding-Box
         */
        template<typename PointT>
		tapl::BBox3d getBoundingBox(typename pcl::PointCloud<PointT>::Ptr cloudCluster)
        {

			// Find bounding box for one of a cloud cluster
			PointT minPoint, maxPoint;
			pcl::getMinMax3D(*cloudCluster, minPoint, maxPoint);

			tapl::BBox3d bbox;
			bbox.x_min = minPoint.x;
			bbox.y_min = minPoint.y;
			bbox.z_min = minPoint.z;
			bbox.x_max = maxPoint.x;
			bbox.y_max = maxPoint.y;
			bbox.z_max = maxPoint.z;

			// return bounding-box
			return bbox;
        }
    };
};

#endif /* PT_ENGINE_H_ */
