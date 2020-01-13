/**
 * @file    visualization.hpp
 * @brief   This file provides APIs for visualization
 * @author  Shubham Shrivastava
 */

#ifndef VISUALIZATION_H_
#define VISUALIZATION_H_

#include <pcl/visualization/pcl_visualizer.h>

namespace tapl {
    namespace viz {
        /** 
         * @brief Enumerations for camera view angle
         */
        enum CameraAngle {
            XY, TopDown, Side, FPS
        };

        /** 
         * @brief Implementation of the visualizer class
         */
        class Visualizer {
            private:
                // visualizer
                pcl::visualization::PCLVisualizer * viewer; 

            public:
                // constructor s
                Visualizer(double r = 0.5, double g = 0.5, double b = 0.5, CameraAngle camAngle = TopDown, double distance = 20.0) :
                    viewer(new pcl::visualization::PCLVisualizer ("3D Viewer")) {
                    // set up visualizer
                    viewer->setBackgroundColor (r, g, b);
                    viewer->initCameraParameters ();

                    // set camera angle
                    switch(camAngle) {
                        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
                        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
                        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
                        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
                    }
                } 

                // de-constructor 
                ~Visualizer() {}

                /** 
                 * @brief This function is used to render a sphere on the visualizer
                 *
                 * @param[in] pt center point of the sphere
                 * @param[in] radius radius of the sphere
                 * @param[in] id a unique id for this sphere
                 */
                template <typename PointT>
                void renderSphere(const PointT& pt, double radius, const std::string &id)
                {
                    viewer->addSphere (pt, radius, id);
                }

                /** 
                 * @brief This function is used to render a sphere on the visualizer
                 *
                 * @param[in] pt center point of the sphere
                 * @param[in] radius radius of the sphere
                 * @param[in] r specifies red in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] g specifies green in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] b specifies blue in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] id a unique id for this sphere
                 */
                template <typename PointT>
                void renderSphere(const PointT& pt, double radius, double r, double g, double b, const std::string &id)
                {
                    viewer->addSphere (pt, radius, r, g, b, id);
                }

                /** 
                 * @brief This function is used to render a line on the visualizer
                 *
                 * @param[in] pt1 start point of the line
                 * @param[in] pt2 end point of the line
                 * @param[in] id a unique id for this line
                 */
                template <typename PointT>
                void renderLine(const PointT &pt1, const PointT &pt2, const std::string &id)
                {
                    viewer->addLine(pt1, pt2, id);
                }

                /** 
                 * @brief This function is used to render a line on the visualizer
                 *
                 * @param[in] pt1 start point of the line
                 * @param[in] pt2 end point of the line
                 * @param[in] r specifies red in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] g specifies green in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] b specifies blue in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] id a unique id for this line
                 */
                template <typename PointT>
                void renderLine(const PointT &pt1, const PointT &pt2, double r, double g, double b, const std::string &id)
                {
                    viewer->addLine(pt1, pt2, r, g, b, id);
                }

                /** 
                 * @brief Adds 3-DOF pose.
                 * @param[in] scale the scale of the axes
                 * @param[in] x x-coordinate
                 * @param[in] y y-coordinate
                 * @param[in] z z-coordinate
                 * @param[in] id the coordinate system object id (default: pose)
                 */
                void renderPose(double scale, double x, double y, double z, const std::string &id = "pose")
                {
                    viewer->addCoordinateSystem(scale, x, y, z, id);
                }

                /** 
                 * @brief Adds 6-DOF pose defined by a 4x4 transformation matrix.
                 * @param[in] scale the scale of the axes
                 * @param[in] pose the pose to be rendered. 4x4 transformation matrix 
                 *                  combines a 3x3 rotation matrix and a 3x1 translation matrix
                 * @param[in] id the coordinate system object id (default: pose)
                 */
                void renderPose(double scale, const Eigen::Affine3f& pose, const std::string &id = "pose")
                {
                    viewer->addCoordinateSystem(scale, pose, id);
                }

                /** 
                 * @brief Render a point-cloud
                 * @param[in] cloud point-cloud to be rendered
                 * @param[in] ptsize size of the point on visualizer to be rendered
                 * @param[in] r specifies red in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] g specifies green in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] b specifies blue in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] id the point-cloud object id (default: pose)
                 */
                template <typename PointT>
                void renderPointCloud(const typename pcl::PointCloud<PointT>::Ptr& cloud, double ptsize=1.0, double r=1.0, double g=0.0, double b=0.0, const std::string &id = "cloud")
                {
                    viewer->addPointCloud(cloud, id);
                    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                ptsize,
                                                                id);
                    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR,
                                                                r, g, b,
                                                                id);
                }

                /** 
                 * @brief Render the scene once.
                 * @param[in] time - How long (in ms) should the visualization 
                 *                      loop be allowed to run.
                 */
                void renderScene(int time = 1)
                {
                    viewer->spinOnce(time);
                }

                /** 
                 * @brief Render the scene and hold.
                 */
                void renderSceneAndHold()
                {
                    while (!viewer->wasStopped ()) viewer->spin();
                }
        };
    };
};

#endif