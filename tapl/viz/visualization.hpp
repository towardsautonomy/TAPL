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
            XY,         /**< Angled XY Camera Position*/
            TopDown,    /**< Top Down View */
            Side,       /**< Side View */
            FPS         /**< First Person View */
        };

        /** 
         * @brief Implementation of the visualizer class
         */
        class Visualizer {
            private:
                // visualizer
                pcl::visualization::PCLVisualizer * viewer; 

            public:
                // constructor 
                Visualizer(float r = 0., float g = 0., float b = 0., CameraAngle camAngle = TopDown, float distance = 20.0) :
                    viewer(new pcl::visualization::PCLVisualizer ("3D Viewer")) {
                    // set up visualizer
                    viewer->setBackgroundColor (r, g, b);
                    viewer->initCameraParameters ();

                    // set camera angle
                    switch(camAngle) {
                        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
                        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
                        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
                        case FPS : viewer->setCameraPosition(-distance, 0, distance, 0, 0, 1); break;
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
                void renderSphere(const PointT& pt, float radius, const std::string &id)
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
                void renderSphere(const PointT& pt, float radius, float r, float g, float b, const std::string &id)
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
                void renderLine(const PointT &pt1, const PointT &pt2, float r, float g, float b, const std::string &id)
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
                void renderPose(float scale, float x, float y, float z, const std::string &id = "pose")
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
                void renderPose(float scale, const Eigen::Affine3f& pose, const std::string &id = "pose")
                {
                    viewer->addCoordinateSystem(scale, pose, id);
                }

                /** 
                 * @brief Render a point-cloud
                 * @param[in] cloud point-cloud to be rendered
                 * @param[in] ptsize size of the point on visualizer to be rendered
                 * @param[in] r specifies red in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0];
                 *              do not render specified color if it is set to -1.0
                 * @param[in] g specifies green in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 *              do not render specified color if it is set to -1.0
                 * @param[in] b specifies blue in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 *              do not render specified color if it is set to -1.0
                 * @param[in] id the point-cloud object id (default: cloud)
                 */
                template <typename PointT>
                void renderPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, float ptsize=1.0, float r=1.0, float g=0.0, float b=0.0, const std::string &id = "cloud")
                {
                    viewer->addPointCloud(cloud, id);
                    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                                ptsize,
                                                                id);
                    if ((r != -1.0) && (g != -1.0) && (g != -1.0)) {
                        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR,
                                                                    r, g, b,
                                                                    id);
                    }
                }

                // /** 
                //  * @brief Render a 3d bounding-box
                //  * @param[in] box 3D Bounding-Box
                //  * @param[in] r specifies red in rgb colorspace for the 
                //  *              color of sphere in range of [0.0, 1.0]
                //  * @param[in] g specifies green in rgb colorspace for the 
                //  *              color of sphere in range of [0.0, 1.0]
                //  * @param[in] b specifies blue in rgb colorspace for the 
                //  *              color of sphere in range of [0.0, 1.0]
                //  * @param[in] opacity opacity of the bounding box in range of [0.0, 1.0]
                //  * @param[in] id the bounding-box object id (default: bbox)
                //  */
                // void renderBbox3d(const BBox3d& box, 
                //                   float r=1.0, float g=0.0, float b=0.0, 
                //                   float opacity=0.5, const std::string &id = "bbox")
                // {
                //     if(opacity > 1.0)
                //         opacity = 1.0;
                //     if(opacity < 0.0)
                //         opacity = 0.0;

                //     double x_min = box.x_center - (box.length/2.0);
                //     double x_max = box.x_center + (box.length/2.0);
                //     double y_min = box.y_center - (box.width/2.0);
                //     double y_max = box.y_center + (box.width/2.0);
                //     double z_min = box.z_center - (box.height/2.0);
                //     double z_max = box.z_center + (box.height/2.0);
                //     viewer->addCube(x_min, x_max, y_min, y_max, z_min, z_max, r, g, b, id);
                //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
                //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
                //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id);

                //     std::string cubeFill = "fill_"+id;
                //     viewer->addCube(x_min, x_max, y_min, y_max, z_min, z_max, r, g, b, cubeFill);
                //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, cubeFill);
                //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, cubeFill);
                //     viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity*0.3, cubeFill);
                // }

                /** 
                 * @brief Render an oriented 3d bounding-box
                 * @param[in] box 3D Bounding-Box
                 * @param[in] r specifies red in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] g specifies green in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] b specifies blue in rgb colorspace for the 
                 *              color of sphere in range of [0.0, 1.0]
                 * @param[in] opacity opacity of the bounding box in range of [0.0, 1.0]
                 * @param[in] id the bounding-box object id (default: bbox)
                 */
                void renderBbox3d(const BBox3d& box, 
                                  float r=1.0, float g=0.0, float b=0.0, 
                                  float opacity=0.5, const std::string &id = "bbox")
                {
                    if(opacity > 1.0)
                        opacity = 1.0;
                    if(opacity < 0.0)
                        opacity = 0.0;

                    auto rot = Eigen::Quaternionf::Identity();
                    viewer->addCube(Eigen::Vector3f(box.x, box.y, box.z), 
                                    Eigen::Quaternionf(box.rotation.x, box.rotation.y, box.rotation.z, box.rotation.w),
                                    box.length, box.width, box.height, id);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, id);

                    std::string cubeFill = "fill_"+id;
                    viewer->addCube(Eigen::Vector3f(box.x, box.y, box.z), 
                                    Eigen::Quaternionf(box.rotation.x, box.rotation.y, box.rotation.z, box.rotation.w),
                                    box.length, box.width, box.height, cubeFill);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, cubeFill);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, cubeFill);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity*0.3, cubeFill);
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

                /** 
                 * @brief Clear the scene
                 */
                void clearScene()
                {
                    viewer->removeAllPointClouds();
                    viewer->removeAllShapes();
                }
        };
    };
};

#endif