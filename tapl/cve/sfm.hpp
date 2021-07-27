/**
 * @file    sfm.hpp
 * @brief   This provides the implementation of structure-from-motion algorithm
 * @author  Shubham Shrivastava
 */

#ifndef SFM_H_
#define SFM_H_

#include <Eigen/Dense>
#include "tapl/common/common.hpp"
#include "tapl/optim/gaussNewton.hpp"
#include "tapl/viz/visualization.hpp"

namespace tapl {
    namespace cve {

        /**< Structure-from-Motion Pipeline */
        class StructureFromMotion {
        private:
            // vector of images
            std::vector<cv::Mat> images;
            // camera intrinsic matrix
            cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
            // camera distortion matrix
            cv::Mat dist_coeff = cv::Mat::zeros(4, 1, CV_32FC1);
            // triangulated 3D points in each image local coordinate systems
            std::vector<std::vector<tapl::Point3d>> points3d_local;
            // triangulated 3D points in the origin's (first image) coordinate systems
            std::vector<tapl::Point3d> points3d_global;
            // bundle size
            uint16_t bundle_size=2;
            // Gauss-Newton Optimizer
            tapl::optim::GaussNewtonOptimizer gnOptim;

            /**
             * @brief Linear estimate of the 3d point
             * 
             * @param[in] point2d projection of the same 3d point in 'n' cameras
             * @param[in] projectionMatrices projection matrices of 'n' cameras
             *
             * @return estimated 3d point
             */
            tapl::Point3d linearEstimate3dPt( const std::vector<tapl::Point2d> &point2d,                                         
                                              const std::vector<Eigen::MatrixXd> &projectionMatrices );

            /**
             * @brief Non-linear estimate of the 3d point
             *
             * @param[in] point2d projection of the same 3d point in 'n' cameras
             * @param[in] projectionMatrices projection matrices of 'n' cameras
             * @param[out] reprojectionErrors pair of pre-optimization and post-optimization 
             *                                  reprojection errors
             * @param[in] nIterations maximum number of iterations for non-linear optimization
             * @param[in] reprErrorThresh reprojection error threshold for non-linear optimization
             *
             * @return estimated 3d point
             */
            tapl::Point3d nonLinearEstimate3dPt( const std::vector<tapl::Point2d> &point2d,                                         
                                                 const std::vector<Eigen::MatrixXd> &projectionMatrices,
                                                 std::pair<std::vector<float>,std::vector<float>> &reprojectionErrors,
                                                 const uint16_t nIterations=1000,
                                                 const float reprErrorThresh=2.0 );

            /**
             * @brief Estimate R, T, and triengulated points
             *
             * @param[in] E essential matrix relating the first and the second camera
             * @param[in] points2d projection of the same 3d point in 'n' cameras
             * @param[in] projectionMat1 projection matrices of the first camera
             * @param[in] maxXYZ maximum values of x, y, and z to be considered
             * @param[in] maxReprojectionErr maximum absolute reprojection error in pixel
             *
             * @return pair of RT and triangulated points
             */
             std::pair<Eigen::MatrixXd, std::vector<tapl::Point3d>> computeSFM( 
                                        const Eigen::MatrixXd &E, 
                                        const std::vector<std::vector<tapl::Point2d>> &points2d, 
                                        const Eigen::MatrixXd &projectionMat1,
                                        const std::vector<float> &maxXYZ={30.0, 30.0, 30.0},
                                        const float &maxReprojectionErr=20.0 ) ;

        public:
            /** 
            * @brief This function initializes the structure-from-motion module
            *
            * @param[in] imgs images from which structure-from-motion is to be computed
            * @param[in] K camera intrinsic matrix
            */
            StructureFromMotion( const std::vector<cv::Mat> &images, 
                                 const cv::Mat &K );

            /** 
            * @brief This function performs structure-from-motion given a set of camera frames
            *
            * @param[out] points point-cloud corresponding to keypoints in the first camera's coordinate frame
            * @param[out] poses poses of each camera frame
            * @param[out] framePairs camera pairs with associated info such as triangulated points
            * 
            * @return tapl::SUCCESS if success
            * @return tapl::FAILURE if failure 
            */
            tapl::ResultCode process( std::vector<tapl::Point3dColor> &points,
                                      std::vector<tapl::Pose6dof> &poses,
                                      std::vector<tapl::CameraPairs> &framePairs);
        
        };
    } 
} 

#endif /* SFM_H_ */