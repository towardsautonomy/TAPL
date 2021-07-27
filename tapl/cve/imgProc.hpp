/**
 * @file    imgProc.hpp
 * @brief   This file provides APIs for computer vision related functions.
 * @author  Shubham Shrivastava
 */

#ifndef IMG_PROC_H_
#define IMG_PROC_H_

#include "tapl/common/common.hpp"
#include <opencv2/imgproc.hpp>

namespace tapl {
    namespace cve {

        /**
         * @brief Draw 2D keypoints on an image
         * 
         * @param[in] image image for which the detected 2D keypoints need to be visualized
         * @param[in] kpts2d 2D keypoints
         * @param[in] color color for 2D keypoints
         *
         * @return image with 2d keypoints overlaid
         */
        cv::Mat drawKeypoints( const cv::Mat &image, 
                               const std::vector<tapl::Point2d> &kpts2d,
                               cv::Scalar color=CV_RGB(255,0,0));

        /**
         * @brief Draw 3D keypoints on an image by projecting it to the 2D plane
         * 
         * @param[in] image image for which the detected 2D keypoints need to be visualized
         * @param[in] kpts3d 3D keypoints
         * @param[in] K Camera intrinsic matrix
         * @param[in] color color for 2D keypoints
         *
         * @return image with 3d keypoints overlaid
         */
        cv::Mat drawKeypoints3D( const cv::Mat &image, 
                                 const std::vector<tapl::Point3d> &kpts3d,
                                 const cv::Mat &K,
                                 cv::Scalar color=CV_RGB(255,0,0));

        /**
         * @brief Draw 3D keypoints on an image by projecting it to the 2D plane
         * 
         * @param[in] image image for which the detected 2D keypoints need to be visualized
         * @param[in] kpts3d 3D keypoints
         * @param[in] K Camera intrinsic matrix
         * @param[in] color color for 2D keypoints
         *
         * @return image with 3d keypoints overlaid
         */
        cv::Mat drawKeypoints3D( const cv::Mat &image, 
                                 const std::vector<tapl::Point3dColor> &kpts3d,
                                 const cv::Mat &K,
                                 cv::Scalar color=CV_RGB(255,0,0));

        /** 
         * @brief This function detects keypoints in an image
         *
         * @param[in] img image for which keypoints need to be detected
         * @param[out] keypoints detected keypoints
         * @param[in] detectorType detector types; SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode detectKeypoints( const cv::Mat &img, 
                                          std::vector<cv::KeyPoint> &keypoints, 
                                          std::string detectorType = "BRISK" );

        /** 
         * @brief This function extracts keypoints descriptors in an image
         *
         * @param[in] img image for which keypoints need to be detected
         * @param[in] keypoints keypoints
         * @param[out] descriptors output descriptors
         * @param[in] descriptorType descriptor types; options: BRISK, BRIEF, ORB, FREAK, AKAZE 
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode extractDescriptors( const cv::Mat &img, 
                                             std::vector<cv::KeyPoint> &keypoints, 
                                             cv::Mat &descriptors, 
                                             std::string descriptorType = "BRISK" );

        /** 
         * @brief This function performs keypoint descriptor matching
         *
         * @param[in] kPtsQuery keypoints in the query image
         * @param[in] kPtsTrain keypoints in the trainimage
         * @param[in] descQuery descriptors in the query image
         * @param[in] descRef descriptors in the trainimage
         * @param[out] matches descriptor match output
         * @param[in] normType norm types; options: NORM_HAMMING, NORM_L2
         *              NORM_HAMMING : Hamming Distance
         *              L2_HAMMING  : L2 Distance
         * @param[in] matcherType types; options: MAT_BF, MAT_FLANN
         *              MAT_BF      : Brute-Force Matching
         *              MAT_FLANN   : FLANN based matching  
         * @param[in] selectorType types; options: SEL_NN, SEL_KNN
         *              SEL_NN      : Nearest-Neighbor
         *              SEL_KNN     : K-Nearest-Neighbor
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode matchDescriptors( std::vector<cv::KeyPoint> &kPtsQuery, 
                                           std::vector<cv::KeyPoint> &kPtsTrain, 
                                           cv::Mat &descQuery, 
                                           cv::Mat &descRef,
                                           std::vector<cv::DMatch> &matches, 
                                           std::string normType = "NORM_HAMMING", 
                                           std::string matcherType = "MAT_BF", 
                                           std::string selectorType = "SEL_KNN" );
        /** 
         * @brief This function detects keypoints in two image frames
         *          and perform keypoints matching.
         *
         * @param[in,out] camPairs camera frames pair with their properties 
         *                              - first: query/source/current frame
         *                              - second: train/reference/previous frame
         * @param[in] detectorType detector types; SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE
         * @param[in] descriptorType descriptor types; options: BRISK, BRIEF, ORB, FREAK, AKAZE 
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode detectAndMatchKpts( tapl::CameraPairs &camPairs,
                                             const std::string detectorType = "BRISK",
                                             const std::string descriptorType = "BRISK" );

        /** 
         * @brief This function retrieves fundamental matrix between two images
         *          contained within their data frame structure
         *
         * @param[in,out] camPairs camera frames pair with their properties 
         *                              - first: query/source/current frame
         *                              - second: train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure  
         */
        tapl::ResultCode computeFundamentalMatrix( tapl::CameraPairs &camPairs );

        /** 
         * @brief This function retrieves essential matrix between two images
         *          contained within their data frame structure
         *
         * @param[in,out] camPairs camera frames pair with their properties 
         *                              - first: query/source/current frame
         *                              - second: train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode computeEssentialMatrix( tapl::CameraPairs &camPairs );

        /** 
         * @brief This function is used to compute the relative camera pose. Pose is computed
         *          for second image contained within dframe2 relative to dframe1.
         *
         * @param[in,out] camPairs camera frames pair with their properties 
         *                              - first: query/source/current frame
         *                              - second: train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode computeRelativePose( tapl::CameraPairs &camPairs );

        /** 
         * @brief This function is used to stitch multiple images as a panaromic image.
         *
         * @param[in] imgs list of images
         * @param[out] panoramic_img stitched panoramic image
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode stitchPanaromic( const std::vector<cv::Mat> &imgs, 
                                          cv::Mat &panoramic_img);
    } 
} 

#endif /* IMG_PROC_H_ */