/**
 * @file    cvEngine.hpp
 * @brief   This file provides APIs for computer vision related functions.
 * @author  Shubham Shrivastava
 */

#ifndef CV_ENGINE_H_
#define CV_ENGINE_H_

#include "taplTypes.hpp"

namespace tapl {
    namespace cve {
        /** 
         * @brief This function detects keypoints in an image
         *
         * @param[in] img image for which keypoints need to be detected
         * @param[out] keypoints detected keypoints
         * @param[in] detectorType detector types; SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode detectKeypoints(cv::Mat &img, 
                                            std::vector<cv::KeyPoint> &keypoints, 
                                            std::string detectorType = "FAST");

        /** 
         * @brief This function extracts keypoints descriptors in an image
         *
         * @param[in] img image for which keypoints need to be detected
         * @param[in] keypoints keypoints
         * @param[out] descriptors output descriptors
         * @param[in] descriptorType descriptor types; options: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT 
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode extractDescriptors(cv::Mat &img, 
                                            std::vector<cv::KeyPoint> &keypoints, 
                                            cv::Mat &descriptors, 
                                            std::string descriptorType = "BRISK");

        /** 
         * @brief This function performs keypoint descriptor matching
         *
         * @param[in] kPtsSource keypoints in the source/query image
         * @param[in] kPtsRef keypoints in the reference/train image
         * @param[in] descSource descriptors in the source/query image
         * @param[in] descRef descriptors in the reference/train image
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
        tapl::ResultCode matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, 
                                            std::vector<cv::KeyPoint> &kPtsRef, 
                                            cv::Mat &descSource, 
                                            cv::Mat &descRef,
                                            std::vector<cv::DMatch> &matches, 
                                            std::string normType = "NORM_HAMMING", 
                                            std::string matcherType = "MAT_BF", 
                                            std::string selectorType = "SEL_KNN");
        /** 
         * @brief This function detects keypoints in two image frames
         *          and perform keypoints matching.
         *
         * @param[in,out] dframe1 first data frame - query/source/current frame
         * @param[in,out] dframe2 second data frame - train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode detectAndMatchKpts(tapl::DataFrame &dframe1, tapl::DataFrame &dframe2);

        /** 
         * @brief This function retrieves fundamental matrix between two images
         *          contained within their data frame structure
         *
         * @param[in,out] dframe1 first data frame - query/source/current frame
         * @param[in,out] dframe1 second data frame - train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure  
         */
        tapl::ResultCode computeFundamentalMatrix(tapl::DataFrame &dframe1, tapl::DataFrame &dframe2);

        /** 
         * @brief This function retrieves essential matrix between two images
         *          contained within their data frame structure
         *
         * @param[in,out] dframe1 first data frame - query/source/current frame
         * @param[in,out] dframe1 second data frame - train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode computeEssentialMatrix(tapl::DataFrame &dframe1, 
                                                tapl::DataFrame &dframe2, 
                                                cv::Mat &camera_matrix);

        /** 
         * @brief This function is used to compute the relative camera pose. Pose is computed
         *          for second image contained within dframe2 relative to dframe1.
         *
         * @param[in,out] dframe1 first data frame - query/source/current frame
         * @param[in,out] dframe1 second data frame - train/reference/previous frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode computeRelativePose(tapl::DataFrame &dframe1, 
                                            tapl::DataFrame &dframe2, 
                                            cv::Mat &camera_matrix);

        /** 
         * @brief This function is used to stitch multiple images as a panaromic image.
         *
         * @param[in] imgs list of images
         * @param[out] panoramic_img stitched panoramic image
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode stitchPanaromic(const std::vector<cv::Mat> &imgs, 
                                         cv::Mat &panoramic_img);
    } 
} 

#endif /* CV_ENGINE_H_ */