#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "cvEngine.hpp"

using namespace std;

/*
 * Detect keypoints
 */
tapl::ResultCode tapl::cve::detectKeypoints(cv::Mat &img, 
                                std::vector<cv::KeyPoint> &keypoints, 
                                std::string detectorType) {
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("SHITOMASI") == 0) {
        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

        double qualityLevel = 0.01; // minimal accepted quality of image corners
        double k = 0.04;

        // Apply corner detection
        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

        // add corners to result vector
        for (auto it = corners.begin(); it != corners.end(); ++it)
        {

            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
            newKeyPoint.size = blockSize;
            keypoints.push_back(newKeyPoint);
        }
    }
    else if (detectorType.compare("HARRIS") == 0) {
        // Detector parameters
        int blockSize = 4;     // for every pixel, a blockSize × blockSize neighborhood is considered
        int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
        int minResponse = 80; // minimum value for a corner in the 8bit scaled response matrix
        double k = 0.04;       // Harris parameter (see equation for details)

        // Detect Harris corners and normalize output
        cv::Mat dst, dst_norm, dst_norm_scaled;
        dst = cv::Mat::zeros(img.size(), CV_32FC1);
        cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
        for (size_t j = 0; j < dst_norm.rows; j++)
        {
            for (size_t i = 0; i < dst_norm.cols; i++)
            {
                int response = (int)dst_norm.at<float>(j, i);
                if (response > minResponse)
                { // only store points above a threshold

                    cv::KeyPoint newKeyPoint;
                    newKeyPoint.pt = cv::Point2f(i, j);
                    newKeyPoint.size = 2 * apertureSize;
                    newKeyPoint.response = response;

                    // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                    bool bOverlap = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                        if (kptOverlap > maxOverlap)
                        {
                            bOverlap = true;
                            if (newKeyPoint.response > (*it).response)
                            {                      // if overlap is >t AND response is higher for new kpt
                                *it = newKeyPoint; // replace old key point with new one
                                break;             // quit loop over keypoints
                            }
                        }
                    }
                    if (!bOverlap)
                    {                                     // only add new key point if no overlap has been found in previous NMS
                        keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                    }
                }
            } 
        }    

    }
    else if (detectorType.compare("FAST") == 0) {
        int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("BRISK") == 0) {
        int threshold = 30;         // FAST/AGAST detection threshold score.
        int octaves = 3;            // detection octaves. Use 0 to do single scale.
        float patternScale = 1.0f;  // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        detector = cv::BRISK::create(threshold, octaves, patternScale);
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();       // using all default values
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();     // using all default values
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("FREAK") == 0) {
        detector = cv::xfeatures2d::FREAK::create();   // using all default values
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("SIFT") == 0) {
        detector = cv::xfeatures2d::SIFT::create();   // using all default values
        detector->detect(img, keypoints);
    }
    else
    {
        std::cout << "ERROR: This detector is not supported" << std::endl;
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}

/*
 * Extract descriptors
 */
tapl::ResultCode tapl::cve::extractDescriptors(cv::Mat &img, 
                                            std::vector<cv::KeyPoint> &keypoints, 
                                            cv::Mat &descriptors, 
                                            std::string descriptorType) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } 
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        // Use BRISK as the default descriptor
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }

    // perform feature description
    extractor->compute(img, keypoints, descriptors);

    return tapl::SUCCESS;
}

/*
 * Match keypoints descriptors 
 */
tapl::ResultCode tapl::cve::matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, 
                                            std::vector<cv::KeyPoint> &kPtsRef, 
                                            cv::Mat &descSource, 
                                            cv::Mat &descRef,
                                            std::vector<cv::DMatch> &matches, 
                                            std::string normType, 
                                            std::string matcherType, 
                                            std::string selectorType    ) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // create a copy of the descriptors so that any modification does not effect rest of the pipeline
    cv::Mat descSrcCopy, descRefCopy;
    descSource.copyTo(descSrcCopy);
    descRef.copyTo(descRefCopy);

    if (matcherType.compare("MAT_BF") == 0)
    {
        int norm = normType.compare("NORM_HAMMING") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(norm, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSrcCopy.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSrcCopy.convertTo(descSrcCopy, CV_32F);
            descRefCopy.convertTo(descRefCopy, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSrcCopy, descRefCopy, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSrcCopy, descRefCopy, knn_matches, 2); // finds the 2 best matches

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }

    }
    else
    {
        // use nearest-neighbor method as default
        matcher->match(descSrcCopy, descRefCopy, matches); // Finds the best match for each descriptor in desc1
    }

    // return success
    return tapl::SUCCESS;
}
                                
/**
 *  Detect and Match Keypoints; Store the result back into tapl::DataFrame 
 */
tapl::ResultCode tapl::cve::detectAndMatchKpts(tapl::DataFrame &dframe1, tapl::DataFrame &dframe2)
{
    // get images
    cv::Mat img1, img2;
    if(dframe1.cameraFrame.getImage(img1) == tapl::FAILURE) {
        std::cout << "ERROR: could not retrieve image" << std::endl;
        return tapl::FAILURE;
    }
    if(dframe2.cameraFrame.getImage(img2) == tapl::FAILURE) {
        std::cout << "ERROR: could not retrieve image" << std::endl;
        return tapl::FAILURE;
    }
    /* Detect Keypoints */
    // extract 2D keypoints from the images
    vector<cv::KeyPoint> keypoints1, keypoints2;    // create empty feature list for current image
        cv::Mat descriptors1, descriptors2;         // create empty feature descriptor list for current image
    
    if((tapl::cve::detectKeypoints(img1, keypoints1) == tapl::SUCCESS) &&
       (tapl::cve::detectKeypoints(img2, keypoints2) == tapl::SUCCESS) ) {
        
        /* Extract Keypoint Descriptors */
        if((tapl::cve::extractDescriptors(img1, keypoints1, descriptors1) == tapl::SUCCESS) &&
        (tapl::cve::extractDescriptors(img2, keypoints2, descriptors2) == tapl::SUCCESS) ) {
            // push descriptors into the frame
            dframe1.cameraFrame.pushDescriptors(descriptors1);
            dframe2.cameraFrame.pushDescriptors(descriptors2);
        }
        else {
            std::cout << "ERROR: Could not perform descriptors extraction" << std::endl;
            return tapl::FAILURE;
        }   
           
        // push keypoints into the frame
        dframe1.cameraFrame.pushKeypoints(keypoints1);
        dframe2.cameraFrame.pushKeypoints(keypoints2);
    }
    else {
        std::cout << "ERROR: Could not perform keypoint detection" << std::endl;
        return tapl::FAILURE;
    }

    /* Perform Keypoints Matching */
    vector<cv::DMatch> matches;
    if(tapl::cve::matchDescriptors(keypoints1, keypoints2, descriptors1, descriptors2, matches) == tapl::SUCCESS) {
        // store matches in current data frame
        dframe1.pushKptsMatches(matches);
    }
    else {
        std::cout << "ERROR: Could not perform descriptors matching" << std::endl;
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}

/**
 * This function computes Fundamental Matrix given two image frames
 */
tapl::ResultCode tapl::cve::computeFundamentalMatrix(tapl::DataFrame &dframe1, tapl::DataFrame &dframe2) {
    if(tapl::cve::detectAndMatchKpts(dframe1, dframe2) == tapl::SUCCESS) {
        /* Find Fundamental Matrix */
        // store matched keypoints as type Point2f
        std::vector<cv::Point2f> matched_points1;
        std::vector<cv::Point2f> matched_points2;
        std::vector<cv::KeyPoint> kpts1;
        std::vector<cv::KeyPoint> kpts2;

        // retrieve keypoints
        if((dframe1.cameraFrame.getKeypoints(kpts1) != tapl::SUCCESS) ||
            (dframe2.cameraFrame.getKeypoints(kpts2) != tapl::SUCCESS)   ) {
                std::cout << "ERROR: could not retrieve keypoints" << std::endl;
                return tapl::FAILURE;
        }

        // retrieve keypoint matches
        std::vector<cv::DMatch> kptsMatches;
        if(dframe1.getKptsMatches(kptsMatches) == tapl::SUCCESS) {
            for (auto matches: kptsMatches) {
                matched_points1.push_back(kpts1[matches.queryIdx].pt);
                matched_points2.push_back(kpts2[matches.trainIdx].pt);
            }
            cv::Mat F = cv::findFundamentalMat(matched_points1, matched_points2, cv::FM_RANSAC, 3, 0.99);
            // store the fundamental matrix in current data frame
            dframe1.pushFundamentalMatrix(F);
        }
    }
    else {
        std::cout << "ERROR: could not perform keypoint detection and matching" << std::endl;
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
} 

/**
 * This function computes Essential Matrix given two image frames
 */
tapl::ResultCode tapl::cve::computeEssentialMatrix(tapl::DataFrame &dframe1, 
                                                tapl::DataFrame &dframe2, 
                                                cv::Mat &camera_matrix) {
    if(tapl::cve::detectAndMatchKpts(dframe1, dframe2) == tapl::SUCCESS) {
        /* Find Fundamental Matrix */
        // store matched keypoints as type Point2f
        std::vector<cv::Point2f> matched_points1;
        std::vector<cv::Point2f> matched_points2;
        std::vector<cv::KeyPoint> kpts1;
        std::vector<cv::KeyPoint> kpts2;

        // retrieve keypoints
        if((dframe1.cameraFrame.getKeypoints(kpts1) != tapl::SUCCESS) ||
            (dframe2.cameraFrame.getKeypoints(kpts2) != tapl::SUCCESS)   ) {
                std::cout << "ERROR: could not retrieve keypoints" << std::endl;
                return tapl::FAILURE;
        }

        // retrieve keypoint matches
        std::vector<cv::DMatch> kptsMatches;
        if(dframe1.getKptsMatches(kptsMatches) == tapl::SUCCESS) {
            for (auto matches: kptsMatches) {
                matched_points1.push_back(kpts1[matches.queryIdx].pt);
                matched_points2.push_back(kpts2[matches.trainIdx].pt);
            }
            cv::Mat E = cv::findEssentialMat(matched_points1, matched_points2, camera_matrix, cv::RANSAC, 0.99, 2.0);
            // store the fundamental matrix in current data frame
            dframe1.pushEssentialMatrix(E);
        }
    }
    else {
        std::cout << "ERROR: could not perform keypoint detection and matching" << std::endl;
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}


/**
 * Rodrigues to Euler angle conversion
 */
void rodrigues2euler(cv::Mat &R, float &roll, float &pitch, float &yaw) {
    float cosine_for_pitch = sqrt(pow(R.at<float>(0,0), 2) + pow(R.at<float>(1,0), 2));
    bool is_singular = false;
    if(cosine_for_pitch < 10e-6) is_singular = true;
    if(is_singular == false) {
        roll = atan2(R.at<float>(2,1), R.at<float>(2,2));
        pitch = atan2(-R.at<float>(2,0), cosine_for_pitch);
        yaw = atan2(R.at<float>(1,0), R.at<float>(0,0));
    }
    else {
        roll = 0;
        pitch = atan2(-R.at<float>(2,0), cosine_for_pitch);
        yaw = atan2(-R.at<float>(1,2), R.at<float>(1,1));
    }
}

/**
 * This function computes relative camera pose between set of image frames
 */
tapl::ResultCode tapl::cve::computeRelativePose(tapl::DataFrame &dframe1, 
                                                tapl::DataFrame &dframe2, 
                                                cv::Mat &camera_matrix) {
    if(tapl::cve::detectAndMatchKpts(dframe1, dframe2) == tapl::SUCCESS) {
        std::vector<cv::KeyPoint> kpts1;
        std::vector<cv::KeyPoint> kpts2;

        // retrieve keypoints
        if((dframe1.cameraFrame.getKeypoints(kpts1) != tapl::SUCCESS) ||
            (dframe2.cameraFrame.getKeypoints(kpts2) != tapl::SUCCESS)   ) {
                std::cout << "ERROR: could not retrieve keypoints" << std::endl;
                return tapl::FAILURE;
        }

        // retrieve keypoint matches
        std::vector<cv::DMatch> kptsMatches;
        if(dframe1.getKptsMatches(kptsMatches) == tapl::SUCCESS) {
            // store matched keypoints as type Point2f
            std::vector<cv::Point2f> currMatch;
            std::vector<cv::Point2f> prevMatch;
            for (auto matches: kptsMatches) {
                currMatch.push_back(kpts1[matches.queryIdx].pt);
                prevMatch.push_back(kpts2[matches.trainIdx].pt);
            }

            cv::Mat mask;
            cv::Mat essential_matrix = cv::findEssentialMat(currMatch, prevMatch, camera_matrix, cv::RANSAC, 0.99, 2.0, mask);

            cv::Mat R, t;
            cv::recoverPose(essential_matrix, currMatch, prevMatch, camera_matrix, R, t, 50, mask);

            std::vector<cv::Point2d> triangulation_points1, triangulation_points2;
            for(int i = 0; i < mask.rows; i++) {
                if(mask.at<unsigned char>(i)){
                triangulation_points1.push_back 
                            (cv::Point2d((double)currMatch[i].x,(double)currMatch[i].y));
                triangulation_points2.push_back 
                            (cv::Point2d((double)prevMatch[i].x,(double)prevMatch[i].y));
                }
            }
            cv::Mat P1 = cv::Mat::eye(3, 4, CV_64FC1);
            cv::Mat P2 = cv::Mat::eye(3, 4, CV_64FC1);
            cv::Mat Rinv = R.inv();
            R.copyTo(P1.rowRange(0,3).colRange(0,3));
            t.copyTo(P1.rowRange(0,3).col(3));
            cv::Mat triangulated_points;
            if((triangulation_points1.size() > 0) && (triangulation_points2.size() > 0)) {
                cv::triangulatePoints(camera_matrix * P2, camera_matrix * P1, 
                                        triangulation_points1, triangulation_points2,
                                        triangulated_points);
            }

            /* compute pose */
            tapl::Pose6dof pose;
            pose.R = Rinv;
            pose.t = t;
            pose.R.copyTo(pose.P(cv::Rect(0, 0, 3, 3)));
            pose.t.copyTo(pose.P(cv::Rect(3, 0, 1, 3)));

            // convert rodrigues to euler angle
            float roll, pitch, yaw;
            rodrigues2euler(R, roll, pitch, yaw);
            cv::Mat euler(1, 3, CV_32FC1);
            euler.at<float>(0,0) = roll;
            euler.at<float>(0,1) = pitch;
            euler.at<float>(0,2) = yaw;
            pose.euler = euler;

            dframe1.pushPose(pose);
            dframe1.pushTriangulatedPts(triangulated_points);
        }

        // return success
        return tapl::SUCCESS;
    }
    else {
        // return failure
        return tapl::FAILURE;
    }
}

/*
 * This function is used to stitch multiple images as a panaromic image.
 */
tapl::ResultCode tapl::cve::stitchPanaromic(const std::vector<cv::Mat> &imgs, 
                                         cv::Mat &panoramic_img) {
    // sanity check 
    if(imgs.size() < 2) {
        std::cout << "ERROR: Number of images must be 2 or more" << std::endl;
        return tapl::FAILURE;
    }

    // create panaromic stitcher
    bool divide_imgs = false;
    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = stitcher->stitch(imgs, panoramic_img);

    // check if successful
    if (status != cv::Stitcher::OK) {
        std::cout << "ERROR: could not stitch images" << std::endl;
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}