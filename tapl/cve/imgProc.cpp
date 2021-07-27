#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "imgProc.hpp"

/*
 * draw 2D keypoints on an image
 */
cv::Mat tapl::cve::drawKeypoints(
                        const cv::Mat &image, 
                        const std::vector<tapl::Point2d> &kpts2d,
                        cv::Scalar color) {
    cv::Mat img_copy;
    image.copyTo(img_copy);
    for(auto& kp: kpts2d) {
        cv::Point2f pt2d = cv::Point2f(kp.x, kp.y);
        cv::circle(img_copy, pt2d, 4, CV_RGB(255,255,255), -1);
        cv::circle(img_copy, pt2d, 3, color, -1);
    }
    return img_copy;
}

/*
 * draw 3D keypoints on an image by projecting it to the 2D plane
 */
cv::Mat tapl::cve::drawKeypoints3D(
                        const cv::Mat &image, 
                        const std::vector<tapl::Point3d> &kpts3d,
                        const cv::Mat &K,
                        cv::Scalar color) {
    cv::Mat img_copy;
    image.copyTo(img_copy);
    for(auto& kp: kpts3d) 
    {
        cv::Mat pt3d = cv::Mat::zeros(3,1, CV_64FC1);
        pt3d.at<double>(0,0) = kp.x;
        pt3d.at<double>(1,0) = kp.y;
        pt3d.at<double>(2,0) = kp.z;
        K.convertTo(K, CV_64FC1);
        cv::Mat pt2d_homogeneous = K * pt3d;
        // std::cout << pt2d_homogeneous << std::endl;
        cv::Point2f pt2d;
        pt2d.x = pt2d_homogeneous.at<double>(0,0) / pt2d_homogeneous.at<double>(2,0);
        pt2d.y = pt2d_homogeneous.at<double>(1,0) / pt2d_homogeneous.at<double>(2,0);

        cv::circle(img_copy, pt2d, 4, CV_RGB(255,255,255), -1);
        cv::circle(img_copy, pt2d, 3, color, -1);
    }
    return img_copy;
}

/*
 * draw 3D keypoints on an image by projecting it to the 2D plane
 */
cv::Mat tapl::cve::drawKeypoints3D(
                        const cv::Mat &image, 
                        const std::vector<tapl::Point3dColor> &kpts3d,
                        const cv::Mat &K,
                        cv::Scalar color) {
    cv::Mat img_copy;
    image.copyTo(img_copy);
    for(auto& kp: kpts3d) 
    {
        cv::Mat pt3d = cv::Mat::zeros(3,1, CV_64FC1);
        pt3d.at<double>(0,0) = kp.x;
        pt3d.at<double>(1,0) = kp.y;
        pt3d.at<double>(2,0) = kp.z;
        K.convertTo(K, CV_64FC1);
        cv::Mat pt2d_homogeneous = K * pt3d;
        // std::cout << pt2d_homogeneous << std::endl;
        cv::Point2f pt2d;
        pt2d.x = pt2d_homogeneous.at<double>(0,0) / pt2d_homogeneous.at<double>(2,0);
        pt2d.y = pt2d_homogeneous.at<double>(1,0) / pt2d_homogeneous.at<double>(2,0);

        cv::circle(img_copy, pt2d, 4, CV_RGB(255,255,255), -1);
        cv::circle(img_copy, pt2d, 3, color, -1);
    }
    return img_copy;
}

/*
 * Detect keypoints
 */
tapl::ResultCode tapl::cve::detectKeypoints( const cv::Mat &img, 
                                             std::vector<cv::KeyPoint> &keypoints, 
                                             std::string detectorType ) {
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("SHITOMASI") == 0) {
        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

        double qualityLevel = 0.01; // minimal accepted quality of image corners
        double k = 0.04;

        // Apply corner detection
        std::vector<cv::Point2f> corners;
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
        for (size_t j = 0; j < dst_norm.rows; j++) {
            for (size_t i = 0; i < dst_norm.cols; i++) {
                int response = (int)dst_norm.at<float>(j, i);
                if (response > minResponse) { // only store points above a threshold

                    cv::KeyPoint newKeyPoint;
                    newKeyPoint.pt = cv::Point2f(i, j);
                    newKeyPoint.size = 2 * apertureSize;
                    newKeyPoint.response = response;

                    // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                    bool bOverlap = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                        double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                        if (kptOverlap > maxOverlap) {
                            bOverlap = true;
                            // if overlap is >t AND response is higher for new kpt
                            if (newKeyPoint.response > (*it).response) { 
                                *it = newKeyPoint; // replace old key point with new one
                                break;             // quit loop over keypoints
                            }
                        }
                    }
                    if (!bOverlap) { // only add new key point if no overlap has been found in previous NMS
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
        int threshold = 10;         // FAST/AGAST detection threshold score.
        int octaves = 0;            // detection octaves. Use 0 to do single scale.
        float patternScale = 1.0f;  // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        detector = cv::BRISK::create(threshold, octaves, patternScale);
        detector->detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0) {
        // keypoint detector
        const int  	nfeatures = 1000;
        const float scaleFactor = 1.2f;
        const int  	nlevels = 4;
        const int  	edgeThreshold = 15;
        const int  	firstLevel = 0;
        const int  	WTA_K = 2;
        const cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        const int  	patchSize = 15;
        const int  	fastThreshold = 15; 
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(nfeatures,
                                                                        scaleFactor,
                                                                        nlevels,
                                                                        edgeThreshold,
                                                                        firstLevel,
                                                                        WTA_K,
                                                                        scoreType,
                                                                        patchSize,
                                                                        fastThreshold);
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
    else
    {
        TLOG_ERROR << "This detector is not supported";
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}

/*
 * Extract descriptors
 */
tapl::ResultCode tapl::cve::extractDescriptors( const cv::Mat &img, 
                                                std::vector<cv::KeyPoint> &keypoints, 
                                                cv::Mat &descriptors, 
                                                std::string descriptorType ) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {

        int threshold = 10;        // FAST/AGAST detection threshold score.
        int octaves = 0;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } 
    else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    }
    else {
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
tapl::ResultCode tapl::cve::matchDescriptors( std::vector<cv::KeyPoint> &kPtsQuery, 
                                              std::vector<cv::KeyPoint> &kPtsTrain, 
                                              cv::Mat &descQuery, 
                                              cv::Mat &descRef,
                                              std::vector<cv::DMatch> &matches, 
                                              std::string normType, 
                                              std::string matcherType, 
                                              std::string selectorType ) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // create a copy of the descriptors so that any modification does not effect rest of the pipeline
    cv::Mat descQryCopy, descTrnCopy;
    descQuery.copyTo(descQryCopy);
    descRef.copyTo(descTrnCopy);

    if (matcherType.compare("MAT_BF") == 0) {
        int norm = normType.compare("NORM_HAMMING") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(norm, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0) {
        if (descQryCopy.type() != CV_32F) { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descQryCopy.convertTo(descQryCopy, CV_32F);
            descTrnCopy.convertTo(descTrnCopy, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) { // nearest neighbor (best match)

        matcher->match(descQryCopy, descTrnCopy, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0) { 
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descQryCopy, descTrnCopy, knn_matches, 2); // finds the 2 best matches

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }

    }
    else {
        // use nearest-neighbor method as default
        matcher->match(descQryCopy, descTrnCopy, matches); // Finds the best match for each descriptor in desc1
    }

    // return success
    return tapl::SUCCESS;
}
                                
/**
 *  Detect and Match Keypoints; Store the result back into tapl::CameraPairs properties 
 */
tapl::ResultCode tapl::cve::detectAndMatchKpts( tapl::CameraPairs &camPairs,
                                                const std::string detectorType,
                                                const std::string descriptorType ) {
    // get images
    cv::Mat img1, img2;
    if (camPairs.first->getImage(img1) == tapl::FAILURE) {
        TLOG_ERROR << "could not retrieve image";
        return tapl::FAILURE;
    }
    if (camPairs.second->getImage(img2) == tapl::FAILURE) {
        TLOG_ERROR << "could not retrieve image";
        return tapl::FAILURE;
    }
    /* Detect Keypoints */
    // extract 2D keypoints from the images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;    // create empty feature list for current image
    cv::Mat descriptors1, descriptors2;                 // create empty feature descriptor list for current image
    
    if ((tapl::cve::detectKeypoints(img1, keypoints1, detectorType) == tapl::SUCCESS) &&
        (tapl::cve::detectKeypoints(img2, keypoints2, detectorType) == tapl::SUCCESS) ) {
        
        /* Extract Keypoint Descriptors */
        if ((tapl::cve::extractDescriptors(img1, keypoints1, descriptors1, descriptorType) == tapl::SUCCESS) &&
           (tapl::cve::extractDescriptors(img2, keypoints2, descriptors2, descriptorType) == tapl::SUCCESS) ) {
            // push descriptors into the frame
            camPairs.first->pushDescriptors(descriptors1);
            camPairs.second->pushDescriptors(descriptors2);
        }
        else {
            TLOG_ERROR << "Could not perform descriptors extraction";
            return tapl::FAILURE;
        }   
           
        // push keypoints into the frame
        camPairs.first->pushKeypoints(keypoints1);
        camPairs.second->pushKeypoints(keypoints2);
    }
    else {
        TLOG_ERROR << "Could not perform keypoint detection";
        return tapl::FAILURE;
    }

    /* Perform Keypoints Matching */
    std::vector<cv::DMatch> matches;
    if (tapl::cve::matchDescriptors(keypoints1, keypoints2, descriptors1, descriptors2, matches) == tapl::SUCCESS) {
        // store matches in current data frame
        camPairs.pushKptsMatches(matches);
    }
    else {
        TLOG_ERROR << "Could not perform descriptors matching";
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}

/**
 * This function computes Fundamental Matrix given two image frames
 */
tapl::ResultCode tapl::cve::computeFundamentalMatrix( tapl::CameraPairs &camPairs ) {
    if (tapl::cve::detectAndMatchKpts(camPairs) == tapl::SUCCESS) {
        /* Find Fundamental Matrix */
        // store matched keypoints as type Point2f
        std::vector<cv::Point2f> matched_points1;
        std::vector<cv::Point2f> matched_points2;
        std::vector<cv::KeyPoint> kpts1;
        std::vector<cv::KeyPoint> kpts2;

        // retrieve keypoints
        if ((camPairs.first->getKeypoints(kpts1) != tapl::SUCCESS) ||
            (camPairs.second->getKeypoints(kpts2) != tapl::SUCCESS) ) {
                TLOG_ERROR << "could not retrieve keypoints";
                return tapl::FAILURE;
        }

        // retrieve keypoint matches
        std::vector<cv::DMatch> kptsMatches;
        if (camPairs.getKptsMatches(kptsMatches) == tapl::SUCCESS) {
            for (auto matches: kptsMatches) {
                matched_points1.push_back(kpts1[matches.queryIdx].pt);
                matched_points2.push_back(kpts2[matches.trainIdx].pt);
            }
            cv::Mat F = cv::findFundamentalMat(matched_points1, matched_points2, cv::FM_RANSAC, 1, 0.99);
            // store the fundamental matrix in current data frame
            camPairs.pushFundamentalMatrix(F);
        }
    }
    else {
        TLOG_ERROR << "could not perform keypoint detection and matching";
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
} 

/**
 * This function computes Essential Matrix given two image frames
 */
tapl::ResultCode tapl::cve::computeEssentialMatrix( tapl::CameraPairs &camPairs ) {

    if (tapl::cve::detectAndMatchKpts(camPairs) == tapl::SUCCESS) {
        /* Find Fundamental Matrix */
        // store matched keypoints as type Point2f
        std::vector<cv::Point2f> matched_points1;
        std::vector<cv::Point2f> matched_points2;
        std::vector<cv::KeyPoint> kpts1;
        std::vector<cv::KeyPoint> kpts2;

        // retrieve keypoints
        if ((camPairs.first->getKeypoints(kpts1) != tapl::SUCCESS) ||
            (camPairs.second->getKeypoints(kpts2) != tapl::SUCCESS)   ) {
                TLOG_ERROR << "could not retrieve keypoints";
                return tapl::FAILURE;
        }

        // retrieve intrinsic matrix
        cv::Mat camera_matrix;
        if (camPairs.first->getIntrinsicMatrix(camera_matrix) != tapl::SUCCESS) {
                TLOG_ERROR << "could not retrieve intrinsic matrix";
                return tapl::FAILURE;
        }

        // retrieve keypoint matches
        std::vector<cv::DMatch> kptsMatches;
        if (camPairs.getKptsMatches(kptsMatches) == tapl::SUCCESS) {
            for (auto matches: kptsMatches) {
                matched_points1.push_back(kpts1[matches.queryIdx].pt);
                matched_points2.push_back(kpts2[matches.trainIdx].pt);
            }
            cv::Mat E = cv::findEssentialMat(matched_points1, matched_points2, camera_matrix, cv::RANSAC, 0.99, 1.0);
            // store the fundamental matrix in current data frame
            camPairs.pushEssentialMatrix(E);
        }
    }
    else {
        TLOG_ERROR << "could not perform keypoint detection and matching";
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}

/*
 * This function is used to stitch multiple images as a panaromic image.
 */
tapl::ResultCode tapl::cve::stitchPanaromic( const std::vector<cv::Mat> &imgs, 
                                             cv::Mat &panoramic_img ) {
    // sanity check 
    if (imgs.size() < 2) {
        TLOG_ERROR << "Number of images must be 2 or more";
        return tapl::FAILURE;
    }

    // create panaromic stitcher
    bool divide_imgs = false;
    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = stitcher->stitch(imgs, panoramic_img);

    // check if successful
    if (status != cv::Stitcher::OK) {
        TLOG_ERROR << "could not stitch images";
        return tapl::FAILURE;
    }

    // return success
    return tapl::SUCCESS;
}