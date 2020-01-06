#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include "cvEngine.hpp"

using namespace std;

/**
 *  Detect and Match Keypoints; Store the result back into tapl::DataFrame 
 */
int tapl::cve::detectAndMatchKpts(RingBuffer<tapl::DataFrame> &dataBuffer, bool verbose)
{
    /* Loop over all the images */
    for (size_t imgIndex = 0; imgIndex < dataBuffer.getSize(); imgIndex++)
    {
        /* Detect Keypoints */
        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = "FAST";   //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, dataBuffer.get_ptr(imgIndex)->cameraImg, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, dataBuffer.get_ptr(imgIndex)->cameraImg, false);
        }
        else
        {
            detKeypointsModern(keypoints, dataBuffer.get_ptr(imgIndex)->cameraImg, detectorType, false);
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer.get_ptr(imgIndex)->keypoints = keypoints;
        if(verbose) {
            cout << "=======================================================" << endl;
            cout << " - [" << keypoints.size() << "] keypoints detected" << endl;
        }

        /* Extract Keypoint Descriptors */
        cv::Mat descriptors;
        string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints(dataBuffer.get_ptr(imgIndex)->keypoints, dataBuffer.get_ptr(imgIndex)->cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        dataBuffer.get_ptr(imgIndex)->descriptors = descriptors;

        if(verbose) {
            cout << " - Keypoint Descriptors extracted" << endl;
        }

        /* Perform Keypoints Matching */
        vector<cv::DMatch> matches;
        if (imgIndex > 0) // wait until at least two images have been processed
        {
            /* Match Keypoint Descriptors */
            string matcherType = "MAT_BF";         // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY";  // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
            matchDescriptors(dataBuffer.get_ptr(imgIndex - 1)->keypoints, dataBuffer.get_ptr(imgIndex)->keypoints,
                             dataBuffer.get_ptr(imgIndex - 1)->descriptors, dataBuffer.get_ptr(imgIndex)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            dataBuffer.get_ptr(imgIndex)->kptMatches = matches;

            if(verbose) {
                cout << " - [" << matches.size() << "] keypoint matches found" << endl;
            }
        }
    } 

    return 0; // success
}

/**
 * This function computes Fundamental Matrix given a set of images
 */
int tapl::cve::getFundamentalMatrix(RingBuffer<tapl::DataFrame> &dataBuffer) {
    if(tapl::cve::detectAndMatchKpts(dataBuffer, true) == 0) {
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            /* Find Fundamental Matrix */
            // store matched keypoints as type Point2f
            std::vector<cv::Point2f> matched_points1;
            std::vector<cv::Point2f> matched_points2;
            for (auto matches: dataBuffer.get_ptr(imgIndex)->kptMatches) {
                matched_points1.push_back(dataBuffer.get_ptr(imgIndex - 1)->keypoints[matches.queryIdx].pt);
                matched_points2.push_back(dataBuffer.get_ptr(imgIndex)->keypoints[matches.trainIdx].pt);
            }
            cv::Mat F = cv::findFundamentalMat(matched_points1, matched_points2, cv::FM_RANSAC, 3, 0.99);
            // store the fundamental matrix in current data frame
            dataBuffer.get_ptr(imgIndex)->F = F;
        }
        // return success
        return 0;
    }
    else {
        // return failure
        return -1;
    }
} 

/**
 * This function computes Essential Matrix given a set of images
 */
int tapl::cve::getEssentialMatrix(RingBuffer<tapl::DataFrame> &dataBuffer, cv::Mat &camera_matrix) {
    if(tapl::cve::detectAndMatchKpts(dataBuffer, true) == 0) {
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            // store matched keypoints as type Point2f
            std::vector<cv::Point2f> matched_points1;
            std::vector<cv::Point2f> matched_points2;
            for (auto matches: dataBuffer.get_ptr(imgIndex)->kptMatches) {
                matched_points1.push_back(dataBuffer.get_ptr(imgIndex - 1)->keypoints[matches.queryIdx].pt);
                matched_points2.push_back(dataBuffer.get_ptr(imgIndex)->keypoints[matches.trainIdx].pt);
            }

            cv::Mat essential_matrix = cv::findEssentialMat(matched_points1, matched_points2, camera_matrix, cv::RANSAC, 0.99, 2.0);

            cv::Mat R, t;
            cv::recoverPose(essential_matrix, matched_points1, matched_points2, camera_matrix, R, t);

            cout << "R: " << endl;
            cout << R << endl;
            cout << "t:" << endl;
            cout << t << endl;

            /* Compute Essential Matrix */
            // cv::Mat essential_matrix = camera_matrix.t() * dataBuffer.get_ptr(imgIndex)->F * camera_matrix;
            dataBuffer.get_ptr(imgIndex)->E = essential_matrix;
        }
        // return success
        return 0;
    }
    else {
        // return failure
        return -1;
    }
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
int tapl::cve::getPose(RingBuffer<tapl::DataFrame> &dataBuffer, cv::Mat &camera_matrix) {
    if(tapl::cve::detectAndMatchKpts(dataBuffer, true) == 0) {
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            // store matched keypoints as type Point2f
            std::vector<cv::Point2f> matched_points1;
            std::vector<cv::Point2f> matched_points2;
            for (auto matches: dataBuffer.get_ptr(imgIndex)->kptMatches) {
                matched_points1.push_back(dataBuffer.get_ptr(imgIndex - 1)->keypoints[matches.queryIdx].pt);
                matched_points2.push_back(dataBuffer.get_ptr(imgIndex)->keypoints[matches.trainIdx].pt);
            }

            cv::Mat mask;
            cv::Mat essential_matrix = cv::findEssentialMat(matched_points1, matched_points2, camera_matrix, cv::RANSAC, 0.99, 2.0, mask);

            cv::Mat R, t;
            cv::Mat triangulated_points;
            cv::recoverPose(essential_matrix, matched_points1, matched_points2, camera_matrix, R, t, 50, mask);
            //cv::recoverPose(essential_matrix, matched_points1, matched_points2, camera_matrix, R, t, 50, mask, triangulated_points);

            std::vector<cv::Point2d> triangulation_points1, triangulation_points2;
            for(int i = 0; i < mask.rows; i++) {
                if(mask.at<unsigned char>(i)){
                triangulation_points1.push_back 
                            (cv::Point2d((double)matched_points1[i].x,(double)matched_points1[i].y));
                triangulation_points2.push_back 
                            (cv::Point2d((double)matched_points2[i].x,(double)matched_points2[i].y));
                }
            }
            cv::Mat P0 = cv::Mat::eye(3, 4, CV_64FC1);
            cv::Mat P1 = cv::Mat::eye(3, 4, CV_64FC1);
            R.copyTo(P1.rowRange(0,3).colRange(0,3));
            t.copyTo(P1.rowRange(0,3).col(3));
            cv::Mat point3d_homo;
            if((triangulation_points1.size() > 0) && (triangulation_points2.size() > 0)) {
                cv::triangulatePoints(camera_matrix * P0, camera_matrix * P1, 
                                        triangulation_points1, triangulation_points2,
                                        triangulated_points);
            }

            /* Populate the Pose */
            dataBuffer.get_ptr(imgIndex)->pose.R = R;
            dataBuffer.get_ptr(imgIndex)->pose.t = t;

            /* convert rodrigues to euler angle */
            float roll, pitch, yaw;
            rodrigues2euler(R, roll, pitch, yaw);

            cv::Mat euler(1, 3, CV_32FC1);
            euler.at<float>(0,0) = roll;
            euler.at<float>(0,1) = pitch;
            euler.at<float>(0,2) = yaw;
            dataBuffer.get_ptr(imgIndex)->pose.euler = euler;
            dataBuffer.get_ptr(imgIndex)->triangulated_pts = triangulated_points;
        }
        // return success
        return 0;
    }
    else {
        // return failure
        return -1;
    }
}