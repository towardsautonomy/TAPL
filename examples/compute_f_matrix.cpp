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

#include "dataStructures.hpp"
#include "matching2D.hpp"
#include "ringBuffer.hpp"
#include "cvEngine.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 10;       // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false;            // visualize results
    RingBuffer<tapl::DataFrame> dataBuffer(dataBufferSize);

    /* Loop over all the images */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* Load image into buffer */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        tapl::DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push(frame);
    } 

    if(tapl::cve::getFundamentalMatrix(dataBuffer) == 0) {
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            cv::Mat imgQuery = (dataBuffer.get_ptr(imgIndex - 1)->cameraImg).clone();
            cv::Mat matchImg = (dataBuffer.get_ptr(imgIndex)->cameraImg).clone();
            // store matched keypoints as type Point2f
           std::vector<cv::Point2f> matched_points1;
           std::vector<cv::Point2f> matched_points2;
            for (auto matches: dataBuffer.get_ptr(imgIndex)->kptMatches) {
                matched_points1.push_back(dataBuffer.get_ptr(imgIndex - 1)->keypoints[matches.queryIdx].pt);
                matched_points2.push_back(dataBuffer.get_ptr(imgIndex)->keypoints[matches.trainIdx].pt);
            }
            cv::Mat fundamental_matrix = dataBuffer.get_ptr(imgIndex)->F;

            std::cout << "Fundamental Matrix:\n " << fundamental_matrix << std::endl;

            // draw the left points corresponding epipolar lines in right image
            std::vector<cv::Vec3f> lines1;
            cv::computeCorrespondEpilines(cv::Mat(matched_points1), // image points
                                        1,                   // in image 1 (can also be 2)
                                        fundamental_matrix,         // F matrix
                                        lines1);             // vector of epipolar lines

            // for all epipolar lines
            for (auto it = lines1.begin(); it != lines1.end(); ++it)
            {

                // draw the epipolar line between first and last column
                cv::line(matchImg, cv::Point(0, -(*it)[2] / (*it)[1]),
                        cv::Point(matchImg.cols, -((*it)[2] + (*it)[0] * matchImg.cols) / (*it)[1]),
                        cv::Scalar(255, 255, 255));
            }

            // draw the left points corresponding epipolar lines in left image
            std::vector<cv::Vec3f> lines2;
            cv::computeCorrespondEpilines(cv::Mat(matched_points2), 2, fundamental_matrix, lines2);
            for (auto it = lines2.begin(); it != lines2.end(); ++it)
            {

                // draw the epipolar line between first and last column
                cv::line(imgQuery, cv::Point(0, -(*it)[2] / (*it)[1]),
                        cv::Point(imgQuery.cols, -((*it)[2] + (*it)[0] * imgQuery.cols) / (*it)[1]),
                        cv::Scalar(255, 255, 255));
            }

            // Display the images with points and epipolar lines
            cv::namedWindow("Right Image Epilines");
            cv::imshow("Right Image Epilines", imgQuery);
            cv::namedWindow("Left Image Epilines");
            cv::imshow("Left Image Epilines", matchImg);
            cv::waitKey(0); // wait for key to be pressed

            cv::FileStorage opencv_file("../scripts/camera_model.yaml", cv::FileStorage::READ);
            cv::Mat camera_matrix;
            opencv_file["camera_matrix"] >> camera_matrix;
            cv::Mat dist_coeff;
            opencv_file["dist_coeff"] >> dist_coeff;
            opencv_file.release();
            std::cout << camera_matrix << std::endl;
            std::cout << dist_coeff << std::endl;
        }
    }

    return 0;
}