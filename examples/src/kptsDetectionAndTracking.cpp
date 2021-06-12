#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "tapl.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // paths
    string calibPath = "../data/calib/camera_model.yaml";
    string dataPath = "../data/living_room";

    // number of images
    int nImages = 50;   // last file index to load

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    tapl::RingBuffer<tapl::CameraPairs> dataBuffer(dataBufferSize);

    // Read camera calibration
    cv::FileStorage opencv_file(calibPath, cv::FileStorage::READ);
    cv::Mat camera_matrix;
    opencv_file["camera_matrix"] >> camera_matrix;
    cv::Mat dist_coeff;
    opencv_file["dist_coeff"] >> dist_coeff;
    opencv_file.release();

    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    int imgIndex = 0;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto it=fnames.begin(); it!=fnames.end(); ++it) {
        // wait for at least 2 images
        if (fnames.begin() == it) {
            // load image from file and convert to grayscale
            cv::Mat img, img_undistorted, img_gray;
            img = cv::imread(*it);
            cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);
            cv::cvtColor(img_undistorted, img_gray, cv::COLOR_BGR2GRAY);

            // push image into data frame buffer
            tapl::CameraPairs camPairs;
            tapl::CameraFrame frame(img_gray);
            *camPairs.second = frame;
            camPairs.second->pushIntrinsicMatrix(camera_matrix);
            dataBuffer.push(camPairs);
        }
        else {
            /* Load image into buffer */
            // load image from file and convert to grayscale
            cv::Mat img, img_undistorted, img_gray;
            img = cv::imread(*it);
            cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);
            cv::cvtColor(img_undistorted, img_gray, cv::COLOR_BGR2GRAY);

            // push image into data frame buffer
            cv::Mat prev_img;
            if (dataBuffer.get(dataBuffer.getSize()-1).second->getImage(prev_img) != tapl::SUCCESS) {
                TLOG_ERROR << "could not retrieve previous frame";
                exit(1);
            }
            tapl::CameraPairs camPairs(prev_img, img_gray);
            camPairs.second->pushIntrinsicMatrix(camera_matrix);
            dataBuffer.push(camPairs);

            TLOG_INFO << "----------------------------------------";
            TLOG_INFO << "Image Pair [" << imgIndex << "] loaded into the ring buffer";

            if (tapl::cve::detectAndMatchKpts(camPairs) == tapl::SUCCESS) {
                // visualize matches between current and previous image
                cv::Mat img1, img1_color, img2, img2_color, matchImg;

                // retrieve both image frames
                camPairs.first->getImage(img1);
                camPairs.second->getImage(img2);
                // convert to RGB
                cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
                cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
                
                // retrieve keypoints
                std::vector<cv::KeyPoint> kpts1, kpts2;
                camPairs.first->getKeypoints(kpts1);
                camPairs.second->getKeypoints(kpts2);
                // retrieve keypoints matches
                std::vector<cv::DMatch> kptsMatches;
                camPairs.getKptsMatches(kptsMatches);

                // draw the matches 
                cv::drawMatches(img1_color, kpts1, img2_color, kpts2,
                                kptsMatches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                // visualize keypoints matches
                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                TLOG_INFO << "Press key to continue to next image";
                cv::waitKey(0); // wait for key to be pressed
            }
        }

        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    } 

    return 0;
}