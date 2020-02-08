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
    tapl::RingBuffer<tapl::DataFrame> dataBuffer(dataBufferSize);

    // Read camera calibration
    std::cout << "loading camera calibration..." << std::endl;
    cv::FileStorage opencv_file(calibPath, cv::FileStorage::READ);
    cv::Mat camera_matrix;
    opencv_file["camera_matrix"] >> camera_matrix;
    cv::Mat dist_coeff;
    opencv_file["dist_coeff"] >> dist_coeff;
    opencv_file.release();
    std::cout << "Camera Matrix:" << std::endl;
    std::cout << camera_matrix << std::endl;
    std::cout << "Distortion Coefficients:" << std::endl;
    std::cout << dist_coeff << std::endl;

    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    int imgIndex = 0;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto fname : fnames) {
        /* Load image into buffer */
        // load image from file and convert to grayscale
        cv::Mat img, img_undistorted, img_gray;
        img = cv::imread(fname);
        cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);
        cv::cvtColor(img_undistorted, img_gray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        tapl::DataFrame frame;
        frame.cameraFrame.pushImage(img_gray);
        dataBuffer.push(frame);

        cout << "----------------------------------------------------" << endl;
        cout << "Image [" << imgIndex << "] loaded into the ring buffer" << endl;

        // perform keypoints detection and matching if more than one image is loaded into the buffer
        if(dataBuffer.getSize() > 1) {
            tapl::DataFrame dframe1, dframe2;
            dframe1 = dataBuffer.get(dataBuffer.getSize()-1);
            dframe2 = dataBuffer.get(dataBuffer.getSize()-2);
            if(tapl::cve::detectAndMatchKpts(dframe1, dframe2) == tapl::SUCCESS) {
                // visualize matches between current and previous image
                cv::Mat img1, img1_color, img2, img2_color, matchImg;

                // retrieve both image frames
                dframe1.cameraFrame.getImage(img1);
                dframe2.cameraFrame.getImage(img2);
                // convert to RGB
                cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
                cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
                
                // retrieve keypoints
                std::vector<cv::KeyPoint> kpts1, kpts2;
                dframe1.cameraFrame.getKeypoints(kpts1);
                dframe2.cameraFrame.getKeypoints(kpts2);
                // retrieve keypoints matches
                std::vector<cv::DMatch> kptsMatches;
                dframe1.getKptsMatches(kptsMatches);

                // draw the matches 
                cv::drawMatches(img1_color, kpts1, img2_color, kpts2,
                                kptsMatches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                // visualize keypoints matches
                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    } 

    return 0;
}