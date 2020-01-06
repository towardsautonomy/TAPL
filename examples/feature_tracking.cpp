#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.hpp"
#include "matching2D.hpp"
#include "ringBuffer.hpp"
#include "cvEngine.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../data/living_room";

    // number of images
    int nImages = 10;   // last file index to load

    // misc
    int dataBufferSize = 30;       // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false;            // visualize results
    RingBuffer<tapl::DataFrame> dataBuffer(dataBufferSize);

    // Read camera calibration
    cv::FileStorage opencv_file("../scripts/camera_model.yaml", cv::FileStorage::READ);
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
        cv::Mat img, imgGray;
        img = cv::imread(fname);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // push image into data frame buffer
        tapl::DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push(frame);

        cout << "Image [" << imgIndex << "] loaded into the ring buffer" << endl;
        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    } 

    if(tapl::cve::detectAndMatchKpts(dataBuffer, true) == 0) {
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            // visualize matches between current and previous image
            cv::Mat matchImg = (dataBuffer.get_ptr(imgIndex)->cameraImg).clone();
            cv::Mat img1;
            cv::cvtColor(dataBuffer.get_ptr(imgIndex - 1)->cameraImg, img1, cv::COLOR_GRAY2BGR);
            cv::Mat img2;
            cv::cvtColor(dataBuffer.get_ptr(imgIndex)->cameraImg, img2, cv::COLOR_GRAY2BGR);

            cv::drawMatches(img1, dataBuffer.get_ptr(imgIndex - 1)->keypoints,
                            img2, dataBuffer.get_ptr(imgIndex)->keypoints,
                            dataBuffer.get_ptr(imgIndex)->kptMatches, matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            string windowName = "Matching keypoints between two camera images";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, matchImg);
            cout << "Press key to continue to next image" << endl;
            cv::waitKey(0); // wait for key to be pressed
        }
    }

    return 0;
}