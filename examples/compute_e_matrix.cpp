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

    if(tapl::cve::getEssentialMatrix(dataBuffer, camera_matrix) == 0) {
        /* Loop over all the images */
        for (size_t imgIndex = 1; imgIndex < dataBuffer.getSize(); imgIndex++) {
            cv::Mat essential_matrix = dataBuffer.get_ptr(imgIndex)->E;
            std::cout << "Essential Matrix:\n " << essential_matrix << std::endl;
        }
    }

    return 0;
}