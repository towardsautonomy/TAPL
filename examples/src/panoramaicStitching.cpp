#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "tapl.hpp"

using namespace std;

int main(int argc, const char *argv[])
{
    // paths
    string calibPath = "../data/calib/camera_model.yaml";
    string dataPath = "../data/living_room_panorama";

    // number of images
    int nImages = -1;   // last file index to load

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
    std::vector<cv::Mat> imgs;
    int imgIndex = 0;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto fname : fnames) {
        // load image from file
        cv::Mat img, img_undistorted;
        img = cv::imread(fname);

        // undistort the image
        cv::undistort(img, img_undistorted, camera_matrix, dist_coeff);

        // push undistorted image to the vector
        imgs.push_back(img_undistorted);
        imgIndex++;
        if((nImages != -1) && (imgIndex >= nImages)) break;
    }
    cout << "[" << imgs.size() << "] Images loaded from file" << endl;

    // perform panoramic stitching
    cv::Mat pano;
    tapl::ResultCode result = tapl::cve::stitchPanaromic(imgs, pano);
    if (result != tapl::SUCCESS) {
        std::cout << "ERROR: Panoramic stitching failed.." << std::endl;
        exit(EXIT_FAILURE);
    }    

    // show stitched image
    string windowName = "Panoramic Image";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, pano);
    cv::waitKey(0); 

    return 0;
}