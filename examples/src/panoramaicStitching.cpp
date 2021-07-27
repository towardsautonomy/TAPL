#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "tapl.hpp"

int main(int argc, const char *argv[])
{
    // paths
    std::string calibPath = "../data/calib/camera_model.yaml";
    std::string dataPath = "../data/living_room_panorama";

    // number of images
    int nImages = -1;   // last file index to load

    // Read camera calibration
    cv::FileStorage opencv_file(calibPath, cv::FileStorage::READ);
    cv::Mat camera_matrix;
    opencv_file["camera_matrix"] >> camera_matrix;
    cv::Mat dist_coeff;
    opencv_file["dist_coeff"] >> dist_coeff;
    opencv_file.release();

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
    TLOG_INFO << "[" << imgs.size() << "] Images loaded from file";

    // perform panoramic stitching
    cv::Mat pano;
    tapl::ResultCode result = tapl::cve::stitchPanaromic(imgs, pano);
    if (result != tapl::SUCCESS) {
        TLOG_INFO << "Panoramic stitching failed..";
        exit(EXIT_FAILURE);
    }    

    // show stitched image
    std::string windowName = "Panoramic Image";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1920, 1080);
    cv::imshow(windowName, pano);
    cv::waitKey(0); 

    return 0;
}