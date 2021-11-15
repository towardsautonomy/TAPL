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

const double rMin = 5.0;
const double rMax = 35.0;
const int16_t bevSize = 1024;

// convert r and theta to pixel coordinate
void rtheta2px( const double &r, 
                const int16_t &theta, 
                int16_t &x_px, 
                int16_t &y_px) {
    double x = r * cos(DEG2RAD(theta));
    double y = r * sin(DEG2RAD(theta));
    // convert to pixel space
    x_px = static_cast<int16_t>(((-y + rMax) * double(bevSize)) / (2 * rMax));
    y_px = static_cast<int16_t>(((-x + rMax) * double(bevSize)) / (2 * rMax));
}

// compute moving average of r for each value of theta
void weightedMovingAvg( const std::vector<double> &r, 
                        const std::vector<double> &theta,
                        const std::vector<double> &weight,
                        const size_t &len,
                        std::map<double, double> &rangeMap_out ) {

    for (auto r_=r.begin(); r_!=r.end(); ++r_) {
        double r_avg=0.0;
        double w_sum=0.0;
        auto idx_ = std::distance(r.begin(), r_);
        for (auto r_hist=max(r.begin(), r_-len); r_hist!=min(r.end(), r_+len); ++r_hist) {
            auto idx_hist = std::distance(r.begin(), r_hist);
            r_avg += weight.at(idx_hist) * r.at(idx_hist);
            w_sum += weight.at(idx_hist);
        }
        r_avg = r_avg / w_sum;
        rangeMap_out[theta[idx_]] = r_avg;
    }
}

// main function
int main (int argc, char** argv) {
    //////////////////////////////////////////
    // Load LiDAR point-clouds and pre-process
    //////////////////////////////////////////
    // data location
    std::string dataPath = "../data/kitti_pc";

    // number of point-cloud to load
    int nPC = -1;   // last file index to load

    // prepare a viewer
    tapl::viz::Visualizer * visualizer = new tapl::viz::Visualizer(0., 0., 0., tapl::viz::XY, 25.0);

    // filenames
    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    int pcIndex = 0;
    // Loop over all the images
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto fname : fnames) {
        // load .pcd files
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ> (fname, *cloud) == -1) {
            TLOG_ERROR << "Couldn't read file ";
            exit(EXIT_FAILURE);
        }

        // downsample the point-cloud
        const float downsample_res = 0.4;
        tapl::pte::downsampleCloud<pcl::PointXYZ> (cloud, downsample_res);

        // clear the viewer
        visualizer->clearScene();

        // render full point-cloud
        visualizer->renderPointCloud<pcl::PointXYZ> (cloud, 3.0, 1.0, 1.0, 1.0, "full_cloud_"+std::to_string(pcIndex));
        
        // crop cloud
        const float x_lim[] = {-20, 50};    // x-axis point to the front
        const float y_lim[] = {-8, 8};    // y-axis point to the left
        const float z_lim[] = {-5, 5};      // z-axis point up
        tapl::pte::cropCloud<pcl::PointXYZ> (cloud, 
                                                x_lim[0], x_lim[1], 
                                                y_lim[0], y_lim[1], 
                                                z_lim[0], z_lim[1]);

        //////////////////////////////////////////
        // Perform ground-plane segmentation and extract object clusters
        //////////////////////////////////////////
        // segment ground plane
        std::pair<typename pcl::PointCloud<pcl::PointXYZ>::Ptr, typename pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedCloud;
        const int maxIterations = 100;
        const float distanceThreshold = downsample_res;
        if (tapl::pte::segmentPlane<pcl::PointXYZ> (cloud, segmentedCloud, maxIterations, distanceThreshold) 
                                                                                            != tapl::SUCCESS) {
            TLOG_ERROR << "Could not perform segmentation";
            exit(EXIT_FAILURE);
        }
        auto groundPlane = segmentedCloud.first;
        auto nonGroundCloud = segmentedCloud.second;

        // render ground plane
        visualizer->renderPointCloud<pcl::PointXYZ> (groundPlane, 3.0, 0.0, 1.0, 0.0, "ground_cloud_"+std::to_string(pcIndex));

        // perform clustering
        const float distTolerance = 0.5; //meters
        const int minNumPoints = 10;     
        const int maxNumPoints = 10000;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = 
            tapl::pte::euclideanClustering<pcl::PointXYZ>(nonGroundCloud, 
                                                            distTolerance, 
                                                            minNumPoints, 
                                                            maxNumPoints);

        // extract each cluster and display bounding box
        int clusterId = 0;
        std::vector<tapl::BBox3d> bboxFiltered;
        for (pcl::PointCloud<pcl::PointXYZ>::Ptr cluster : cloudClusters)
        {
            ++clusterId;
            tapl::BBox3d bbox = tapl::pte::getBoundingBox<pcl::PointXYZ> (cluster);
            tapl::BBox3d oriented_bbox = tapl::pte::getOrientedBoundingBox<pcl::PointXYZ> (cluster);

            //////////////////////////////////////////
            // Point-Cloud Visualization
            //////////////////////////////////////////
            // any cluster bigger than certain threshold could be 
            // either a building or side boundary of the road
            const float obj_len_thres[] = {0.3, 6};     // {min, max}
            const float obj_width_thres[] = {0.3, 5};   // {min, max}
            const float obj_height_thres[] = {0.3, 5};  // {min, max}

            if( (bbox.length >= obj_len_thres[0]) &&
                (bbox.length <= obj_len_thres[1]) &&
                (bbox.width >= obj_width_thres[0]) &&
                (bbox.width <= obj_width_thres[1]) &&
                (bbox.height >= obj_height_thres[0]) &&
                (bbox.height <= obj_height_thres[1]))
            {
                // cars
                visualizer->renderPointCloud<pcl::PointXYZ> 
                            (cluster, 3.0, 1.0, 0.0, 0.0, "obstCloud_"+std::to_string(clusterId));
                visualizer->renderBbox3d (bbox, 1.0, 0.0, 0.0, 0.5, "bbox_"+std::to_string(clusterId));
                bboxFiltered.push_back(bbox);
            }
            else {
                // everything else
                visualizer->renderBbox3d (bbox, 0.0, 0.0, 1.0, 0.5, "bbox_"+std::to_string(clusterId));
            }
        }

        // render scene
        visualizer->renderScene(100);
        cv::waitKey(0);
        pcIndex++;
        if((nPC != -1) && (pcIndex >= nPC)) break;

        //////////////////////////////////////////
        // Occupancy Map Visualization
        //////////////////////////////////////////
        // find drivable area
        std::map<int16_t, double> rangeMap;
        for (auto& point : *nonGroundCloud) {
            double r = sqrt(pow(point.x, 2) + pow(point.y, 2));
            int16_t theta = static_cast<int16_t>(RAD2DEG(atan2(point.y, point.x)));
            if (rangeMap.find(theta) == rangeMap.end()) {
                if (r > rMin) rangeMap[theta] = fmin(r, rMax);
                else continue;
            }
            else {
                if ((rangeMap[theta] > r) && (r > rMin)) rangeMap[theta] = fmin(r, rMax);
            }
        }

        // set r for missing values of theta
        for (auto theta=-180; theta <= 180; ++theta) {
            if (rangeMap.find(theta) == rangeMap.end()) rangeMap[theta] = rMax;
        }


        // filter for robustness
        std::vector<double> r_;
        std::vector<double> theta_;
        std::vector<double> weight_;
        for (auto& rtheta : rangeMap) {
            theta_.push_back(rtheta.first);
            r_.push_back(rtheta.second);
            weight_.push_back(exp(-rtheta.second));
        }
        std::map<double, double> rangeMap_filtered;
        weightedMovingAvg(r_, theta_, weight_, 2, rangeMap_filtered);

        // iterate over range map
        // define a square image for displaying birds-eye view occupancy map
        cv::Mat bev(bevSize, bevSize, CV_8UC3, cv::Scalar(0,0,0));
        cv::Point * pts = new cv::Point[rangeMap.size()];
        size_t idx = 0;
        for (auto& rtheta : rangeMap_filtered) {
            int16_t x_px, y_px;
            rtheta2px(rtheta.second, rtheta.first, x_px, y_px);
            pts[idx] = cv::Point(x_px, y_px);
            idx++;
        }
        const cv::Point* ppt[1] = { pts };
        int npt[] = { static_cast<int>(rangeMap_filtered.size()) };
        int rectw=25, recth=60;
        cv::Rect rect((bevSize/2-rectw/2), (bevSize/2-recth/2), rectw, recth);
        // drivable area
        cv::polylines( bev, ppt, npt, 1, false, cv::Scalar( 255, 255, 255 ), 8 );
        cv::fillPoly( bev, ppt, npt, 1, cv::Scalar( 51, 128, 51 ), 8 );
        // filled red rectangle representing ego-vehicle
        cv::rectangle( bev, rect, cv::Scalar(255, 255, 255), -1);
        cv::rectangle( bev, rect, cv::Scalar(0, 0, 0), 2);
        // object bboxes
        for (auto &bbox : bboxFiltered) {
            int16_t bbox_xmin_px = static_cast<int16_t>(((-(bbox.y + (bbox.width/2.0)) + rMax) * double(bevSize)) / (2 * rMax));
            int16_t bbox_xmax_px = static_cast<int16_t>(((-(bbox.y - (bbox.width/2.0)) + rMax) * double(bevSize)) / (2 * rMax));
            int16_t bbox_ymin_px = static_cast<int16_t>(((-(bbox.x + (bbox.length/2.0)) + rMax) * double(bevSize)) / (2 * rMax));
            int16_t bbox_ymax_px = static_cast<int16_t>(((-(bbox.x - (bbox.length/2.0)) + rMax) * double(bevSize)) / (2 * rMax));
            cv::rectangle(  bev, 
                            cv::Point(bbox_xmin_px, bbox_ymin_px), 
                            cv::Point(bbox_xmax_px, bbox_ymax_px), 
                            cv::Scalar(0, 0, 255), -1);
        }

        // print range
        cv::Mat range(bevSize, 150, CV_8UC3, cv::Scalar(0,0,0));
        const double rangeStop = 5.0;
        for (auto r_=(-rMax); r_ < rMax; r_+=rangeStop) {
            int16_t rangePx = static_cast<int16_t>(((-r_ + rMax) * double(bevSize)) / (2 * rMax));
            if ((rangePx < bevSize) && (r_ < 0))
                cv::putText(range,std::to_string(static_cast<int>(r_))+"m",cv::Point(10,rangePx),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(255,255,255),1,true);
            else if (r_ >= 0)
                cv::putText(range," "+std::to_string(static_cast<int>(r_))+"m",cv::Point(10,rangePx),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(255,255,255),1,true);
        }
        cv::Mat imgViz;
        cv::hconcat(bev, range, imgViz);

        cv::imshow("OccupancyMap", imgViz);
        cv::waitKey(1);
    }

    // render and hold
    visualizer->renderSceneAndHold();
}