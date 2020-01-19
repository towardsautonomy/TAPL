#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <filesystem>

#include "tapl.hpp"

int main (int argc, char** argv) {
    // data location
    std::string dataPath = "../data/kitti_pc";

    // number of point-cloud to load
    int nPC = -1;   // last file index to load

    // prepare a viewer
    tapl::viz::Visualizer * visualizer = new tapl::viz::Visualizer(0., 0., 0., tapl::viz::XY, 25.0);

    namespace fs = std::filesystem;
    std::vector<std::string> fnames;
    int pcIndex = 0;
    /* Loop over all the images */
    for (auto& p: fs::directory_iterator(dataPath)) {
        fnames.push_back(p.path());
    }
    std::sort(fnames.begin(), fnames.end());
    for (auto fname : fnames) {
        // load .pcd files
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ> (fname, *cloud) == -1) {
            std::cerr << "Couldn't read file " << std::endl;
            exit(EXIT_FAILURE);
        }

        // downsample the point-cloud
        const float downsample_res = 0.4;
        tapl::pte::downsampleCloud<pcl::PointXYZ> (cloud, downsample_res);

        // clear the viewer
        visualizer->clearScene();

        // render full point-cloud
        visualizer->renderPointCloud<pcl::PointXYZ> (cloud, 2.0, 1.0, 1.0, 1.0, "cull_cloud_"+std::to_string(pcIndex));
        
        // crop cloud
        const float x_lim[] = {-20, 50};    // x-axis point to the front
        const float y_lim[] = {-8, 8};    // y-axis point to the left
        const float z_lim[] = {-5, 5};      // z-axis point up
        tapl::pte::cropCloud<pcl::PointXYZ> (cloud, 
                                                x_lim[0], x_lim[1], 
                                                y_lim[0], y_lim[1], 
                                                z_lim[0], z_lim[1]);

        // segment ground plane
        std::pair<typename pcl::PointCloud<pcl::PointXYZ>::Ptr, typename pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentedCloud;
        const int maxIterations = 100;
        const float distanceThreshold = downsample_res;
        if(tapl::pte::segmentPlane<pcl::PointXYZ> (cloud, segmentedCloud, maxIterations, distanceThreshold) 
                                                                                            != tapl::SUCCESS) {
            std::cerr << "Could not perform segmentation" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto groundPlane = segmentedCloud.first;
        auto nonGroundCloud = segmentedCloud.second;

        // render ground plane
        visualizer->renderPointCloud<pcl::PointXYZ> (groundPlane, 2.0, 0.0, 1.0, 0.0, "ground_cloud_"+std::to_string(pcIndex));

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
        for(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster : cloudClusters)
        {
            ++clusterId;

            tapl::BBox3d bbox = tapl::pte::getBoundingBox<pcl::PointXYZ> (cluster);

            // any cluster bigger than certain threshold could be 
            // either a building or side boundary of the road
            const float obj_len_thres[] = {0.3, 7};     // {min, max}
            const float obj_width_thres[] = {0.3, 5};   // {min, max}
            const float obj_height_thres[] = {0.3, 5};  // {min, max}

            if( ((bbox.x_max - bbox.x_min) >= obj_len_thres[0]) &&
                ((bbox.x_max - bbox.x_min) <= obj_len_thres[1]) &&
                ((bbox.y_max - bbox.y_min) >= obj_width_thres[0]) &&
                ((bbox.y_max - bbox.y_min) <= obj_width_thres[1]) &&
                ((bbox.z_max - bbox.z_min) >= obj_height_thres[0]) &&
                ((bbox.z_max - bbox.z_min) <= obj_height_thres[1]))
            {
                // cars
                visualizer->renderPointCloud<pcl::PointXYZ> 
                            (cluster, 2.0, 1.0, 0.0, 0.0, "obstCloud_"+std::to_string(clusterId));
                visualizer->renderBbox3d (bbox, 1.0, 0.0, 0.0, 0.5, "bbox_"+std::to_string(clusterId));
            }
            else {
                // everything else
                visualizer->renderBbox3d (bbox, 0.0, 0.0, 1.0, 0.5, "bbox_"+std::to_string(clusterId));
            }
        }

        // render scene
        visualizer->renderScene(100);
        pcIndex++;
        if((nPC != -1) && (pcIndex >= nPC)) break;
    }

    // render and hold
    visualizer->renderSceneAndHold();
}