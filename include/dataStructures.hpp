#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>

namespace tapl {
    struct Pose6dof {
        cv::Mat R;                          // 3x3 rotation matrix
        cv::Mat t;                          // 1x3 translation vector
        cv::Mat euler;                      // roll, pitch, yaw in radians
    };

    struct DataFrame {                      // represents the available sensor information at the same time instance
        
        cv::Mat cameraImg;                  // camera image
        
        std::vector<cv::KeyPoint> keypoints;// 2D keypoints within camera image
        cv::Mat descriptors;                // keypoint descriptors
        std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
        cv::Mat F;                          // fundamental matrix for keypoint correspondences between previous and current frame
        cv::Mat E;                          // essential matrix for keypoint correspondences between previous and current frame
        struct Pose6dof pose;               // pose
        cv::Mat triangulated_pts;           // triangulated points
    };
};


#endif /* dataStructures_h */
