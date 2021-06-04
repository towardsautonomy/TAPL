/**
 * @file    taplTypes.hpp
 * @brief   This file provides type definitions used throughout this library
 * @author  Shubham Shrivastava
 */

#ifndef TAPL_TYPES_H_
#define TAPL_TYPES_H_

#include <vector>
#include <opencv2/core.hpp>

namespace tapl
{
    /**
     * @brief Result code enumerations 
     */
    typedef enum {
        FAILURE=-1,                         /**< Failure */
        SUCCESS=0                           /**< Success */
    } ResultCode;

    /**
     * @brief 3D Point
     */
    struct Point3d {
        double x;                           /*!< x-coordinate */
        double y;                           /*!< y-coordinate */
        double z;                           /*!< z-coordinate */
    };

    /**
     * @brief 3D Bounding-Box
     */
    struct BBox3d {
        double x_min;                       /*!< min x-coordinate */
        double x_max;                       /*!< max x-coordinate */
        double y_min;                       /*!< min y-coordinate */
        double y_max;                       /*!< max y-coordinate */
        double z_min;                       /*!< min z-coordinate */
        double z_max;                       /*!< max z-coordinate */
    };

    /**
     * @brief 6-DOF Camera Pose
     */
    struct Pose6dof {
        cv::Mat R;                          /*!< 3x3 rotation matrix */
        cv::Mat t;                          /*!< 1x3 translation vector */
        cv::Mat P;                          /*!< 4x4 projection matrix which combines 
                                                both rotation and translation matrix */
        cv::Mat euler;                      /*!< roll, pitch, yaw in radians */

         // cunstructor
        Pose6dof() {
            R = cv::Mat::eye(3, 3, CV_64FC1);
            t = cv::Mat::zeros(3, 1, CV_64FC1);
            P = cv::Mat::eye(4, 4, CV_64FC1);
            euler = cv::Mat::zeros(1, 3, CV_64FC1);
        }
    };

    /**
     * @brief represents a camera frame
     */
    struct CameraFrame {
    private:
        bool img_exists;                    /*!< flag to specify if image exists */
        bool kpts_exists;                   /*!< flag to specify if keypoints have been computed */
        bool desc_exists;                   /*!< flag to specify if keypoint descriptors have been computed */
        cv::Mat img;                        /*!< camera image */
        std::vector<cv::KeyPoint> keypoints;/*!< 2D keypoints within camera image */
        cv::Mat descriptors;                /*!< keypoint descriptors */

    public:
        // cunstructor
        CameraFrame() :
            img_exists(false), kpts_exists(false), desc_exists(false) {}

        /* setters */
        // push image into this frame
        void pushImage(cv::Mat &img) { img_exists = true; this->img = img; }
        // push keypoints into this frame
        void pushKeypoints(std::vector<cv::KeyPoint> &keypoints) { kpts_exists = true; this->keypoints = keypoints; }
        // push descriptors into this frame
        void pushDescriptors(cv::Mat &descriptors) { desc_exists = true; this->descriptors = descriptors; }

        /* getters */
        // get image
        ResultCode getImage(cv::Mat &img) { 
            if(img_exists) {
                img = this->img;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Image does not exist" << std::endl;
                return FAILURE;
            }
        }
        // get keypoints
        ResultCode getKeypoints(std::vector<cv::KeyPoint> &keypoints) { 
            if(kpts_exists) {
                keypoints = this->keypoints;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Keypoints have not been computed yet" << std::endl;
                return FAILURE;
            }
        }
        // get descriptors
        ResultCode getDescriptors(cv::Mat &descriptors) { 
            if(desc_exists) {
                descriptors = this->descriptors;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Descriptors have not been computed yet" << std::endl;
                return FAILURE;
            }
        }
    };

    /**
     * @brief represents the available sensor information at the same time instance
     */
    struct DataFrame {
    private:
        bool kpts_matches_exists;           /*!< flag to specify if keypoints matches have been computed */
        bool f_exists;                      /*!< flag to specify if fundamental matrix has been computed */
        bool e_exists;                      /*!< flag to specify if essential matrix has been computed */
        bool pose_exists;                   /*!< flag to specify if camera relative pose has been computed */
        bool triangulated_pts_exists;       /*!< flag to specify if triangulated 3D points has been computed */
        std::vector<cv::DMatch> kptMatches; /*!< keypoint matches between previous frame and current/this frame */
        cv::Mat F;                          /*!< fundamental matrix for keypoint correspondences between previous and current/this frame */
        cv::Mat E;                          /*!< essential matrix for keypoint correspondences between previous and current/this frame */
        Pose6dof pose;                      /*!< pose */
        cv::Mat triangulatedPts;            /*!< triangulated 3D points corresponding to the tracked keypoints */

    public:
        /*! cunstructor */
        DataFrame() :
            kpts_matches_exists(false), f_exists(false), e_exists(false), pose_exists(false), triangulated_pts_exists(false) {}

        CameraFrame cameraFrame;            /*!< Image frame */

        /* setters */
        // push keypoints matches
        void pushKptsMatches(std::vector<cv::DMatch> &kptMatches) { kpts_matches_exists = true; this->kptMatches = kptMatches; }
        // push fundamental matrix
        void pushFundamentalMatrix(cv::Mat &F) { f_exists = true; this->F = F; }
        // push essential matrix
        void pushEssentialMatrix(cv::Mat &E) { e_exists = true; this->E = E; }
        // push pose
        void pushPose(Pose6dof &pose) { pose_exists = true; this->pose = pose; }
        // push triangulated points
        void pushTriangulatedPts(cv::Mat &pts) { triangulated_pts_exists = true; this->triangulatedPts = pts; }

        /* getters */
        // get keypoints matches
        ResultCode getKptsMatches(std::vector<cv::DMatch> &kptMatches) {
            if(kpts_matches_exists) {
                kptMatches = this->kptMatches;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Keypoints matches have not been computed yet" << std::endl;
                return FAILURE;
            }
        }
        // get fundamental matrix
        ResultCode getFundamentalMatrix(cv::Mat &F) {
            if(f_exists) {
                F = this->F;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Fundamental Matrix has not been computed yet" << std::endl;
                return FAILURE;
            }
        }
        // get essential matrix
        ResultCode getEssentialMatrix(cv::Mat &E) {
            if(e_exists) {
                E = this->E;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Essential Matrix has not been computed yet" << std::endl;
                return FAILURE;
            }
        }
        // get relative pose
        ResultCode getPose(Pose6dof &pose) {
            if(pose_exists) {
                pose = this->pose;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Pose has not been computed yet" << std::endl;
                return FAILURE;
            }
        }
        // get triangulated points
        ResultCode getTriangulatedPoints(cv::Mat &pts) {
            if(triangulated_pts_exists) {
                pts = this->triangulatedPts;
                return SUCCESS;
            }
            else {
                std::cout << "ERROR: Triangulated points have not been computed yet" << std::endl;
                return FAILURE;
            }
        }
    };
}; 

#endif /* TAPL_TYPES_H_ */
