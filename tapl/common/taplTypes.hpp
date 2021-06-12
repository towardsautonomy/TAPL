/**
 * @file    taplTypes.hpp
 * @brief   This file provides type definitions used throughout this library
 * @author  Shubham Shrivastava
 */

#ifndef TAPL_TYPES_H_
#define TAPL_TYPES_H_

#include <vector>
#include <opencv2/core.hpp>
#include "tapl/common/taplLog.hpp"

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
        double x;                           /**< x-coordinate */
        double y;                           /**< y-coordinate */
        double z;                           /**< z-coordinate */
    };

    /**
     * @brief 3D Bounding-Box
     */
    struct BBox3d {
        double x_min;                       /**< min x-coordinate */
        double x_max;                       /**< max x-coordinate */
        double y_min;                       /**< min y-coordinate */
        double y_max;                       /**< max y-coordinate */
        double z_min;                       /**< min z-coordinate */
        double z_max;                       /**< max z-coordinate */
    };

    /**
     * @brief 6-DOF Camera Pose
     */
    struct Pose6dof {
        cv::Mat R;                          /**< 3x3 rotation matrix */
        cv::Mat t;                          /**< 1x3 translation vector */
        cv::Mat P;                          /**< 4x4 projection matrix which combines 
                                                both rotation and translation matrix */
        cv::Mat euler;                      /**< roll, pitch, yaw in radians */

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
        bool img_exists;                    /**< flag to specify if image exists */
        bool k_exists;                      /**< flag to specify if camera intrinsic matrix exists */
        bool kpts_exists;                   /**< flag to specify if keypoints have been computed */
        bool desc_exists;                   /**< flag to specify if keypoint descriptors have been computed */
        cv::Mat img;                        /**< camera image */
        cv::Mat K;                          /**< camera intrinsic matrix */
        std::vector<cv::KeyPoint> keypoints;/**< 2D keypoints within camera image */
        cv::Mat descriptors;                /**< keypoint descriptors */

    public:
        /* cunstructors */
        CameraFrame() :
            k_exists(false), 
            img_exists(false), 
            kpts_exists(false), 
            desc_exists(false) {}

        CameraFrame(const cv::Mat &img) :
            k_exists(false),
            img_exists(false), 
            kpts_exists(false), 
            desc_exists(false) {
                this->pushImage(img);
            }

        /* setters */
        // push image into this frame
        void pushImage(const cv::Mat &img) { img_exists = true; this->img = img; }
        // push camera intrinsic matrix into this frame
        void pushIntrinsicMatrix(const cv::Mat &K) { k_exists = true; this->K = K; }
        // push keypoints into this frame
        void pushKeypoints(const std::vector<cv::KeyPoint> &keypoints) { kpts_exists = true; this->keypoints = keypoints; }
        // push descriptors into this frame
        void pushDescriptors(const cv::Mat &descriptors) { desc_exists = true; this->descriptors = descriptors; }

        /* getters */
        // get image
        ResultCode getImage(cv::Mat &img) const { 
            if(img_exists) {
                img = this->img;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Image does not exist";
                return FAILURE;
            }
        }
        // get intrinsic matrix
        ResultCode getIntrinsicMatrix(cv::Mat &K) const { 
            if(k_exists) {
                K = this->K;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Intrinsic Matrix does not exist";
                return FAILURE;
            }
        }
        // get keypoints
        ResultCode getKeypoints(std::vector<cv::KeyPoint> &keypoints) const { 
            if(kpts_exists) {
                keypoints = this->keypoints;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Keypoints have not been computed yet";
                return FAILURE;
            }
        }
        // get descriptors
        ResultCode getDescriptors(cv::Mat &descriptors) const { 
            if(desc_exists) {
                descriptors = this->descriptors;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Descriptors have not been computed yet";
                return FAILURE;
            }
        }
    };

    /**
     * @brief represents a pair of camera frames and their properties
     */
    struct CameraPairs {
    private:
        bool kpts_matches_exists;           /**< flag to specify if keypoints matches have been computed */
        bool f_exists;                      /**< flag to specify if fundamental matrix has been computed */
        bool e_exists;                      /**< flag to specify if essential matrix has been computed */
        bool pose_exists;                   /**< flag to specify if camera relative pose has been computed */
        bool triangulated_pts_exists;       /**< flag to specify if triangulated 3D points has been computed */
        std::vector<cv::DMatch> kptMatches; /**< keypoint matches between previous frame and current/this frame */
        cv::Mat F;                          /**< fundamental matrix for keypoint correspondences between previous and current/this frame */
        cv::Mat E;                          /**< essential matrix for keypoint correspondences between previous and current/this frame */
        Pose6dof pose;                      /**< pose */
        cv::Mat triangulatedPts;            /**< triangulated 3D points corresponding to the tracked keypoints */

    public:
        /**< cunstructors */
        CameraPairs() :
            kpts_matches_exists(false), 
            f_exists(false), 
            e_exists(false), 
            pose_exists(false), 
            triangulated_pts_exists(false),
            first(new CameraFrame),
            second(new CameraFrame) {}
        
        CameraPairs(const cv::Mat &img1, const cv::Mat &img2) :
            kpts_matches_exists(false), 
            f_exists(false), 
            e_exists(false), 
            pose_exists(false), 
            triangulated_pts_exists(false) {
                *first = CameraFrame(img1);
                *second = CameraFrame(img2);
            }

        CameraFrame * first;                /**< first camera frame */
        CameraFrame * second;               /**< first camera frame used for computing F, E, etc. */

        /* setters */
        // push keypoints matches
        void pushKptsMatches(const std::vector<cv::DMatch> &kptMatches) { kpts_matches_exists = true; this->kptMatches = kptMatches; }
        // push fundamental matrix
        void pushFundamentalMatrix(const cv::Mat &F) { f_exists = true; this->F = F; }
        // push essential matrix
        void pushEssentialMatrix(const cv::Mat &E) { e_exists = true; this->E = E; }
        // push pose
        void pushPose(const Pose6dof &pose) { pose_exists = true; this->pose = pose; }
        // push triangulated points
        void pushTriangulatedPts(const cv::Mat &pts) { triangulated_pts_exists = true; this->triangulatedPts = pts; }

        /* getters */
        // get keypoints matches
        ResultCode getKptsMatches(std::vector<cv::DMatch> &kptMatches) const {
            if(kpts_matches_exists) {
                kptMatches = this->kptMatches;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Keypoints matches have not been computed yet";
                return FAILURE;
            }
        }
        // get fundamental matrix
        ResultCode getFundamentalMatrix(cv::Mat &F) const {
            if(f_exists) {
                F = this->F;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Fundamental Matrix has not been computed yet";
                return FAILURE;
            }
        }
        // get essential matrix
        ResultCode getEssentialMatrix(cv::Mat &E) const {
            if(e_exists) {
                E = this->E;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Essential Matrix has not been computed yet";
                return FAILURE;
            }
        }
        // get relative pose
        ResultCode getPose(Pose6dof &pose) const {
            if(pose_exists) {
                pose = this->pose;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Pose has not been computed yet";
                return FAILURE;
            }
        }
        // get triangulated points
        ResultCode getTriangulatedPoints(cv::Mat &pts) const {
            if(triangulated_pts_exists) {
                pts = this->triangulatedPts;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Triangulated points have not been computed yet";
                return FAILURE;
            }
        }
    };
}; 

#endif /* TAPL_TYPES_H_ */
