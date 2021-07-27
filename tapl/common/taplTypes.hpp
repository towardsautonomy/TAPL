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
     * @brief 2D Point
     */
    struct Point2d {
        double x;                           /**< x-coordinate */
        double y;                           /**< y-coordinate */

        /**< constructors */
        Point2d() {}
        Point2d(const double x, const double y) :
            x(x), y(y) {}

        /**< print operator */
        friend std::ostream& operator<<(std::ostream& os, const Point2d &p) {
            os << "[" << p.x << ", " << p.y << "]" << std::endl;
            return os;
        }
    };

    /**
     * @brief 3D Point
     */
    struct Point3d {
        double x;                           /**< x-coordinate */
        double y;                           /**< y-coordinate */
        double z;                           /**< z-coordinate */

        /**< constructors */
        Point3d() {}
        Point3d(const double x, const double y, const double z) :
            x(x), y(y), z(z) {}

        /**< print operator */
        friend std::ostream& operator<<(std::ostream& os, const Point3d &p) {
            os << "[" << p.x << ", " << p.y << ", " << p.z << "]" << std::endl;
            return os;
        }
    };

    /**
     * @brief 3D Color Point
     */
    struct Point3dColor {
        double x;                           /**< x-coordinate */
        double y;                           /**< y-coordinate */
        double z;                           /**< z-coordinate */
        uint8_t r;                          /**< red pixel value */
        uint8_t g;                          /**< green pixel value */
        uint8_t b;                          /**< blue pixel value */

        /**< constructors */
        Point3dColor() {}
        Point3dColor(const double x, const double y, const double z,
                     const uint8_t r, const uint8_t g, const uint8_t b) :
                x(x), y(y), z(z), r(r), g(g), b(b) {}

        /**< print operator */
        friend std::ostream& operator<<(std::ostream& os, const Point3dColor &p) {
            os << "xyz:[" << p.x << ", " << p.y << ", " << p.z << "]" <<
                  ", rgb:[" << std::to_string(p.r) << ", " << std::to_string(p.g) << ", " << std::to_string(p.b) << "]" << std::endl;
            return os;
        }
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
    public:
        cv::Mat R;                          /**< 3x3 rotation matrix */
        cv::Mat t;                          /**< 1x3 translation vector */
        cv::Mat T;                          /**< 4x4 projection matrix which combines 
                                                 both rotation and translation matrix */
        cv::Mat euler;                      /**< roll, pitch, yaw in radians */

        /**< constructor */
        Pose6dof() {}
        /**< constructor */
        Pose6dof(const cv::Mat &pose) : T(pose) {
            // define
            R = cv::Mat::eye(3, 3, CV_64FC1);
            t = cv::Mat::zeros(3, 1, CV_64FC1);
            euler = cv::Mat::zeros(1, 3, CV_64FC1);
            // extract rotation and translation
            pose(cv::Rect(0, 0, 3, 3)).copyTo(this->R);
            pose(cv::Rect(3, 0, 1, 3)).copyTo(this->t);
            // // compute euler angles
            // float roll, pitch, yaw;
            // rodrigues2euler(R, roll, pitch, yaw);
            // this->euler.at<float>(0,0) = roll;
            // this->euler.at<float>(0,1) = pitch;
            // this->euler.at<float>(0,2) = yaw;
        }

        // /**< Rodrigues to Euler angle conversion */
        // void rodrigues2euler( cv::Mat &R, 
        //                       float &roll, 
        //                       float &pitch, 
        //                       float &yaw ) {
        //     float cosine_for_pitch = sqrt(pow(R.at<float>(0,0), 2) + pow(R.at<float>(1,0), 2));
        //     bool is_singular = false;
        //     if (cosine_for_pitch < 10e-6) is_singular = true;
        //     if (is_singular == false) {
        //         roll = atan2(R.at<float>(2,1), R.at<float>(2,2));
        //         pitch = atan2(-R.at<float>(2,0), cosine_for_pitch);
        //         yaw = atan2(R.at<float>(1,0), R.at<float>(0,0));
        //     }
        //     else {
        //         roll = 0;
        //         pitch = atan2(-R.at<float>(2,0), cosine_for_pitch);
        //         yaw = atan2(-R.at<float>(1,2), R.at<float>(1,1));
        //     }
        // }
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
        /**< constructors */
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

        /**< setters */
        // push image into this frame
        void pushImage(const cv::Mat &img) { img_exists = true; this->img = img; }
        // push camera intrinsic matrix into this frame
        void pushIntrinsicMatrix(const cv::Mat &K) { k_exists = true; this->K = K; }
        // push keypoints into this frame
        void pushKeypoints(const std::vector<cv::KeyPoint> &keypoints) { kpts_exists = true; this->keypoints = keypoints; }
        // push descriptors into this frame
        void pushDescriptors(const cv::Mat &descriptors) { desc_exists = true; this->descriptors = descriptors; }

        /**< getters */
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
        bool tracked_kpts_exists;           /**< flag to specify if triangulated 3D points has been computed */
        bool triangulated_pts_exists;       /**< flag to specify if triangulated 3D points has been computed */
        std::vector<cv::DMatch> kptMatches; /**< keypoint matches between previous frame and current/this frame */
        cv::Mat F;                          /**< fundamental matrix for keypoint correspondences between previous and current/this frame */
        cv::Mat E;                          /**< essential matrix for keypoint correspondences between previous and current/this frame */
        Pose6dof pose;                      /**< pose */
        std::vector<tapl::Point2d> trackedKpts;           /**< tracked 2D keypoints used for triangulation */
        std::vector<tapl::Point3dColor> triangulatedPts;  /**< triangulated 3D points in the first camera's coordinate frame */

    public:
        /**< cunstructors */
        CameraPairs() :
            kpts_matches_exists(false), 
            f_exists(false), 
            e_exists(false), 
            pose_exists(false), 
            tracked_kpts_exists(false),
            triangulated_pts_exists(false),
            first(new CameraFrame),
            second(new CameraFrame) {}
        
        CameraPairs(const cv::Mat &img1, const cv::Mat &img2) :
            kpts_matches_exists(false), 
            f_exists(false), 
            e_exists(false), 
            pose_exists(false), 
            tracked_kpts_exists(false),
            triangulated_pts_exists(false),
            first(new CameraFrame(img1)),
            second(new CameraFrame(img2)) {}

        CameraPairs(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &K) :
            kpts_matches_exists(false), 
            f_exists(false), 
            e_exists(false), 
            pose_exists(false), 
            tracked_kpts_exists(false),
            triangulated_pts_exists(false) {
                first = new CameraFrame(img1);
                first->pushIntrinsicMatrix(K);
                second = new CameraFrame(img2);
                second->pushIntrinsicMatrix(K);
            }

        /**< camera frames */
        CameraFrame * first;                /**< first camera frame */
        CameraFrame * second;               /**< second camera frame used for computing F, E, relative pose, etc. */

        /**< setters */
        // push keypoints matches
        void pushKptsMatches(const std::vector<cv::DMatch> &kptMatches) { kpts_matches_exists = true; this->kptMatches = kptMatches; }
        // push fundamental matrix
        void pushFundamentalMatrix(const cv::Mat &F) { f_exists = true; this->F = F; }
        // push essential matrix
        void pushEssentialMatrix(const cv::Mat &E) { e_exists = true; this->E = E; }
        // push pose
        void pushPose(const Pose6dof &pose) { pose_exists = true; this->pose = pose; }
        // push tracked 2d keypoints
        void pushTrackedKpts(std::vector<tapl::Point2d> &pts) { 
            tracked_kpts_exists = true; 
            for (auto &point : pts) this->trackedKpts.push_back(point);
        }
        // push triangulated points
        void pushTriangulatedPts(std::vector<tapl::Point3dColor> &pts) { 
            triangulated_pts_exists = true; 
            for (auto &point : pts) this->triangulatedPts.push_back(point);
        }

        /**< getters */
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
        // get tracked 2D keypoints
        ResultCode getTrackedKpts(std::vector<tapl::Point2d> &pts) const {
            if(tracked_kpts_exists) {
                pts = this->trackedKpts;
                return SUCCESS;
            }
            else {
                TLOG_ERROR << "Tracked keypoints not found";
                return FAILURE;
            }
        }
        // get triangulated points
        ResultCode getTriangulatedPoints(std::vector<tapl::Point3dColor> &pts) const {
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
