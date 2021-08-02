#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <opencv2/opencv.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include "imgProc.hpp"
#include "sfm.hpp"
#include "tapl/pte/ptEngine.hpp"

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

/**
 * Estimate four possible R and T from the Essential Matrix
 */
std::vector<Eigen::MatrixXd> computeInitialRTfromE(const Eigen::MatrixXd &E) {
   
    // SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // build 'Z' and 'W'
    Eigen::MatrixXd Z(3,3);
    Z << 0, 1, 0, 
        -1, 0, 0, 
         0, 0, 0;
    Eigen::MatrixXd W(3,3);
    W << 0, -1, 0, 
         1,  0, 0, 
         0,  0, 1;

    // E can be re-written as E=MQ; where M=U.Z.U^T and Q=U.W.V^T or Q=U.W^T.V^T
    Eigen::MatrixXd M = svd.matrixU() * Z * svd.matrixU().transpose();
    Eigen::MatrixXd Q1 = svd.matrixU() * W * svd.matrixV().transpose();
    Eigen::MatrixXd Q2 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();

    // R can be computed as R=(det Q)Q
    Eigen::MatrixXd R1 = Q1.determinant() * Q1;
    Eigen::MatrixXd R2 = Q2.determinant() * Q2;

    // E = UΣV^T, T is simply either u3 or −u3, where u3 is the third column vector of U
    Eigen::MatrixXd T1 = svd.matrixU().col(2);
    Eigen::MatrixXd T2 = -svd.matrixU().col(2);

    // compose four possible RT
    std::vector<Eigen::MatrixXd> RT;
    // R1 and T1
    Eigen::MatrixXd R1T1(3,4);
    R1T1.block<3,3>(0,0) = R1;
    R1T1.block<3,1>(0,3) = T1;
    RT.push_back(R1T1);
    // R1 and T2
    Eigen::MatrixXd R1T2(3,4);
    R1T2.block<3,3>(0,0) = R1;
    R1T2.block<3,1>(0,3) = T2;
    RT.push_back(R1T2);
    // R2 and T1
    Eigen::MatrixXd R2T1(3,4);
    R2T1.block<3,3>(0,0) = R2;
    R2T1.block<3,1>(0,3) = T1;
    RT.push_back(R2T1);
    // R2 and T2
    Eigen::MatrixXd R2T2(3,4);
    R2T2.block<3,3>(0,0) = R2;
    R2T2.block<3,1>(0,3) = T2;
    RT.push_back(R2T2);

    // return
    return RT;
}

/**
 * Linear estimate of the 3d point
 */
tapl::Point3d tapl::cve::StructureFromMotion::linearEstimate3dPt( 
                    const std::vector<tapl::Point2d> &point2d,                                         
                    const std::vector<Eigen::MatrixXd> &projectionMatrices ) {
    // We can re-write [p x MP = 0] in the form [AP = 0] and solve for P 
    // by decomposing A using SVD
    // let's formulate the A matrix
    Eigen::MatrixXd A(point2d.size()*2, 4);
    for (auto i=0; i<point2d.size(); ++i) {
        A.row(i*2) << (point2d.at(i).x * projectionMatrices.at(i).row(2)) - 
                                    projectionMatrices.at(i).row(0);
        A.row((i*2)+1) << (point2d.at(i).y * projectionMatrices.at(i).row(2)) - 
                                    projectionMatrices.at(i).row(1);
    }
    // SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // P can be obtained as the last column of v or last row of v_transpose
    Eigen::MatrixXd pt3d_homogeneous = svd.matrixV().col(3);
    // homogeneous to cartesian coordinate conversion
    tapl::Point3d pt3d( pt3d_homogeneous(0,0) / pt3d_homogeneous(3,0), 
                        pt3d_homogeneous(1,0) / pt3d_homogeneous(3,0), 
                        pt3d_homogeneous(2,0) / pt3d_homogeneous(3,0) );
    return pt3d;
}

/**
 * Non-linear estimate of the 3d point
 */
tapl::Point3d tapl::cve::StructureFromMotion::nonLinearEstimate3dPt( 
                    const std::vector<tapl::Point2d> &point2d,                                         
                    const std::vector<Eigen::MatrixXd> &projectionMatrices,
                    std::pair<std::vector<float>,std::vector<float>> &reprojectionErrors,
                    const uint16_t nIterations,
                    const float reprErrorThresh ) {
    // compute linear estimate of the 3d point
    tapl::Point3d pt3d_initial = linearEstimate3dPt(point2d, projectionMatrices);
    // perform non-linear least squares optimization
    tapl::Point3d pt3d_optimize;
    reprojectionErrors = \
        gnOptim.optimize(pt3d_initial, point2d, projectionMatrices, nIterations, reprErrorThresh, pt3d_optimize);
    return pt3d_optimize;
}

/**
 * Estimate R, T, and triengulated points
 */
std::pair<Eigen::MatrixXd, std::vector<tapl::Point3d>> 
    tapl::cve::StructureFromMotion::computeSFM( 
                    const Eigen::MatrixXd &E, 
                    const std::vector<std::vector<tapl::Point2d>> &points2d, 
                    const Eigen::MatrixXd &projectionMat1,
                    const float &maxReprojectionErr ) {
    // compute four possible RT
    std::vector<Eigen::MatrixXd> RT = computeInitialRTfromE(E);
    // projection matrices
    std::vector<Eigen::MatrixXd> projectionMatrices(2);
    projectionMatrices.at(0) = projectionMat1;
    // set of triangulated 3d points
    std::vector<std::vector<tapl::Point3d>> pointsTriangulatedRef(4, std::vector<tapl::Point3d>(points2d.size()));
    std::vector<std::vector<tapl::Point3d>> pointsTriangulated(4, std::vector<tapl::Point3d>(points2d.size()));
    // reprojection errors
    std::vector<std::vector<std::pair<std::vector<float>,std::vector<float>>>> 
        reprojectionErrors(4, std::vector<std::pair<std::vector<float>,std::vector<float>>>(points2d.size()));
    // iterate over 4 possible R & T
    for (auto it1=RT.begin(); it1!= RT.end(); ++it1) {
        auto rtIdx = std::distance(RT.begin(), it1);
        // build projection matrix
        Eigen::MatrixXd projectionMat2 = projectionMat1 * (*it1);
        // TODO: Update the implementation so that we have 1 projection matrix per camera
        projectionMatrices.at(1) = projectionMat2;
        // go through each point and triangulate it through multiple-views
        for (auto it2=points2d.begin(); it2!=points2d.end(); ++it2) {
            auto ptIdx = std::distance(points2d.begin(), it2);
            // triangulate 3d point
            std::pair<std::vector<float>,std::vector<float>> reprojectionError;
            auto pt3d = this->nonLinearEstimate3dPt(*it2, projectionMatrices, reprojectionError);
            reprojectionErrors[rtIdx][ptIdx] = reprojectionError;
            pointsTriangulatedRef[rtIdx][ptIdx] = pt3d;

            // convert to homogeneous coordinate system
            Eigen::MatrixXd point3dHomogeneous(4,1);
            point3dHomogeneous << pt3d.x, pt3d.y, pt3d.z, 1.0;
            auto ptsTransformed = (*it1) * point3dHomogeneous;
            pointsTriangulated[rtIdx][ptIdx] = *(new tapl::Point3d(ptsTransformed(0), ptsTransformed(1), ptsTransformed(2)));
        }
    }

    // for each RT, count number of points that fall in front of all cameras.
    // the one with maximum number of points will correspond to the correct RT
    size_t correctIdx = 0;
    uint16_t maxCount = 0;
    for (auto i=0; i<RT.size(); ++i) {
        uint16_t count = 0;
        for (auto j=0; j<points2d.size(); ++j) {
            if ((pointsTriangulatedRef[i][j].z > 0.0) && 
                (pointsTriangulated[i][j].z > 0.0)) {
                   count ++;
               }
        }
        if (count > maxCount) {
            maxCount = count;
            correctIdx = i;
        }
    }

    Eigen::MatrixXd correctRT = RT[correctIdx];
    std::vector<tapl::Point3d> correctTriangulatedPtsRef = pointsTriangulatedRef[correctIdx];
    std::vector<tapl::Point3d> correctTriangulatedPts = pointsTriangulated[correctIdx];

    // use PnP to get refined pose
    // get pose using ransac
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);    
    const int iterationsCount = 500;       // number of Ransac iterations. default 100
    const float reprojectionError = 1.0;    // maximum allowed distance to consider it an inlier. default 8.0
    const float confidence = 0.99;          // RANSAC successful confidence. default 0.99
    const bool useExtrinsicGuess = true;   // default false
    const int flags = cv::SOLVEPNP_ITERATIVE;
    cv::Mat inliers;
    std::vector<cv::Point2f> kpts2d_;
    std::vector<cv::Point3f> kpts3d_;
    std::vector<uint16_t> idxRemove;
    for (auto i=0; i<points2d.size(); ++i) {
        // filter out the points outside of region-of-interest and the points with high reprojection error
        if ((correctTriangulatedPts[i].z > 0.0) && (correctTriangulatedPtsRef[i].z > 0.0) &&
            (correctTriangulatedPtsRef[i].z > this->minXYZ.at(2)) && (correctTriangulatedPtsRef[i].z < this->maxXYZ.at(2)) &&
            (fabs(correctTriangulatedPtsRef[i].x) > this->minXYZ.at(0)) && (fabs(correctTriangulatedPtsRef[i].x) < this->maxXYZ.at(0)) &&
            (fabs(correctTriangulatedPtsRef[i].y) > this->minXYZ.at(1)) && (fabs(correctTriangulatedPtsRef[i].y) < this->maxXYZ.at(1)) &&
            reprojectionErrors[correctIdx][i].second.at(0) < maxReprojectionErr ) { // post-optimization reprojection error in the first camera frame
            
            // compute epipolar lines
            Eigen::MatrixXf pt2(3,1);
            pt2 << points2d[i][1].x, points2d[i][1].y, 1.0;
            Eigen::MatrixXf epipolarLine1 = (E.transpose().cast <float> ()) * pt2.cast <float> ();
            epipolarLine1 = epipolarLine1 / epipolarLine1(2,0); // normalize
            // compute distance between epipolar line to the point in the other image
            auto dist2ep = fabs(static_cast<float>(epipolarLine1(0,0)) * points2d[i][0].x + 
                                static_cast<float>(epipolarLine1(1,0)) * points2d[i][0].y + 
                                static_cast<float>(epipolarLine1(2,0))) / 
                            sqrt(pow(static_cast<float>(epipolarLine1(0,0)), 2) + 
                                 pow(static_cast<float>(epipolarLine1(1,0)), 2));
            // if (dist2ep < 50.0) { 
            // points in the camera plane for which the pose needs to be computed
            kpts2d_.push_back(cv::Point2f(points2d[i][1].x, points2d[i][1].y));
            // 3d points in the coordinate of reference camera frame
            kpts3d_.push_back(cv::Point3f(correctTriangulatedPtsRef[i].x, correctTriangulatedPtsRef[i].y, correctTriangulatedPtsRef[i].z));
            // }
            // else idxRemove.push_back(i);
        }
        else idxRemove.push_back(i);
    }
    // remove outliers
    for (auto it_idx=idxRemove.end()-1; it_idx!=idxRemove.begin()-1; --it_idx) {
        correctTriangulatedPtsRef.erase(correctTriangulatedPtsRef.begin()+(*it_idx));
    } 
    if(kpts3d_.size() >= 6) {
        cv::solvePnPRansac(kpts3d_, kpts2d_, this->K, this->dist_coeff, rvec, tvec,
                            useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                            inliers, flags );
        float inliers_ratio = static_cast<float>(inliers.size().height) / static_cast<float>(kpts3d_.size());
        // TLOG_DEBUG << "Inliers ratio [" << kpts3d_.size() << "/" << inliers.size().height << "] = [" << inliers_ratio << "]";

        // get global pose in world reference frame
        // check for NaNs
        rvec.convertTo(rvec, CV_32F);
        tvec.convertTo(tvec, CV_32F);
        cv::patchNaNs(rvec, 0.0); // replace NaN with 0.0
        cv::patchNaNs(tvec, 0.0); // replace NaN with 0.0
        // rotation matrix
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        // translation matrix
        cv::Mat t = cv::Mat::eye(4,4, CV_32FC1);
        tvec.copyTo(t(cv::Rect(3,0,1,3)));
        // build an RT matrix
        Eigen::MatrixXd R_eigen(3,3);
        cv2eigen(R, R_eigen);
        Eigen::MatrixXd RT_pnp(3,4);
        RT_pnp.block<3,3>(0,0) << R_eigen;
        RT_pnp.block<3,1>(0,3) << tvec.at<float>(0,0), tvec.at<float>(0,1), tvec.at<float>(0,2);

        return std::pair<Eigen::MatrixXd, std::vector<tapl::Point3d>>(RT_pnp, correctTriangulatedPtsRef);
    }
    else {
        TLOG_WARN << "PnP failed";
        return std::pair<Eigen::MatrixXd, std::vector<tapl::Point3d>>(correctRT, correctTriangulatedPtsRef);
    }
}

/**
 * Structure-from-Motion constructor
 */
tapl::cve::StructureFromMotion::StructureFromMotion( 
                                     const std::vector<cv::Mat> &images, 
                                     const cv::Mat &K,
                                     const std::vector<float> &minXYZ,
                                     const std::vector<float> &maxXYZ,
                                     const bool verbose) {
    // copy inputs to private variable
    this->images = images;
    this->K = K;
    this->minXYZ = minXYZ;
    this->maxXYZ = maxXYZ;
    this->verbose = verbose;
    // make bundles of 'm' images
}

/**
 * Structure-from-Motion implementation
 */
tapl::ResultCode tapl::cve::StructureFromMotion::process(
                                std::vector<tapl::Point3dColor> &points,
                                std::vector<tapl::Pose6dof> &poses,
                                std::vector<tapl::CameraPairs> &framePairs) {

    // global pose
    Eigen::Matrix4f globalPose = Eigen::Matrix4f::Identity();
    // Go through each camera frame
    for (auto it=this->images.begin()+1; it!=this->images.end(); ++it) {
        if (this->verbose) TLOG_INFO << "processing frames [" << 
                                        std::distance(images.begin(), it) <<
                                        "] and [" << 
                                        std::distance(images.begin(), it+1) << "]";

        tapl::CameraPairs camPairs(*(it-1), *it, this->K);

        // compute fundamental matrix
        if (tapl::cve::computeFundamentalMatrix(camPairs) != tapl::SUCCESS) {
            TLOG_ERROR << "could not compute fundamental matrix";
            return tapl::FAILURE;
        }

        // retrieve fundamental matrix
        cv::Mat F;
        if (camPairs.getFundamentalMatrix(F) !=  tapl::SUCCESS) {
            TLOG_ERROR << "could not retrieve fundamental matrix";
            return tapl::FAILURE;
        }

        // compute essential matrix
        cv::Mat E = this->K.t() * F * this->K;

        // get keypoints
        std::vector<cv::KeyPoint> kpts1;
        if (camPairs.first->getKeypoints(kpts1) !=  tapl::SUCCESS) {
            TLOG_ERROR << "could not retrieve keypoints";
            return tapl::FAILURE;
        }
        std::vector<cv::KeyPoint> kpts2;
        if (camPairs.second->getKeypoints(kpts2) !=  tapl::SUCCESS) {
            TLOG_ERROR << "could not retrieve keypoints";
            return tapl::FAILURE;
        }

        // get keypoints matches
        std::vector<cv::DMatch> kptMatches;
        if (camPairs.getKptsMatches(kptMatches) !=  tapl::SUCCESS) {
            TLOG_ERROR << "could not retrieve keypoints matches";
            return tapl::FAILURE;
        }

        // build vector of 'n' points in 'm' camera frames (n x m - Point2d)
        std::vector<std::vector<tapl::Point2d>> kptSet;
        std::vector<tapl::Point2d> trackedKpts;
        for (auto it_match=kptMatches.begin(); it_match!=kptMatches.end(); ++it_match) {
            if (it_match->distance < 100.0) {
                tapl::Point2d pt1 = tapl::Point2d( kpts1.at((*it_match).queryIdx).pt.x, 
                                                   kpts1.at((*it_match).queryIdx).pt.y );
                tapl::Point2d pt2 = tapl::Point2d( kpts2.at((*it_match).trainIdx).pt.x, 
                                                   kpts2.at((*it_match).trainIdx).pt.y );
                trackedKpts.push_back(pt1);
                // TODO: compute corresponding lines, find and reject outliers

                // this point in all cameras
                std::vector<tapl::Point2d> ptCameras = {pt1, pt2};
                kptSet.push_back(ptCameras);
            }
        }

        // draw the matches 
        cv::Mat matchImg;
        cv::drawMatches(*(it-1), kpts1, *(it), kpts2,
                        kptMatches, matchImg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // convert E to eigen format
        Eigen::MatrixXd eigenE(3,3); 
        cv2eigen(E, eigenE);
        // convert K to eigen format
        Eigen::MatrixXd eigenK(3,3); 
        Eigen::MatrixXd projectionMat1(3,4); 
        cv2eigen(this->K, eigenK);
        projectionMat1.block<3,3>(0,0) = eigenK;
        projectionMat1(3,3) = 1.0;
        auto sfm = computeSFM(eigenE, kptSet, projectionMat1);
        Eigen::MatrixXd RT = sfm.first;
        std::vector<tapl::Point3d> triangulatedPts = sfm.second;
        // Get point color
        std::vector<tapl::Point3dColor> triangulatedPtsColor;
        cv::Mat imgProj = cv::Mat::zeros((it-1)->rows, (it-1)->cols, CV_8UC3);
        for ( auto &pt3d : triangulatedPts) {
            Eigen::MatrixXd eigenPt3d(3,1);
            eigenPt3d << pt3d.x, pt3d.y, pt3d.z;
            auto eigenPt2dHomogeneous = eigenK * eigenPt3d;
            float pxX, pxY;
            pxX = eigenPt2dHomogeneous(0,0) / eigenPt2dHomogeneous(2,0);
            pxY = eigenPt2dHomogeneous(1,0) / eigenPt2dHomogeneous(2,0);
            // check if the coordinate is within limits
            cv::Vec3b rgb;
            if ((pxX >= 0) && (pxX < (it-1)->cols) &&
                (pxY >= 0) && (pxY < (it-1)->rows)) {
                rgb = (it-1)->at<cv::Vec3b>(static_cast<int>(pxY),static_cast<int>(pxX));
                imgProj.at<cv::Vec3b>(static_cast<int>(pxY),static_cast<int>(pxX)) = rgb;
            }
            else rgb = cv::Vec3b(0, 0, 0);
            tapl::Point3dColor pt3dColor( pt3d.x, pt3d.y, pt3d.z, static_cast<uint8_t>(rgb[0]), 
                                          static_cast<uint8_t>(rgb[1]), static_cast<uint8_t>(rgb[2]));
            triangulatedPtsColor.push_back(pt3dColor);
        }

        // plot
        Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
        P.block<3,4>(0,0) = RT.cast<float>();
        Eigen::Matrix4f Pi = Eigen::Matrix4f::Identity();
        Pi.block<3,3>(0,0) = P.block<3,3>(0,0).transpose();
        Pi.block<3,1>(0,3) = -P.block<3,3>(0,0).transpose() * P.block<3,1>(0,3);
        globalPose = globalPose * Pi;

        // compute pose 
        cv::Mat cvPose;
        eigen2cv(globalPose, cvPose);
        tapl::Pose6dof pose(cvPose);
        // add to camera 
        camPairs.pushPose(pose);
        camPairs.pushTrackedKpts(trackedKpts);
        camPairs.pushTriangulatedPts(triangulatedPtsColor);
        // push to the output vector
        poses.push_back(pose);
        for (auto &point : triangulatedPtsColor) points.push_back(point);
        // push camera pairs
        framePairs.push_back(camPairs);
    }

    // return success
    return tapl::SUCCESS;
}