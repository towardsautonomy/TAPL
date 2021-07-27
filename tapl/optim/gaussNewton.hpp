/**
 * @file    gaussNewton.hpp
 * @brief   This provides the Gauss-Newton algorithm for Non-Linear Least Squares
 * @author  Shubham Shrivastava
 */

#ifndef GAUSS_NEWTON_H_
#define GAUSS_NEWTON_H_

#include <Eigen/Dense>
#include "tapl/common/common.hpp"

namespace tapl {
    namespace optim {

        /**< Gauss-Newton Optimizer */
        class GaussNewtonOptimizer {
        public:
            /**
             * @brief Compute reprojection error of an estimated 3d point in multiple cameras
             *
             * @param[in] point3d estimated 3d point
             * @param[in] points2d projection of the same 3d point in 'n' cameras
             * @param[in] projectionMatrices projection matrices of 'n' cameras (n x 3 x 4)
             * @param[out] errors reprojection error in 'n' cameras (n x 2)
             */
            Eigen::MatrixXd reprojectionError( 
                                    const Eigen::MatrixXd &point3d,
                                    const std::vector<tapl::Point2d> &points2d,
                                    const std::vector<Eigen::MatrixXd> &projectionMatrices) {
                
                // build homogeneous point
                Eigen::MatrixXd point3d_homogeneous(4,1);
                point3d_homogeneous.block<3,1>(0,0) << point3d;
                point3d_homogeneous.block<1,1>(3,0) << 1.0;
                
                // reprojection errors
                Eigen::MatrixXd errors(projectionMatrices.size()*2,1);
                // go through each camera
                for (auto it=projectionMatrices.begin();
                          it!=projectionMatrices.end();
                          ++it) {
                    // project to camera plane
                    Eigen::MatrixXd point2d_homogeneous(1,3); 
                    point2d_homogeneous = (*it) * point3d_homogeneous;
                    Eigen::MatrixXd point2d_euclidean(2,1); 
                    point2d_euclidean << (point2d_homogeneous(0,0) / point2d_homogeneous(2,0)),
                                         (point2d_homogeneous(1,0) / point2d_homogeneous(2,0));
                    // compute reprojection error
                    auto cam_idx = std::distance(projectionMatrices.begin(), it);
                    Eigen::MatrixXd reprojection_err(2,1);
                    Eigen::MatrixXd points2d_eigen(2,1);
                    points2d_eigen << points2d[cam_idx].x, points2d[cam_idx].y;
                    reprojection_err << (point2d_euclidean - points2d_eigen);
                    errors.block<2,1>(2*cam_idx,0) = reprojection_err;
                }
                // return reprojection errors
                return errors;
            }

            /**
             * @brief Given a 3D point and its corresponding points in the image 
             *         planes, compute the associated Jacobian
             *
             * @param[in] point3d estimated 3d point
             * @param[in] projectionMatrices projection matrices of 'n' cameras (n x 3 x 4)
             *
             * @return J Jacobian Matrix
             */
            const Eigen::MatrixXd jacobian( 
                            const Eigen::MatrixXd &point3d,
                            const std::vector<Eigen::MatrixXd> &projectionMatrices) {

                // build homogeneous point
                Eigen::MatrixXd point3d_homogeneous(4,1);
                point3d_homogeneous.block<3,1>(0,0) << point3d;
                point3d_homogeneous.block<1,1>(3,0) << 1.0;

                // jacobian matrix
                Eigen::MatrixXd J(projectionMatrices.size()*2,3);
                // go through each camera
                for (auto it=projectionMatrices.begin();
                          it!=projectionMatrices.end();
                          ++it) {
                    double m1P_hat = ((*it).block<1,4>(0,0) * point3d_homogeneous)(0,0);
                    double m2P_hat = ((*it).block<1,4>(1,0) * point3d_homogeneous)(0,0);
                    double m3P_hat = ((*it).block<1,4>(2,0) * point3d_homogeneous)(0,0);

                    // build Jacobian
                    auto cam_idx = std::distance(projectionMatrices.begin(), it);
                    J.block<1,3>(2*cam_idx,0) << (((*it)(0,0)*m3P_hat - (*it)(2,0)*m1P_hat) / (m3P_hat*m3P_hat)),
                                                 (((*it)(0,1)*m3P_hat - (*it)(2,1)*m1P_hat) / (m3P_hat*m3P_hat)),
                                                 (((*it)(0,2)*m3P_hat - (*it)(2,2)*m1P_hat) / (m3P_hat*m3P_hat));
                    J.block<1,3>((2*cam_idx)+1,0) << (((*it)(1,0)*m3P_hat - (*it)(2,0)*m2P_hat) / (m3P_hat*m3P_hat)),
                                                     (((*it)(1,1)*m3P_hat - (*it)(2,1)*m2P_hat) / (m3P_hat*m3P_hat)),
                                                     (((*it)(1,2)*m3P_hat - (*it)(2,2)*m2P_hat) / (m3P_hat*m3P_hat));
                }

                // return the Jacobian Matrix
                return J;
            }

            /**
             * @brief Compute L2 reprojection error of an estimated 3d point in multiple cameras
             *
             * @param[in] point3d estimated 3d point
             * @param[in] points2d projection of the same 3d point in 'n' cameras
             * @param[in] projectionMatrices projection matrices of 'n' cameras (n x 3 x 4)
             * @param[out] errors reprojection error in 'n' cameras (n x 2)
             */
            Eigen::MatrixXd reprojectionErrorL2( 
                                    const Eigen::MatrixXd &point3d,
                                    const std::vector<tapl::Point2d> &points2d,
                                    const std::vector<Eigen::MatrixXd> &projectionMatrices) {
                
                // build homogeneous point
                Eigen::MatrixXd point3d_homogeneous(4,1);
                point3d_homogeneous.block<3,1>(0,0) << point3d;
                point3d_homogeneous.block<1,1>(3,0) << 1.0;
                
                // reprojection errors
                Eigen::MatrixXd errors(projectionMatrices.size()*2,1);
                // go through each camera
                for (auto it=projectionMatrices.begin();
                          it!=projectionMatrices.end();
                          ++it) {
                    // project to camera plane
                    Eigen::MatrixXd point2d_homogeneous(1,3); 
                    point2d_homogeneous = (*it) * point3d_homogeneous;
                    Eigen::MatrixXd point2d_euclidean(2,1); 
                    point2d_euclidean << (point2d_homogeneous(0,0) / point2d_homogeneous(2,0)),
                                         (point2d_homogeneous(1,0) / point2d_homogeneous(2,0));
                    // compute reprojection error
                    auto cam_idx = std::distance(projectionMatrices.begin(), it);
                    Eigen::MatrixXd reprojection_err(2,1);
                    Eigen::MatrixXd points2d_eigen(2,1);
                    points2d_eigen << points2d[cam_idx].x, points2d[cam_idx].y;
                    reprojection_err << (point2d_euclidean - points2d_eigen)*(point2d_euclidean - points2d_eigen);
                    errors.block<2,1>(2*cam_idx,0) = reprojection_err;
                }
                // return reprojection errors
                return errors;
            }

            /**
             * @brief Given a 3D point and its corresponding points in the image 
             *         planes, compute the associated Jacobian for L2 reprojection errors
             *
             * @param[in] point3d estimated 3d point
             * @param[in] projectionMatrices projection matrices of 'n' cameras (n x 3 x 4)
             *
             * @return J Jacobian Matrix
             */
            const Eigen::MatrixXd jacobianL2( 
                            const Eigen::MatrixXd &point3d,
                            const std::vector<tapl::Point2d> &points2d,
                            const std::vector<Eigen::MatrixXd> &projectionMatrices) {

                // build homogeneous point
                Eigen::MatrixXd point3d_homogeneous(4,1);
                point3d_homogeneous.block<3,1>(0,0) << point3d;
                point3d_homogeneous.block<1,1>(3,0) << 1.0;

                // jacobian matrix
                Eigen::MatrixXd J(projectionMatrices.size()*2,3);
                // compute L1 reprojection error
                Eigen::MatrixXd reprErr = reprojectionError(point3d, points2d, projectionMatrices);
                // go through each camera
                for (auto it=projectionMatrices.begin();
                          it!=projectionMatrices.end();
                          ++it) {
                    double m1P_hat = ((*it).block<1,4>(0,0) * point3d_homogeneous)(0,0);
                    double m2P_hat = ((*it).block<1,4>(1,0) * point3d_homogeneous)(0,0);
                    double m3P_hat = ((*it).block<1,4>(2,0) * point3d_homogeneous)(0,0);

                    // build Jacobian
                    auto cam_idx = std::distance(projectionMatrices.begin(), it);
                    J.block<1,3>(2*cam_idx,0) << 2*reprErr(2*cam_idx,0)*(((*it)(0,0)*m3P_hat - (*it)(2,0)*m1P_hat) / (m3P_hat*m3P_hat)),
                                                 2*reprErr(2*cam_idx,0)*(((*it)(0,1)*m3P_hat - (*it)(2,1)*m1P_hat) / (m3P_hat*m3P_hat)),
                                                 2*reprErr(2*cam_idx,0)*(((*it)(0,2)*m3P_hat - (*it)(2,2)*m1P_hat) / (m3P_hat*m3P_hat));
                    J.block<1,3>((2*cam_idx)+1,0) << 2*reprErr(2*cam_idx+1,0)*(((*it)(1,0)*m3P_hat - (*it)(2,0)*m2P_hat) / (m3P_hat*m3P_hat)),
                                                     2*reprErr(2*cam_idx+1,0)*(((*it)(1,1)*m3P_hat - (*it)(2,1)*m2P_hat) / (m3P_hat*m3P_hat)),
                                                     2*reprErr(2*cam_idx+1,0)*(((*it)(1,2)*m3P_hat - (*it)(2,2)*m2P_hat) / (m3P_hat*m3P_hat));
                }

                // return the Jacobian Matrix
                return J;
            }

            /**
             * @brief Given a 3D point and its corresponding points in the image 
             *         planes, compute the associated Jacobian
             *
             * @param[in] point3d initial estimate of the 3d point
             * @param[in] points2d projection of the same 3d point in 'n' cameras
             * @param[in] projectionMatrices projection matrices of 'n' cameras (n x 3 x 4)
             * @param[in] nIterations projection matrices of 'n' cameras (n x 3 x 4)
             * @param[out] optimPoint3d optimized point 3d
             * 
             * @return pair of pre-optimization and post-optimization reprojection error norms
             */
             std::pair<std::vector<float>,std::vector<float>> optimize( 
                        const tapl::Point3d &point3d,
                        const std::vector<tapl::Point2d> &points2d,
                        const std::vector<Eigen::MatrixXd> &projectionMatrices,
                        const uint16_t nIterations,
                        const float reprErrorThresh,
                        tapl::Point3d &optimPoint3d) {

                Eigen::MatrixXd point3d_eigen(3,1);
                point3d_eigen << point3d.x, point3d.y, point3d.z;
                std::vector<float> preOptimReprErrNorm(points2d.size());
                std::vector<float> postOptimReprErrNorm(points2d.size());
                // log reprojection errors
                Eigen::MatrixXd errPreOptim = reprojectionErrorL2(point3d_eigen, points2d, projectionMatrices);
                for (auto i=0; i<points2d.size(); ++i) {
                    preOptimReprErrNorm.at(i) = sqrt( errPreOptim(2*i,0)*errPreOptim(2*i,0) + 
                                                      errPreOptim(2*i+1,0)*errPreOptim(2*i+1,0));
                }
                // start optimization
                for (auto n=0; n<nIterations; ++n) {
                    errPreOptim = 
                        reprojectionErrorL2(point3d_eigen, points2d, projectionMatrices);
                    auto errPreOptimNorm = sqrt( errPreOptim(0,0)*errPreOptim(0,0) + 
                                                 errPreOptim(1,0)*errPreOptim(1,0));
                    if (errPreOptimNorm < reprErrorThresh) {break;}
                    auto J = jacobianL2(point3d_eigen, points2d, projectionMatrices);
                    point3d_eigen = point3d_eigen - (((J.transpose() * J).inverse()) * J.transpose() * errPreOptim);
                }
                Eigen::MatrixXd errPostOptim = reprojectionErrorL2(point3d_eigen, points2d, projectionMatrices);
                for (auto i=0; i<points2d.size(); ++i) {
                    postOptimReprErrNorm.at(i) = sqrt( errPostOptim(2*i,0)*errPostOptim(2*i,0) + 
                                                       errPostOptim(2*i+1,0)*errPostOptim(2*i+1,0));
                }

                // optimized point
                optimPoint3d = *(new tapl::Point3d(point3d_eigen(0), point3d_eigen(1), point3d_eigen(2)));

                // return optimization errors
                return std::make_pair(preOptimReprErrNorm, postOptimReprErrNorm);
            }
        };
    } 
} 

#endif /* GAUSS_NEWTON_H_ */