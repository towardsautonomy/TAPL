/**
 * @file    sfm.hpp
 * @brief   This provides the implementation of structure-from-motion algorithm
 * @author  Shubham Shrivastava
 */

#ifndef SFM_H_
#define SFM_H_

#include "tapl/common/taplLog.hpp"
#include "tapl/common/taplTypes.hpp"

namespace tapl {
    namespace cve {
        /** 
         * @brief This function performs structure-from-motion given a set of camera frames
         *
         * @param[in] frames image frames from which structure-from-motion is to be computed
         * @param[out] points point-cloud corresponding to keypoints in the first camera's coordinate frame
         * @param[out] poses poses of each camera frame
         * 
         * @return tapl::SUCCESS if success
         * @return tapl::FAILURE if failure 
         */
        tapl::ResultCode sfm(std::vector<tapl::CameraFrame> &frames, 
                             std::vector<tapl::Point3d> &points,
                             std::vector<tapl::Pose6dof> &poses);

    } 
} 

#endif /* SFM_H_ */