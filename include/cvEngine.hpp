#include "dataStructures.hpp"
#include "matching2D.hpp"
#include "ringBuffer.hpp"

namespace tapl {
    namespace cve {
        int detectAndMatchKpts(RingBuffer<tapl::DataFrame> &dataBuffer, bool verbose=false);
        int getFundamentalMatrix(RingBuffer<tapl::DataFrame> &dataBuffer);
        int getEssentialMatrix(RingBuffer<tapl::DataFrame> &dataBuffer, cv::Mat &camera_matrix);
        int getPose(RingBuffer<tapl::DataFrame> &dataBuffer, cv::Mat &camera_matrix);
    }
}