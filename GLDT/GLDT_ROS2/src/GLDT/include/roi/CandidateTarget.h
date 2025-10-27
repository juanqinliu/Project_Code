#ifndef CANDIDATETARGET_H
#define CANDIDATETARGET_H

#include <opencv2/opencv.hpp>
#include <deque>
#include "common/Config.h"

namespace tracking {

// Candidate target tracked before promotion to ROI
class CandidateTarget {
public:
    int target_id;
    std::deque<int> detections; // max length 30
    std::deque<cv::Point2f> centers; // max length 30
    int first_frame;
    int last_frame;
    int miss_count;
    
    CandidateTarget(int id, cv::Point2f center, int frame_id);
    void update(cv::Point2f center, int frame_id);
    void miss(int frame_id);
    int detectionCount() const;
    cv::Point2f avgCenter() const;
    bool shouldConfirm(const Config& config) const; // whether to promote to confirmed
    
    static const size_t MAX_HISTORY = 30;
};


} // namespace tracking

#endif // CANDIDATETARGET_H 