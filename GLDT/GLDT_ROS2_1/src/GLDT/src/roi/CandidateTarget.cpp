#include "roi/CandidateTarget.h"
#include <numeric>

namespace tracking {

// CandidateTarget Implementation
CandidateTarget::CandidateTarget(int id, cv::Point2f center, int frame_id) 
    : target_id(id), first_frame(frame_id), last_frame(frame_id), miss_count(0) {
    detections.push_back(1);
    centers.push_back(center);
    if (detections.size() > MAX_HISTORY) detections.pop_front();
    if (centers.size() > MAX_HISTORY) centers.pop_front();
}

void CandidateTarget::update(cv::Point2f center, int frame_id) {
    detections.push_back(1);
    centers.push_back(center);
    last_frame = frame_id;
    miss_count = 0;
    if (detections.size() > MAX_HISTORY) detections.pop_front();
    if (centers.size() > MAX_HISTORY) centers.pop_front();
}

void CandidateTarget::miss(int frame_id) {
    detections.push_back(0);
    last_frame = frame_id;
    miss_count++;
    if (detections.size() > MAX_HISTORY) detections.pop_front();
}

int CandidateTarget::detectionCount() const {
    // Count the number of 1s in detections
    return std::count(detections.begin(), detections.end(), 1);
}

cv::Point2f CandidateTarget::avgCenter() const {
    if (centers.empty()) return cv::Point2f(0, 0);
    cv::Point2f avg(0, 0);
    for (const auto& center : centers) {
        avg += center;
    }
    avg.x /= centers.size();
    avg.y /= centers.size();
    return avg;
}

bool CandidateTarget::shouldConfirm(const Config& config) const {
    // Only check the number of detections
    return detectionCount() >= config.candidate_confirm_frames;
}


} // namespace tracking 