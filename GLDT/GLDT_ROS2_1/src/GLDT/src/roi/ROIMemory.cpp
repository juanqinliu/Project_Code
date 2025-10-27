#include "roi/ROIManager.h"
#include <algorithm>

namespace tracking {

// ROIMemory implementation
ROIMemory::ROIMemory(int id, cv::Point2f pos, int frame) 
    : track_id(id), last_position(pos), last_seen_frame(frame), 
      lost_duration(0), quality_score(1.0f) {
}

void ROIMemory::updateMemory(cv::Point2f position, float confidence, int frame_id) {
    // Update memory information
    last_position = position;
    last_seen_frame = frame_id;
    confidence_history.push_back(confidence);
    if (confidence_history.size() > MAX_CONFIDENCE_HISTORY) {
        confidence_history.pop_front();
    }
    position_history.push_back(position);
    if (position_history.size() > MAX_POSITION_HISTORY) {
        position_history.pop_front();
    }
    lost_duration = 0;
    quality_score = std::min(1.0f, quality_score + 0.1f);
}

void ROIMemory::updateMemory(cv::Point2f position, cv::Size2f size, float confidence, int frame_id) {

    updateMemory(position, confidence, frame_id);

    size_history.push_back(size);
    if (size_history.size() > MAX_SIZE_HISTORY) {
        size_history.pop_front();
    }
}

void ROIMemory::incrementLost() {

    lost_duration += 1;
    quality_score = std::max(0.1f, quality_score - 0.03f);
}

bool ROIMemory::isReliable() const {

    return (lost_duration <= 30 &&
            quality_score > 0.2f &&
            position_history.size() >= 2);
}

void ROIMemory::setTemplate(const cv::Mat& tmpl) {
    if (!tmpl.empty()) {

        cv::Mat resized;
        cv::resize(tmpl, resized, cv::Size(64, 64));
    }
}

} // namespace tracking 