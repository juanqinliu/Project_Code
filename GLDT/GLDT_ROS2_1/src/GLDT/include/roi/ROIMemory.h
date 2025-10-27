#ifndef ROIMEMORY_H
#define ROIMEMORY_H

#include <opencv2/opencv.hpp>
#include <deque>
#include "common/Logger.h"

namespace tracking {

// ROI memory for track recovery and quality estimation
class ROIMemory {
public:
    int track_id;
    cv::Point2f last_position;
    int last_seen_frame;
    std::deque<float> confidence_history;  // max length 15
    std::deque<cv::Point2f> position_history;  // max length 15
    std::deque<cv::Size2f> size_history;  // store target size history
    int lost_duration;
    float quality_score;
    
    ROIMemory(int id, cv::Point2f pos, int frame);
    void updateMemory(cv::Point2f position, float confidence, int frame_id);
    void updateMemory(cv::Point2f position, cv::Size2f size, float confidence, int frame_id);
    void incrementLost();
    bool isReliable() const;
    
    // Appearance template helpers
    void setTemplate(const cv::Mat& tmpl);
    const cv::Mat& getTemplate() const { return appearance_template_; }
    bool hasTemplate() const { return !appearance_template_.empty(); }
    
private:
    static const size_t MAX_CONFIDENCE_HISTORY = 15;
    static const size_t MAX_POSITION_HISTORY = 15;
    static const size_t MAX_SIZE_HISTORY = 15;
    
    // Stored appearance template for recovery
    cv::Mat appearance_template_;
};

} // namespace tracking

#endif // ROIMEMORY_H 