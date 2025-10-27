#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include "common/Config.h"
#include "roi/ROIMemory.h"

namespace tracking {

class ROI {
public:
    int id;
    cv::Rect bbox;
    int no_detection_count;
    int no_tracking_count;
    int last_updated;
    bool is_merged;
    float safety_ratio;
    std::vector<int> track_ids;
    std::unordered_map<int, std::unique_ptr<ROIMemory>> track_memories;
    
    ROI(int roi_id, cv::Rect rect);
    
    // Basic geometry
    cv::Point2f center() const;
    int area() const;
    int x() const { return bbox.x; }
    int y() const { return bbox.y; }
    int width() const { return bbox.width; }
    int height() const { return bbox.height; }

    // Safety zone management
    cv::Rect safetyBbox() const;
    bool containsPoint(cv::Point2f point, float margin = 0) const;
    bool isInSafetyZone(cv::Point2f point) const;
    std::vector<cv::Point2f> getSafetyZoneViolations(const std::vector<cv::Point2f>& points) const;

    // Position update
    void updatePosition(int new_x, int new_y, int new_width = -1, int new_height = -1);

    // Track memory utilities
    std::optional<ROIMemory*> findCandidateForRecovery(cv::Point2f detection_center, const Config& config);

    // Adaptive ROI update (center/size based on tracks)
    bool adaptiveUpdate(const std::vector<cv::Point2f>& track_centers, 
                       int frame_width, int frame_height, bool force_update = false);
    
    // Adaptive ROI update with track sizes
    bool adaptiveUpdateWithTrackInfo(const std::vector<std::pair<cv::Point2f, cv::Size2f>>& track_info,
                                    int frame_width, int frame_height, const Config& config, bool force_update = false);
    
    // Helpers for adaptive update
    cv::Point2f calculateTargetsCenter(const std::vector<cv::Point2f>& track_centers) const;
    cv::Size calculateRequiredSize(const std::vector<cv::Point2f>& track_centers) const;
    cv::Size calculateSingleTargetSize(const cv::Point2f& target_center) const;
    cv::Size calculateMultiTargetSize(const std::vector<cv::Point2f>& track_centers) const;
    
    // ROI size calculation using track size info
    cv::Size calculateAdaptiveROISize(const std::vector<std::pair<cv::Point2f, cv::Size2f>>& track_info, 
                                     const Config& config) const;

    // ==================== ROI optimization ====================
    
    // Overlap ratio between ROIs
    float calculateOverlapRatio(const ROI& other) const;
    
    // Merged bounding box between two ROIs
    cv::Rect calculateMergedBbox(const ROI& other, int frame_width, int frame_height, int margin = 0) const;
    
    // Clamp to frame bounds
    cv::Rect validateBounds(int frame_width, int frame_height) const;
    
    // Merge decision
    bool shouldMergeWith(const ROI& other) const;
    
    // Split decision
    bool shouldSplit(const std::vector<cv::Point2f>& track_centers, int base_size = 300) const;
    
    // Generate split configs
    std::vector<cv::Rect> generateSplitConfigs(const std::vector<cv::Point2f>& track_centers,
                                              int frame_width, int frame_height, int base_size = 300) const;
    
    // Target clustering
    std::vector<std::vector<int>> clusterTrackCenters(const std::vector<cv::Point2f>& track_centers, 
                                                     float distance_threshold) const;
};

} // namespace tracking