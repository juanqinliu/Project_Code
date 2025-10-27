#include "roi/ROI.h"
#include <algorithm>
#include <cmath>
#include <cfloat> 

namespace tracking {

// ROI implementation
ROI::ROI(int roi_id, cv::Rect rect) 
    : id(roi_id), bbox(rect), no_detection_count(0), no_tracking_count(0), 
      last_updated(0), is_merged(false), safety_ratio(0.8f) {
}

cv::Point2f ROI::center() const {
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
}

int ROI::area() const {
    return bbox.width * bbox.height;
}

cv::Rect ROI::safetyBbox() const {

    int margin_w = static_cast<int>(bbox.width * (1 - safety_ratio) / 2);
    int margin_h = static_cast<int>(bbox.height * (1 - safety_ratio) / 2);
    

    int safe_x1 = bbox.x + margin_w;
    int safe_y1 = bbox.y + margin_h;
    int safe_x2 = bbox.x + bbox.width - margin_w;
    int safe_y2 = bbox.y + bbox.height - margin_h;

    return cv::Rect(safe_x1, safe_y1, safe_x2 - safe_x1, safe_y2 - safe_y1);
}

bool ROI::containsPoint(cv::Point2f point, float margin) const {
    return (bbox.x - margin <= point.x && point.x <= bbox.x + bbox.width + margin &&
            bbox.y - margin <= point.y && point.y <= bbox.y + bbox.height + margin);
}

bool ROI::isInSafetyZone(cv::Point2f point) const {

    cv::Rect safe_bbox = safetyBbox();

    return (safe_bbox.x <= point.x && point.x <= safe_bbox.x + safe_bbox.width &&
            safe_bbox.y <= point.y && point.y <= safe_bbox.y + safe_bbox.height);
}

std::vector<cv::Point2f> ROI::getSafetyZoneViolations(const std::vector<cv::Point2f>& points) const {
    std::vector<cv::Point2f> violations;
    for (const auto& point : points) {
        if (!isInSafetyZone(point)) {
            violations.push_back(point);
        }
    }
    return violations;
}

void ROI::updatePosition(int new_x, int new_y, int new_width, int new_height) {

    bbox.x = new_x;
    bbox.y = new_y;
    if (new_width != -1) bbox.width = new_width;
    if (new_height != -1) bbox.height = new_height;
}

std::optional<ROIMemory*> ROI::findCandidateForRecovery(cv::Point2f detection_center, const Config& config) {
    std::vector<std::pair<ROIMemory*, float>> candidates;
    
    for (auto& [track_id, memory] : track_memories) {
        if (memory->isReliable() && memory->lost_duration <= config.roi_memory_frames) {
            float distance = cv::norm(detection_center - memory->last_position);
            if (distance < config.roi_spatial_tolerance) {
                candidates.emplace_back(memory.get(), distance);
            }
        }
    }
    
    if (candidates.empty()) {
        return std::nullopt;
    }

    auto min_it = std::min_element(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    return min_it->first;
}

// ==================== ROI optimization algorithm ====================

float ROI::calculateOverlapRatio(const ROI& other) const {
    cv::Rect intersection = bbox & other.bbox;
    if (intersection.area() <= 0) {
        return 0.0f;
    }
    
    int union_area = bbox.area() + other.bbox.area() - intersection.area();
    if (union_area <= 0) {
        return 0.0f;
    }
    
    return static_cast<float>(intersection.area()) / union_area;
}

cv::Rect ROI::calculateMergedBbox(const ROI& other, int frame_width, int frame_height, int margin) const {

    int x1 = std::min(bbox.x, other.bbox.x);
    int y1 = std::min(bbox.y, other.bbox.y);
    int x2 = std::max(bbox.x + bbox.width, other.bbox.x + other.bbox.width);
    int y2 = std::max(bbox.y + bbox.height, other.bbox.y + other.bbox.height);
    

    if (margin <= 0) {
        int avg_size = (bbox.width + bbox.height + other.bbox.width + other.bbox.height) / 4;
        margin = static_cast<int>(avg_size * 0.05f);
    }
    

    x1 = x1 - margin;
    y1 = y1 - margin;
    x2 = x2 + margin;
    y2 = y2 + margin;
    

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(frame_width, x2);
    y2 = std::min(frame_height, y2);
    
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

cv::Rect ROI::validateBounds(int frame_width, int frame_height) const {
    int x = std::max(0, std::min(bbox.x, frame_width - bbox.width));
    int y = std::max(0, std::min(bbox.y, frame_height - bbox.height));
    int width = std::min(bbox.width, frame_width - x);
    int height = std::min(bbox.height, frame_height - y);
    return cv::Rect(x, y, width, height);
}

bool ROI::shouldMergeWith(const ROI& other) const {

    float overlap_ratio = calculateOverlapRatio(other);
    return overlap_ratio > 0.05f;  // 重合面积超过20%才合并
}

bool ROI::shouldSplit(const std::vector<cv::Point2f>& track_centers, int base_size) const {

    if (track_centers.size() < 2) {
        // LOG_INFO("    目标数量不足(" << track_centers.size() << " < 2)，不分割");
        return false;
    }
    
    float max_distance = 0.0f;
    std::pair<size_t, size_t> max_distance_indices = {0, 0};
    
    for (size_t i = 0; i < track_centers.size(); ++i) {
        for (size_t j = i + 1; j < track_centers.size(); ++j) {
            float distance = cv::norm(track_centers[i] - track_centers[j]);
            if (distance > max_distance) {
                max_distance = distance;
                max_distance_indices = {i, j};
            }
        }
    }
    

    if (max_distance > 500.0f) {
        return true;
    } else {

        return false;
    }
}

// ==================== ROI adaptive update ====================

bool ROI::adaptiveUpdate(const std::vector<cv::Point2f>& track_centers, 
                        int frame_width, int frame_height, bool force_update) {
    if (track_centers.empty()) {
        return false;
    }
    

    cv::Point2f targets_center = calculateTargetsCenter(track_centers);
    cv::Point2f current_center = center();
    float center_offset = cv::norm(targets_center - current_center);
    

    cv::Size required_size = calculateRequiredSize(track_centers);


    bool should_update = force_update || 
                        (center_offset > 5.0f) || 
                        (std::abs(bbox.width - required_size.width) > 20) ||
                        (std::abs(bbox.height - required_size.height) > 20);
    
    if (should_update) {

        const float alpha = 0.3f; 
        cv::Point2f smoothed_center = cv::Point2f(
            (1 - alpha) * current_center.x + alpha * targets_center.x,
            (1 - alpha) * current_center.y + alpha * targets_center.y
        );

        int desired_width = required_size.width;
        int desired_height = required_size.height;


        const float max_shrink_ratio = 0.10f; // 10%
        int min_width = static_cast<int>(bbox.width * (1.0f - max_shrink_ratio));
        int min_height = static_cast<int>(bbox.height * (1.0f - max_shrink_ratio));
        desired_width = std::max(desired_width, min_width);
        desired_height = std::max(desired_height, min_height);

        int side = std::max(desired_width, desired_height);

        int min_size = 300;
        int max_size = std::min(800, std::min(frame_width, frame_height));
        side = std::max(min_size, std::min(max_size, side));
        desired_width = desired_height = side;
        

        int new_x = static_cast<int>(smoothed_center.x - desired_width / 2);
        int new_y = static_cast<int>(smoothed_center.y - desired_height / 2);

        new_x = std::max(0, std::min(new_x, frame_width - desired_width));
        new_y = std::max(0, std::min(new_y, frame_height - desired_height));
        

        updatePosition(new_x, new_y, desired_width, desired_height);
        return true;
    }
    
    return false;
}


bool ROI::adaptiveUpdateWithTrackInfo(const std::vector<std::pair<cv::Point2f, cv::Size2f>>& track_info,
                                     int frame_width, int frame_height, const Config& config, bool force_update) {
    if (track_info.empty()) {
        return false;
    }
    

    std::vector<cv::Point2f> track_centers;
    for (const auto& info : track_info) {
        track_centers.push_back(info.first);
    }
    

    cv::Point2f targets_center = calculateTargetsCenter(track_centers);
    cv::Point2f current_center = center();
    float center_offset = cv::norm(targets_center - current_center);
    

    cv::Size required_size = calculateAdaptiveROISize(track_info, config);
    

    bool has_safety_violations = false;
    auto violations = getSafetyZoneViolations(track_centers);
    if (!violations.empty()) {
        has_safety_violations = true;

    }
    

    bool approaching_boundary = false;
    cv::Rect safety_bbox = safetyBbox();
    for (const auto& center : track_centers) {

        float dist_to_boundary = std::min({
            center.x - safety_bbox.x,  
            (safety_bbox.x + safety_bbox.width) - center.x,  
            center.y - safety_bbox.y,  
            (safety_bbox.y + safety_bbox.height) - center.y  
        });
        

        if (dist_to_boundary < 30.0f) {
            approaching_boundary = true;
            break;
        }
    }
    
    bool should_update = force_update || 
                        has_safety_violations ||           
                        approaching_boundary ||            
                        (center_offset > 3.0f) ||          
                        (std::abs(bbox.width - required_size.width) > 15) ||  
                        (std::abs(bbox.height - required_size.height) > 15);
    
    if (should_update) {

        const float alpha = static_cast<float>(config.roi_size_smooth_factor);
        

        float position_alpha = alpha;
        float size_alpha = alpha;
        if (has_safety_violations) {
            position_alpha = 0.7f;  
            size_alpha = 0.6f;      
        } else if (approaching_boundary) {
            position_alpha = 0.5f;  
            size_alpha = 0.4f;
        }
        
        cv::Point2f smoothed_center = cv::Point2f(
            (1 - position_alpha) * current_center.x + position_alpha * targets_center.x,
            (1 - position_alpha) * current_center.y + position_alpha * targets_center.y
        );

        int desired_width = required_size.width;
        int desired_height = required_size.height;


        if (config.enable_adaptive_roi_size) {
            desired_width = static_cast<int>((1 - size_alpha) * bbox.width + size_alpha * required_size.width);
            desired_height = static_cast<int>((1 - size_alpha) * bbox.height + size_alpha * required_size.height);
        }


        float max_shrink_ratio = has_safety_violations ? 0.05f : 0.15f; // 安全违规时只允许5%收缩
        int min_width = static_cast<int>(bbox.width * (1.0f - max_shrink_ratio));
        int min_height = static_cast<int>(bbox.height * (1.0f - max_shrink_ratio));
        desired_width = std::max(desired_width, min_width);
        desired_height = std::max(desired_height, min_height);


        int side = std::max(desired_width, desired_height);
        

        int min_size = config.roi_min_size;
        int max_size = std::min(config.roi_max_size, std::min(frame_width, frame_height));
        side = std::max(min_size, std::min(max_size, side));
        desired_width = desired_height = side;

        int new_x = static_cast<int>(smoothed_center.x - desired_width / 2);
        int new_y = static_cast<int>(smoothed_center.y - desired_height / 2);
        

        new_x = std::max(0, std::min(new_x, frame_width - desired_width));
        new_y = std::max(0, std::min(new_y, frame_height - desired_height));
        

        updatePosition(new_x, new_y, desired_width, desired_height);
        return true;
    }
    
    return false;
}


cv::Point2f ROI::calculateTargetsCenter(const std::vector<cv::Point2f>& track_centers) const {
    float sum_x = 0, sum_y = 0;
    for (const auto& pt : track_centers) {
        sum_x += pt.x;
        sum_y += pt.y;
    }
    return cv::Point2f(sum_x / track_centers.size(), sum_y / track_centers.size());
}


cv::Size ROI::calculateRequiredSize(const std::vector<cv::Point2f>& track_centers) const {
    if (track_centers.size() == 1) {
        
        return calculateSingleTargetSize(track_centers[0]);
    } else {
        
        return calculateMultiTargetSize(track_centers);
    }
}


cv::Size ROI::calculateSingleTargetSize(const cv::Point2f& target_center) const {

    int base_size = 300; 
    

    int required_size = static_cast<int>(base_size / safety_ratio);
    
    return cv::Size(required_size, required_size);
}

cv::Size ROI::calculateAdaptiveROISize(const std::vector<std::pair<cv::Point2f, cv::Size2f>>& track_info, 
                                      const Config& config) const {
    if (track_info.empty()) {
        return cv::Size(config.roi_min_size, config.roi_min_size);
    }
    
    if (!config.enable_adaptive_roi_size) {

        std::vector<cv::Point2f> centers;
        for (const auto& info : track_info) {
            centers.push_back(info.first);
        }
        return (track_info.size() == 1) ? calculateSingleTargetSize(centers[0]) : calculateMultiTargetSize(centers);
    }
    

    if (track_info.size() == 1) {

        const auto& [center, target_size] = track_info[0];
        

        float max_target_dimension = std::max(target_size.width, target_size.height);
        

        int adaptive_roi_size = static_cast<int>(max_target_dimension * config.roi_target_scale_factor);
        

        adaptive_roi_size = std::max(config.roi_min_size, 
                                   std::min(config.roi_max_size, adaptive_roi_size));
        
        return cv::Size(adaptive_roi_size, adaptive_roi_size);
    } else {

        float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
        float max_target_dimension = 0.0f;
        

        for (const auto& [center, target_size] : track_info) {
            min_x = std::min(min_x, center.x);
            min_y = std::min(min_y, center.y);
            max_x = std::max(max_x, center.x);
            max_y = std::max(max_y, center.y);
            
            float target_dimension = std::max(target_size.width, target_size.height);
            max_target_dimension = std::max(max_target_dimension, target_dimension);
        }

        float targets_span_x = max_x - min_x;
        float targets_span_y = max_y - min_y;
        float targets_span = std::max(targets_span_x, targets_span_y);
        

        float span_based_size = targets_span + max_target_dimension * config.roi_target_scale_factor;
        float target_based_size = max_target_dimension * config.roi_target_scale_factor;
        
        int adaptive_roi_size = static_cast<int>(std::max(span_based_size, target_based_size));

        adaptive_roi_size = std::max(config.roi_min_size, 
                                   std::min(config.roi_max_size, adaptive_roi_size));
        
        return cv::Size(adaptive_roi_size, adaptive_roi_size);
    }
}


cv::Size ROI::calculateMultiTargetSize(const std::vector<cv::Point2f>& track_centers) const {

    float min_x = track_centers[0].x, max_x = track_centers[0].x;
    float min_y = track_centers[0].y, max_y = track_centers[0].y;
    
    for (const auto& pt : track_centers) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }
    

    float targets_width = max_x - min_x;
    float targets_height = max_y - min_y;
    

    float span = std::max(targets_width, targets_height);
    float expansion = 1.5f;  
    int side = static_cast<int>(span / safety_ratio * expansion);
    
    int min_size = 300;
    int max_size = 800;
    side = std::max(min_size, std::min(max_size, side));
    
    return cv::Size(side, side);
}

// ==================== ROI split algorithm ====================

std::vector<cv::Rect> ROI::generateSplitConfigs(const std::vector<cv::Point2f>& track_centers,
                                                int frame_width, int frame_height, int base_size) const {

    
    if (track_centers.size() <= 1) {

        return {};
    }
    

    if (base_size <= 0) {
        base_size = 300;
    }

    std::vector<std::vector<int>> clusters = clusterTrackCenters(track_centers, 500.0f);
    

    for (size_t i = 0; i < clusters.size(); ++i) {

        for (int idx : clusters[i]) {
            // LOG_INFO("        Target " << idx << ": (" << track_centers[idx].x << ", " << track_centers[idx].y << ")");
        }
    }
    

    if (clusters.size() < 2) {

        return {};
    }
    
    // 步骤2: 为每个聚类创建ROI
    std::vector<cv::Rect> candidate_rois;
    for (size_t cluster_idx = 0; cluster_idx < clusters.size(); ++cluster_idx) {
        const auto& cluster = clusters[cluster_idx];
        

        
        float sum_x = 0, sum_y = 0;
        float min_x = FLT_MAX, min_y = FLT_MAX, max_x = -FLT_MAX, max_y = -FLT_MAX;
        
        for (int idx : cluster) {
            const cv::Point2f& pt = track_centers[idx];
            sum_x += pt.x;
            sum_y += pt.y;
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }
        

        float center_x = sum_x / cluster.size();
        float center_y = sum_y / cluster.size();
        

        float cluster_width = max_x - min_x;
        float cluster_height = max_y - min_y;
        
        int roi_width = std::max(static_cast<int>(cluster_width * 1.5f), base_size);
        int roi_height = std::max(static_cast<int>(cluster_height * 1.5f), base_size);
        
        int roi_size = std::max(roi_width, roi_height);
        
        int roi_x = static_cast<int>(center_x - roi_size / 2);
        int roi_y = static_cast<int>(center_y - roi_size / 2);
        
        roi_x = std::max(0, std::min(roi_x, frame_width - roi_size));
        roi_y = std::max(0, std::min(roi_y, frame_height - roi_size));
        

        candidate_rois.emplace_back(roi_x, roi_y, roi_size, roi_size);
        

    }
    

    bool has_overlap = true;
    int max_iterations = 5; 
    int iteration = 0;
    
    while (has_overlap && iteration < max_iterations) {
        has_overlap = false;
        
        for (size_t i = 0; i < candidate_rois.size(); ++i) {
            for (size_t j = i + 1; j < candidate_rois.size(); ++j) {
                cv::Rect intersection = candidate_rois[i] & candidate_rois[j];
                if (intersection.area() > 0) {
                   
                    has_overlap = true;

                    int new_size_i = static_cast<int>(candidate_rois[i].width * 0.9f);
                    int new_size_j = static_cast<int>(candidate_rois[j].width * 0.9f);
                    

                    new_size_i = std::max(new_size_i, base_size);
                    new_size_j = std::max(new_size_j, base_size);

                    int center_x_i = candidate_rois[i].x + candidate_rois[i].width / 2;
                    int center_y_i = candidate_rois[i].y + candidate_rois[i].height / 2;
                    int center_x_j = candidate_rois[j].x + candidate_rois[j].width / 2;
                    int center_y_j = candidate_rois[j].y + candidate_rois[j].height / 2;
                    
                    candidate_rois[i] = cv::Rect(center_x_i - new_size_i/2, center_y_i - new_size_i/2, new_size_i, new_size_i);
                    candidate_rois[j] = cv::Rect(center_x_j - new_size_j/2, center_y_j - new_size_j/2, new_size_j, new_size_j);
                    
                    candidate_rois[i].x = std::max(0, std::min(candidate_rois[i].x, frame_width - new_size_i));
                    candidate_rois[i].y = std::max(0, std::min(candidate_rois[i].y, frame_height - new_size_i));
                    candidate_rois[j].x = std::max(0, std::min(candidate_rois[j].x, frame_width - new_size_j));
                    candidate_rois[j].y = std::max(0, std::min(candidate_rois[j].y, frame_height - new_size_j));
                    
    
                }
            }
        }
        iteration++;
    }
    
    if (has_overlap) {
        return {};
    }
    

    return candidate_rois;
}


std::vector<std::vector<int>> ROI::clusterTrackCenters(
        const std::vector<cv::Point2f>& track_centers, float distance_threshold) const {

    
    std::vector<std::vector<int>> clusters;
    std::vector<bool> assigned(track_centers.size(), false);
    
    for (size_t i = 0; i < track_centers.size(); ++i) {
        if (assigned[i]) continue;

        std::vector<int> new_cluster = {static_cast<int>(i)};
        assigned[i] = true;

        bool added = true;
        while (added) {
            added = false;
            for (size_t j = 0; j < track_centers.size(); ++j) {
                if (assigned[j]) continue;

                bool can_add = true;
                for (int idx : new_cluster) {
                    float dist = cv::norm(track_centers[idx] - track_centers[j]);
                    if (dist > distance_threshold) {
                        can_add = false;
                        break;
                    }
                }
                
                if (can_add) {
                    new_cluster.push_back(static_cast<int>(j));
                    assigned[j] = true;
                    added = true;
                    
                }
            }
        }
        
        clusters.push_back(new_cluster);
    }

    return clusters;
}

} // namespace tracking 