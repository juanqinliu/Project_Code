#pragma once

#include <unordered_map>
#include <memory>
#include <vector>
#include <deque>
#include <set>
#include <optional>
#include <functional>
#include "common/Config.h"
#include "roi/ROI.h"
#include "CandidateTarget.h"
// Forward declaration to avoid circular dependency
namespace tracking {
    class STrack;
}
#include "common/Detection.h"

namespace tracking {

class ROIManager {
public:
    explicit ROIManager(const Config& config);
    
    // ROI lifecycle management
    int createROIForTrack(const STrack& track, int frame_width, int frame_height, int frame_id);
    void addTrackMemory(int roi_id, const STrack& track, int frame_id);
    void updateROIPositions(std::vector<std::unique_ptr<STrack>>& tracks, 
                           int frame_width, int frame_height);
    int cleanupInactiveROIs(int current_frame);
    
    // ROI management strategies
    int mergeOverlappingROIs(int frame_width, int frame_height);
    int splitOversizedROIs(std::vector<std::unique_ptr<STrack>>& tracks, 
                          int frame_width, int frame_height, int frame_id);
    bool executeROISplit(int roi_id, const std::vector<cv::Point2f>& track_centers, 
                        const std::vector<int>& track_ids, int frame_width, int frame_height, int frame_id,
                        std::vector<std::unique_ptr<STrack>>& tracks);
    
    // Candidate targets
    void updateCandidates(const std::vector<Detection>& detections_outside_roi, int frame_id);
    int createROIsForConfirmedCandidates(int frame_width, int frame_height, int frame_id);
    
    // Status updates
    void updateROIDetectionStatus(const std::unordered_map<int, std::vector<Detection>>& roi_detections);
    void updateROITrackingStatus(std::vector<std::unique_ptr<STrack>>& tracks, int frame_id);
    void updateROITrackMemories(std::vector<std::unique_ptr<STrack>>& tracks, int frame_id);
    
    // Main management entry points
    std::unordered_map<std::string, int> performROIManagement(
        std::vector<std::unique_ptr<STrack>>& tracks, 
        int frame_width, int frame_height, int frame_id);
    std::unordered_map<std::string, int> dynamicROIManagement(
        std::vector<std::unique_ptr<STrack>>& tracks, 
        int frame_width, int frame_height, int frame_id);
    std::unordered_map<std::string, int> localPhaseROIManagement(
        std::vector<std::unique_ptr<STrack>>& tracks, 
        int frame_width, int frame_height, int frame_id);

    // Safety utilities
    int expandROIsForSafety(std::vector<std::unique_ptr<STrack>>& tracks, 
                           int frame_width, int frame_height);

    // Queries
    std::optional<ROIMemory*> findCandidateForRecovery(cv::Point2f detection_center, int roi_id);
    std::optional<int> findROIIdByPoint(cv::Point2f point, int margin = 0) const;
    std::optional<int> findBestROIForTrack(cv::Point2f track_center);
    
    // Accessors
    const std::unordered_map<int, std::unique_ptr<ROI>>& getROIs() const { return rois_; }
    const std::unordered_map<int, std::unique_ptr<CandidateTarget>>& getCandidates() const { return candidates_; }
    ROI* getROI(int roi_id);
    const ROI* getROI(int roi_id) const;
    void setCurrentFrameId(int frame_id) { current_frame_id_ = frame_id; }
    
    // Iteration helper
    void forEachROI(std::function<void(int, ROI&)> func);

    // Helpers
    std::vector<STrack*> getTracksInROI(std::vector<std::unique_ptr<STrack>>& tracks, 
                                               int roi_id, const ROI& roi);
    std::vector<STrack*> findTracksByPosition(const std::vector<std::unique_ptr<STrack>>& tracks, 
                                                     const ROI& roi);
    void fixTrackROIAssociations(std::vector<std::unique_ptr<STrack>>& tracks);

    // Internals
private:
    void mergeROIs(int roi_id1, int roi_id2, int frame_width, int frame_height);
    int allocateROIId();
    void recycleROIId(int roi_id);

    // Members
    Config config_;
    std::unordered_map<int, std::unique_ptr<ROI>> rois_;
    std::unordered_map<int, std::unique_ptr<CandidateTarget>> candidates_;
    std::deque<int> available_roi_ids_;
    int next_roi_id_;
    int current_frame_id_;
};

} // namespace tracking