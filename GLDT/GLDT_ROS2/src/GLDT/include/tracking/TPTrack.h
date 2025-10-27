#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <opencv2/core.hpp>
#include "common/Config.h"
#include "roi/ROIManager.h"
#include "tracking/STrack.h"
#include "common/Detection.h"
#include "common/Logger.h"

namespace tracking {

// Constants
constexpr int HISTORY_FRAMES = 30;  // history length in frames

// Cost matrix type
enum CostMatrixType {
    PRIMARY_MATCH,      // high-confidence detections with active tracks
    RESCUE_MATCH,       // low-confidence detections with unmatched tracks
    UNCONFIRMED_MATCH   // high-confidence detections with unconfirmed tracks
};

// ===== 🔥 New: Geometric Constraints System =====

// Matching constraints configuration
struct MatchingConstraints {
    float max_distance_to_last = 100.0f;     // Hard limit: max distance to last position
    float gating_distance_factor = 3.0f;     // Adaptive gating factor based on object size
    float base_gating_distance = 50.0f;      // Minimum gating distance
    bool strict_mode = false;                // Strict mode for lost tracks
    
    // Get adaptive gating distance
    float getGatingDistance(float avg_size) const {
        float adaptive = std::max(base_gating_distance, avg_size * gating_distance_factor);
        return strict_mode ? adaptive * 0.67f : adaptive;
    }
};

// Geometric constraints manager
class GeometricConstraints {
public:
    // Check if track-detection pair satisfies geometric constraints
    static bool checkConstraints(
        const STrack* track,
        const STrack* detection,
        const MatchingConstraints& constraints);
    
    // Apply geometric gating to cost matrix (reject invalid matches)
    static void applyGeometricGating(
        cv::Mat& cost_matrix,
        const std::vector<STrack*>& tracks,
        const std::vector<STrack*>& detections,
        const MatchingConstraints& constraints,
        bool verbose = false);
    
    // Get standard constraints for tracked tracks
    static MatchingConstraints getStandardConstraints();
    
    // Get strict constraints for lost tracks
    static MatchingConstraints getStrictConstraints();
};

// GMM-based memory recovery features
struct TrajectoryFeatures {
    std::deque<cv::Point2f> position_history;  // positions
    std::vector<cv::Point2f> velocity_history;  // velocities
    cv::Mat appearance_histogram;               // appearance histogram
    std::vector<float> hog_features;            // HOG features
    // ORB features
    std::vector<cv::KeyPoint> orb_keypoints;    // ORB keypoints
    cv::Mat orb_descriptors;                    // ORB descriptors
    // Optical flow features
    std::vector<cv::Point2f> optical_flow_points; // flow points
    std::vector<cv::Point2f> optical_flow_vectors; // flow vectors
    cv::Mat prev_gray_patch;                    // previous gray patch
    std::vector<cv::Point2f> relative_positions; // relative positions to others
    float avg_confidence;                       // average detection confidence
    int track_length;                           // track length
    float bbox_area;                            // bbox area
};

// Three-stage recovery model
struct ThreeStageRecoveryModel {
    // Stage 1: motion prediction
    struct MotionPrediction {
        cv::Point2f predicted_position;    // predicted center
        cv::Point2f avg_velocity;          // average velocity
        cv::Point2f avg_acceleration;      // average acceleration
        float prediction_radius;            // search radius
        int lost_frames;                   // lost frames
    };
    
    // Stage 2: appearance model
    struct AppearanceModel {
        cv::Mat color_histogram;           // color histogram
        std::vector<float> hog_features;   // HOG features
        // ORB features
        std::vector<cv::KeyPoint> orb_keypoints;    // ORB keypoints
        cv::Mat orb_descriptors;                    // ORB descriptors
        // Optical flow features
        std::vector<cv::Point2f> optical_flow_points; // flow points
        std::vector<cv::Point2f> optical_flow_vectors; // flow vectors
        cv::Mat prev_gray_patch;                    // previous gray patch
        float color_weight;                // color weight
        float hog_weight;                  // HOG weight
        // ORB and optical flow weights
        float orb_weight;                  // ORB weight
        float optical_flow_weight;         // optical flow weight
        float bbox_area;                   // bbox area
    };
    
    // Stage 3: interaction model
    struct InteractionModel {
        std::vector<cv::Point2f> relative_positions;  // relative positions
        std::vector<float> interaction_patterns;       // interaction patterns
        float interaction_consistency;                 // consistency
    };
    
    MotionPrediction motion;
    AppearanceModel appearance;
    InteractionModel interaction;
    float final_similarity;                // final similarity score
};

class TPTrack {
public:
    TPTrack(const Config& config, int frame_rate,
                       std::shared_ptr<ROIManager> roi_manager = nullptr);
    
    // Update with detections
    std::vector<std::unique_ptr<STrack>> update(
        const std::vector<cv::Rect2f>& bboxes, 
        const std::vector<float>& scores, 
        const std::vector<int>& classes);
    
    std::vector<std::unique_ptr<STrack>> update(
        const std::vector<cv::Rect2f>& bboxes, 
        const std::vector<float>& scores, 
        const std::vector<int>& classes, 
        const cv::Mat& frame);
    
    std::vector<std::unique_ptr<STrack>> update(
        const std::vector<Detection>& detections,
        const std::unordered_map<int, std::vector<Detection>>& roi_detections = {});

    // Active tracks
    const std::vector<std::unique_ptr<STrack>>& getActiveTracks() const { 
        return tracked_stracks_; 
    }
    
    // Non-const access
    std::vector<std::unique_ptr<STrack>>& getActiveTracks() { 
        return tracked_stracks_; 
    }
    
    // Lost tracks
    const std::vector<std::unique_ptr<STrack>>& getLostTracks() const { 
        return lost_stracks_; 
    }
    
    // Toggle original ByteTrack
    void setUseOriginalByteTrack(bool value) { 
        use_original_bytetrack_ = value;
    }
    
    // Reset tracker state
    void resetTrackerState();
    
    // Save/restore state
    void saveTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks);
    void restoreTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks);
    
    // Transfer state when switching modes
    void transferState(TPTrack& other_tracker);
    
    // Current frame id
    int getFrameId() const { return frame_id_; }
    
protected:
    // Initialize tracks
    std::vector<std::unique_ptr<STrack>> initTrack(const std::vector<Detection>& detections);
    
    // Heatmap-based recovery
    std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>> 
    processHeatmapBasedRecovery(std::vector<std::unique_ptr<STrack>>& detections,
                               const std::unordered_map<int, std::vector<Detection>>& roi_detections);
    
    // GMM memory recovery
    std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>>
    processGMMMemoryRecovery(std::vector<std::unique_ptr<STrack>>& detections,
                               const std::unordered_map<int, std::vector<Detection>>& roi_detections,
                               const std::vector<std::unique_ptr<STrack>>& current_frame_lost_stracks);
    
    std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>>
    processGMMMemoryRecoveryWithFailedTracks(std::vector<std::unique_ptr<STrack>>& detections,
                                             const std::vector<STrack*>& failed_tracks,
                            const std::unordered_map<int, std::vector<Detection>>& roi_detections);
    
    // Feature extraction
    TrajectoryFeatures extractTrajectoryFeatures(const STrack& track, 
                                               const std::vector<STrack*>& all_tracks);
    
    // Interaction features
    std::vector<cv::Point2f> calculateInteractionFeatures(const STrack& track,
                                                         const std::vector<STrack*>& all_tracks);
    
    // Three-stage recovery steps
    ThreeStageRecoveryModel buildThreeStageRecoveryModel(const TrajectoryFeatures& lost_features,
                                  const TrajectoryFeatures& detection_features);
    
    // Stage 1: motion filter
    bool stageOneMotionPrediction(const ThreeStageRecoveryModel& model,
                                 const cv::Point2f& detection_position);
    
    // Stage 2: appearance matching
    float stageTwoAppearanceMatching(const ThreeStageRecoveryModel& model,
                                const TrajectoryFeatures& detection_features);
    
    // Stage 3: interaction verification
    float stageThreeInteractionVerification(const ThreeStageRecoveryModel& model,
                                          const TrajectoryFeatures& detection_features,
                                          const std::vector<STrack*>& all_tracks);
    
    // Final similarity
    float calculateThreeStageSimilarity(const ThreeStageRecoveryModel& model,
                                       const TrajectoryFeatures& lost_features,
                                       const TrajectoryFeatures& detection_features,
                                       const std::vector<STrack*>& all_tracks);
    
    // Appearance similarity
    float calculateAppearanceSimilarity(const cv::Mat& hist1, const cv::Mat& hist2);
    
    // Motion similarity
    float calculateMotionSimilarity(const cv::Point2f& predicted, const cv::Point2f& observed, float radius);
    float calculateSpatialSimilarity(const cv::Point2f& pos1, const cv::Point2f& pos2);
    float calculateInteractionSimilarity(const std::vector<float>& p1, const std::vector<float>& p2);
    float calculateMotionSimilarity(const std::deque<cv::Point2f>& pos1, const std::deque<cv::Point2f>& pos2);
    
    // Calculate interaction similarity
    float calculateInteractionSimilarity(const std::vector<cv::Point2f>& rel1,
                                      const std::vector<cv::Point2f>& rel2);

    // Matching method
    void associateDetectionsToTracks(const std::vector<STrack*>& tracks,
                                    const std::vector<STrack*>& detections,
                                    std::vector<std::pair<int, int>>& matches,
                                    std::vector<int>& unmatched_tracks,
                                    std::vector<int>& unmatched_dets,
                                    float thresh = 0.5f);

    // Calculate IoU matrix
    cv::Mat calculateIoUMatrix(const std::vector<STrack*>& tracks,
                              const std::vector<STrack*>& detections);
    
    // Calculate IoU
    float calculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2);
    
    // Hungarian algorithm allocation
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    hungarianAssignment(const cv::Mat& cost_matrix, float thresh);
    
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    linearAssignment(const cv::Mat& cost_matrix, float thresh);
    
    void multiPredict(const std::vector<STrack*>& stracks);

    // Track collection operation
    std::vector<std::unique_ptr<STrack>> jointSTracks(
        const std::vector<std::unique_ptr<STrack>>& tlista,
        const std::vector<std::unique_ptr<STrack>>& tlistb);
    
    std::vector<std::unique_ptr<STrack>> subSTracks(
        const std::vector<std::unique_ptr<STrack>>& tlista,
        const std::vector<std::unique_ptr<STrack>>& tlistb);
    
    std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>>
    removeDuplicateSTracks(std::vector<std::unique_ptr<STrack>>& stracksa,
                          std::vector<std::unique_ptr<STrack>>& stracksb);
    
    // Original ByteTrack cost matrix
    cv::Mat buildOriginalByteTrackCostMatrix(
        const std::vector<STrack*>& tracks, 
        const std::vector<STrack*>& detections, 
        CostMatrixType matrix_type);
        
    // New: build enhanced ROI constraint cost matrix
    cv::Mat buildEnhancedCostMatrix(
        const std::vector<STrack*>& tracks, 
        const std::vector<STrack*>& detections, 
        CostMatrixType matrix_type);

    // New: calculate motion feature cost - solve drone sudden change problem
    float calculateMotionCost(const STrack* track, const STrack* detection);
    
    // New: calculate appearance feature cost - color histogram + HOG feature
    float calculateAppearanceCost(const STrack* track, const STrack* detection);
    
    // New: calculate relative position feature cost - relative position relationship between multiple targets
    float calculateRelativePositionCost(const STrack* track, 
                                      const STrack* detection,
                                      const std::vector<STrack*>& all_tracks);
    
    // Calculate template matching score
    float calculateTemplateMatchScore(const cv::Mat& detection_appearance, const cv::Mat& template_appearance);
    
private:
    int frame_id_;
    
    // Track container
    std::vector<std::unique_ptr<STrack>> tracked_stracks_;
    std::vector<std::unique_ptr<STrack>> lost_stracks_;
    std::vector<std::unique_ptr<STrack>> removed_stracks_;
    
    // Configuration
    Config config_;
    
    // ROI manager
    std::shared_ptr<ROIManager> roi_manager_;
    
    // Algorithm selection
    bool use_original_bytetrack_ = false;
    
    // New: three-stage memory recovery related parameters
    static const int HISTORY_FRAMES = 30;  // History frames
    static constexpr float MOTION_PREDICTION_THRESHOLD = 0.8f;  // Motion prediction threshold
    static constexpr float APPEARANCE_SIMILARITY_THRESHOLD = 0.3f; // Appearance similarity threshold
    static constexpr float INTERACTION_CONSISTENCY_THRESHOLD = 0.6f; // Interaction consistency threshold
    static constexpr float FINAL_RECOVERY_THRESHOLD = 0.65f; // Final recovery threshold
};

} // namespace tracking 