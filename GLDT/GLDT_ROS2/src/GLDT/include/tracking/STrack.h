#pragma once

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <deque>
#include <memory>
#include "roi/ROIManager.h"
#include "common/Detection.h"
#include "common/Logger.h"

namespace tracking {

class STrack {
public:
    // Tracking state
    enum TrackState { New = 0, Tracked, Lost, Removed };
    
    // Constructor
    STrack(cv::Rect2f tlwh_val, float score_val, int cls = 0);
    
    // Bounding box and meta
    cv::Rect2f tlwh;
    float score;
    int class_id;
    int roi_id;
    
    // State
    bool is_activated;
    bool is_real_target;
    int frame_id;
    int tracklet_len;
    int start_frame;
    TrackState state;
    int lost_frames_count;  // lost frame counter
    int miss_count_in_grace;  // miss count in grace period (before真正Lost)
    
    // Additional attributes
    float quality_score;
    int roi_miss_count;
    float spatial_confidence;
    bool is_recovered;
    float recovery_confidence;
    
    // Confirmation mechanism
    bool is_confirmed;
    int confirmation_frames;
    int min_confirmation_frames;
    
    // Position history
    std::deque<cv::Point2f> position_history;
    std::deque<float> confidence_history;
    
    // Motion history
    std::deque<cv::Point2f> velocity_history;     // velocities
    std::deque<float> direction_history;          // movement directions
    std::deque<float> speed_history;              // movement speeds
    std::deque<int> frame_history;                // frame ids for time weighting
    
    
    // ID management
    int track_id;     // base-compatible id
    int temp_id;      // temporary id
    int permanent_id; // persistent id
    
    // Static id counters
    static int next_permanent_id_;
    static int temp_id_counter_;
    
    // Public API
    void activate(int frame_id);
    void predict();
    void update(const cv::Rect2f& new_tlwh, float new_score, int new_frame_id);
    void markLost();
    bool isLost() const;
    bool isActivated() const;
    int getTimeSinceUpdate() const;
    cv::Point2f center() const;
    cv::Point2f getPredictedCenter() const;
    bool isStable() const;
    int displayId() const;
    void assignTempId();
    void confirmAsRealTarget();
    void updateConfirmationStatus();
    void recoverWithMemory(const ROIMemory& memory, const Detection& detection, int frame_id);
    
    // Deletion mark
    void markForDeletion() { marked_for_deletion_ = true; }
    bool isMarkedForDeletion() const { return marked_for_deletion_; }
    
    // Appearance features
    cv::Mat getAppearance() const { return appearance_; }
    void setAppearance(const cv::Mat& appearance) { appearance_ = appearance.clone(); }
    bool hasAppearance() const { return !appearance_.empty(); }
    
    // Optical flow features
    void updateOpticalFlowFeatures(const cv::Mat& current_gray_patch, 
                                   const std::vector<cv::Point2f>& flow_points);
    
    // Optical flow accessors
    const cv::Mat& getPrevGrayPatch() const { return prev_gray_patch_; }
    const std::vector<cv::Point2f>& getOpticalFlowPoints() const { return optical_flow_points_; }
    bool hasOpticalFlowFeatures() const { return !prev_gray_patch_.empty() && !optical_flow_points_.empty(); }
    
    // Predict n future centers
    std::vector<cv::Point2f> predictFutureCenters(int n) const;
    
    // Kalman filter utilities
    static int getNextPermanentId();
    static int getNextTempId();
    static void resetIdCounters();  // reset id counters

    // DTW-based trajectory matching
    std::vector<cv::Point2f> getRecentTrajectory(int num_frames = 5) const;
    float calculateDTWDistance(const std::vector<cv::Point2f>& trajectory1, 
                              const std::vector<cv::Point2f>& trajectory2) const;
    float calculateTrajectorySimilarity(const STrack* other, int num_frames = 5) const;

private:
    // Kalman filter state
    bool kf_initialized_;
    cv::Mat kf_mean_;                    // state mean [x,y,w,h,vx,vy,vw,vh]
    cv::Mat kf_covariance_;              // state covariance
    cv::Mat kf_motion_mat_;              // transition matrix
    cv::Mat kf_update_mat_;              // observation matrix
    float kf_std_weight_position_;       // std weight for position
    float kf_std_weight_velocity_;       // std weight for velocity
    
    // Appearance features
    cv::Mat appearance_;
    
    // Optical flow features
    cv::Mat prev_gray_patch_;                    // previous gray patch
    std::vector<cv::Point2f> optical_flow_points_; // tracked points
    
    // Deletion mark
    bool marked_for_deletion_ = false;
    
    // Kalman filter helpers
    void initKalmanFilter();                             // initialize filter
    void initiateKalman(const cv::Rect2f& measurement);  // initialize from measurement
    std::pair<cv::Mat, cv::Mat> predictKalman();         // predict step
    std::pair<cv::Mat, cv::Mat> predictKalman(const cv::Mat& mean, const cv::Mat& covariance) const; // overload
    std::pair<cv::Mat, cv::Mat> projectKalman(const cv::Mat& mean, const cv::Mat& covariance); // project to meas.
    std::pair<cv::Mat, cv::Mat> updateKalman(const cv::Mat& mean, const cv::Mat& covariance, 
                                             const cv::Rect2f& measurement); // update step
};


// GMM modeling (motion + spatial + history + ROI)
cv::Ptr<cv::ml::EM> fitGMM(
    const std::deque<cv::Point2f>& positions,
    const std::deque<cv::Point2f>& velocities,
    const std::deque<float>& directions,
    const std::deque<float>& speeds,
    const std::deque<int>& frame_ids,
    const cv::Rect& roi_bbox,
    int current_frame,
    int n_components = 2
);

float GMMProbability(
    const cv::Ptr<cv::ml::EM>& gmm,
    const cv::Point2f& position,
    const cv::Point2f& velocity,
    float direction,
    float speed,
    const cv::Rect& roi_bbox
);

} // namespace tracking 