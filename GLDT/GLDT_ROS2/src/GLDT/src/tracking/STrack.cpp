#include "tracking/STrack.h"
#include <algorithm>
#include <opencv2/ml.hpp>

namespace tracking {

int STrack::next_permanent_id_ = 1;
int STrack::temp_id_counter_ = 1;

// STrack Implementation
STrack::STrack(cv::Rect2f tlwh_val, float score_val, int cls) 
    : tlwh(tlwh_val), score(score_val), class_id(cls), roi_id(-1),
      is_activated(false), is_real_target(false), frame_id(0), tracklet_len(0), 
      start_frame(0), state(New), lost_frames_count(0),
      quality_score(score_val), roi_miss_count(0), spatial_confidence(1.0f),
      is_recovered(false), recovery_confidence(0.0f),
      is_confirmed(false), confirmation_frames(0), min_confirmation_frames(3),
      track_id(-1), temp_id(-1), permanent_id(-1),
      kf_initialized_(false), kf_std_weight_position_(0.1f), kf_std_weight_velocity_(0.1f) {
    
    // Init Kalman Filter
    initKalmanFilter();
}

int STrack::getNextPermanentId() {
    return next_permanent_id_++;
}

int STrack::getNextTempId() {
    return temp_id_counter_--;
}

int STrack::displayId() const {
    if (permanent_id != -1) {
        return permanent_id;
    } else if (temp_id != -1) {
        return temp_id;
    } else {
        return -1;  
    }
}

void STrack::assignTempId() {
    if (temp_id == -1) {
        temp_id = getNextTempId();
        track_id = temp_id;
    }
}

void STrack::confirmAsRealTarget() {
    // Confirm as real target, assign permanent ID
    if (!is_confirmed) {
        permanent_id = getNextPermanentId();
        is_confirmed = true;
        is_real_target = true;
        track_id = permanent_id;  
    }
}

void STrack::updateConfirmationStatus() {
    // Update confirmation status
    if (!is_confirmed && state == Tracked) {
        confirmation_frames++;
        if (confirmation_frames >= min_confirmation_frames) {
            confirmAsRealTarget();
        }
    }
}

void STrack::activate(int frame_id) {
    // Assign temporary ID and activate
    assignTempId();  
    
    this->frame_id = frame_id;
    this->start_frame = frame_id;
    this->is_activated = true;
    this->state = Tracked;
    this->tracklet_len = 1;
    
    // init position and confidence history
    position_history.clear();
    confidence_history.clear();
    cv::Point2f center_pt = center();
    position_history.push_back(center_pt);
    confidence_history.push_back(score);
    
    // If Kalman filter is not initialized, initialize it
    if (!kf_initialized_) {
        initiateKalman(tlwh);
    }
}

cv::Point2f STrack::center() const {
    // If Kalman filter is initialized, use the filter's state
    if (kf_initialized_ && !kf_mean_.empty()) {
        return cv::Point2f(kf_mean_.at<float>(0), kf_mean_.at<float>(1));
    }
    
    // Otherwise, use the center of the bounding box
    return cv::Point2f(tlwh.x + tlwh.width / 2.0f, tlwh.y + tlwh.height / 2.0f);
}

cv::Point2f STrack::getPredictedCenter() const {
    // If Kalman filter is initialized, use the filter's predicted state
    if (kf_initialized_ && !kf_mean_.empty()) {
        // Return predicted position plus velocity
        return cv::Point2f(kf_mean_.at<float>(0) + kf_mean_.at<float>(4), 
                           kf_mean_.at<float>(1) + kf_mean_.at<float>(5));
    }
    
    // If Kalman filter is not initialized, use simple linear prediction
    if (position_history.size() >= 2) {
        cv::Point2f last_pos = position_history.back();
        cv::Point2f prev_pos = position_history[position_history.size() - 2];
        cv::Point2f velocity = last_pos - prev_pos;
        return last_pos + velocity;
    }
    return center();
}

bool STrack::isStable() const {
    // Check if tracking is stable
    return (tracklet_len > 5 && 
            quality_score > 0.6f &&
            frame_id - start_frame > 10);
}

void STrack::predict() {
    if (kf_initialized_) {
        // Use Kalman filter to predict
        auto [mean, covariance] = predictKalman();
        
        // Update state
        kf_mean_ = mean;
        kf_covariance_ = covariance;
        
        // Update bounding box information
        float x = mean.at<float>(0);
        float y = mean.at<float>(1);
        float w = mean.at<float>(2);
        float h = mean.at<float>(3);
        
        w = std::max(0.1f, w);
        h = std::max(0.1f, h);
        tlwh = cv::Rect2f(x - w/2, y - h/2, w, h);
    } else {
        // If Kalman filter is not initialized, use simple linear prediction
        if (position_history.size() >= 2) {
            cv::Point2f last_pos = position_history.back();
            cv::Point2f prev_pos = position_history[position_history.size() - 2];
            cv::Point2f velocity = last_pos - prev_pos;

            cv::Point2f predicted_center = last_pos + velocity;
            tlwh.x = predicted_center.x - tlwh.width / 2;
            tlwh.y = predicted_center.y - tlwh.height / 2;
        }
        
        // Init Kalman filter
        initiateKalman(tlwh);
    }
}

void STrack::update(const cv::Rect2f& new_tlwh, float new_score, int new_frame_id) {
    // Update using Kalman filter
    if (kf_initialized_) {
        // Create measurement vector [x, y, w, h]
        cv::Mat measurement(4, 1, CV_32F);
        
        // Set measurement value (center point and width/height)
        float center_x = new_tlwh.x + new_tlwh.width / 2;
        float center_y = new_tlwh.y + new_tlwh.height / 2;
        measurement.at<float>(0) = center_x;
        measurement.at<float>(1) = center_y;
        measurement.at<float>(2) = new_tlwh.width;
        measurement.at<float>(3) = new_tlwh.height;
        
        // Update Kalman filter
        auto [mean, covariance] = updateKalman(kf_mean_, kf_covariance_, new_tlwh);
        kf_mean_ = mean;
        kf_covariance_ = covariance;
        
        float x = mean.at<float>(0) - mean.at<float>(2)/2;
        float y = mean.at<float>(1) - mean.at<float>(3)/2;
        float w = mean.at<float>(2);
        float h = mean.at<float>(3);
        
        // Use filtered result
        tlwh = cv::Rect2f(x, y, w, h);
    } else {
        // If Kalman filter is not initialized, use new measurement value
        tlwh = new_tlwh;
        initiateKalman(new_tlwh);
    }
    
    // Update other track information
    score = new_score;
    frame_id = new_frame_id;
    tracklet_len++;
    lost_frames_count = 0; 
    

    quality_score = (quality_score * 0.8f + new_score * 0.2f); 
    state = Tracked;
    position_history.push_back(center());
    confidence_history.push_back(score);
    
    // Update motion history information
    frame_history.push_back(new_frame_id);
    
    // Calculate and update speed history
    if (position_history.size() >= 2) {
        cv::Point2f current_pos = position_history.back();
        cv::Point2f prev_pos = position_history[position_history.size() - 2];
        cv::Point2f velocity = current_pos - prev_pos;
        velocity_history.push_back(velocity);
        
        // Calculate motion direction (radians)
        float direction = std::atan2(velocity.y, velocity.x);
        direction_history.push_back(direction);
        
        // Calculate motion speed (pixels/frame)
        float speed = cv::norm(velocity);
        speed_history.push_back(speed);
    } else {
        // First frame, initialize with zero speed
        velocity_history.push_back(cv::Point2f(0.0f, 0.0f));
        direction_history.push_back(0.0f);
        speed_history.push_back(0.0f);
    }
    
    // Limit history length (keep consistent length)
    const size_t max_history_size = 30;
    if (position_history.size() > max_history_size) {
        position_history.pop_front();
        velocity_history.pop_front();
        direction_history.pop_front();
        speed_history.pop_front();
        frame_history.pop_front();
    }
    if (confidence_history.size() > 10) {
        confidence_history.pop_front();
    }
    
    // Update confirmation status
    updateConfirmationStatus();
}

void STrack::markLost() {
    state = Lost;
    lost_frames_count++;
}

bool STrack::isLost() const {
    return state == Lost;
}

bool STrack::isActivated() const {
    return is_activated;
}

int STrack::getTimeSinceUpdate() const {
    // Return the number of frames since the last update
    return lost_frames_count;
}

void STrack::recoverWithMemory(const ROIMemory& memory, const Detection& detection, int frame_id) {
    // Recover tracking with memory information
    int original_id = memory.track_id;
    is_recovered = true;
    
    // Calculate recovery confidence
    if (!memory.confidence_history.empty()) {
        float avg_conf = 0.0f;
        for (float conf : memory.confidence_history) {
            avg_conf += conf;
        }
        recovery_confidence = avg_conf / memory.confidence_history.size();
    }
    
    // Set recovered ID information
    permanent_id = original_id;  // Recover target directly use original permanent ID
    is_confirmed = true;         // Recover target is considered confirmed
    is_real_target = true;
    track_id = original_id;
    
    // Update tracking information
    tlwh = detection.bbox;
    score = detection.confidence;
    this->frame_id = frame_id;
    state = Tracked;
    is_activated = true;
    lost_frames_count = 0; 
    
    start_frame = memory.last_seen_frame;
    tracklet_len += memory.lost_duration + 1;
    
    position_history.push_back(center());
    confidence_history.push_back(score);

    initiateKalman(tlwh);
}

// Kalman filter related methods implementation
void STrack::initKalmanFilter() {
    int ndim = 4;  // State dimension: x, y, w, h
    float dt = 1.0f;  // Time step
    
    // Init state transition matrix (8x8)
    kf_motion_mat_ = cv::Mat::eye(2 * ndim, 2 * ndim, CV_32F);
    for (int i = 0; i < ndim; ++i) {
        kf_motion_mat_.at<float>(i, ndim + i) = dt;
    }
    
    // Init observation matrix (4x8)
    kf_update_mat_ = cv::Mat::eye(ndim, 2 * ndim, CV_32F);
}

void STrack::initiateKalman(const cv::Rect2f& measurement) {
    // Init Kalman filter state from measurement value
    float x = measurement.x + measurement.width / 2;  
    float y = measurement.y + measurement.height / 2; 
    float w = measurement.width;
    float h = measurement.height;
    
    // Init mean vector [x,y,w,h,vx,vy,vw,vh]
    kf_mean_ = cv::Mat::zeros(8, 1, CV_32F);
    kf_mean_.at<float>(0) = x;
    kf_mean_.at<float>(1) = y;
    kf_mean_.at<float>(2) = w;
    kf_mean_.at<float>(3) = h;

    kf_covariance_ = cv::Mat::zeros(8, 8, CV_32F);

    float std_pos_x = 2 * kf_std_weight_position_ * w;
    float std_pos_y = 2 * kf_std_weight_position_ * h;
    float std_pos_w = 2 * kf_std_weight_position_ * w;
    float std_pos_h = 2 * kf_std_weight_position_ * h;
    
    float std_vel_x = 10 * kf_std_weight_velocity_ * w;
    float std_vel_y = 10 * kf_std_weight_velocity_ * h;
    float std_vel_w = 10 * kf_std_weight_velocity_ * w;
    float std_vel_h = 10 * kf_std_weight_velocity_ * h;
    
    kf_covariance_.at<float>(0, 0) = std_pos_x * std_pos_x;
    kf_covariance_.at<float>(1, 1) = std_pos_y * std_pos_y;
    kf_covariance_.at<float>(2, 2) = std_pos_w * std_pos_w;
    kf_covariance_.at<float>(3, 3) = std_pos_h * std_pos_h;
    kf_covariance_.at<float>(4, 4) = std_vel_x * std_vel_x;
    kf_covariance_.at<float>(5, 5) = std_vel_y * std_vel_y;
    kf_covariance_.at<float>(6, 6) = std_vel_w * std_vel_w;
    kf_covariance_.at<float>(7, 7) = std_vel_h * std_vel_h;
    
    kf_initialized_ = true;
}

std::pair<cv::Mat, cv::Mat> STrack::predictKalman() {
    // Predict step
    cv::Mat mean = kf_mean_.clone();
    cv::Mat covariance = kf_covariance_.clone();
    
    float w = mean.at<float>(2);
    float h = mean.at<float>(3);

    float std_pos_x = kf_std_weight_position_ * w;
    float std_pos_y = kf_std_weight_position_ * h;
    float std_pos_w = kf_std_weight_position_ * w;
    float std_pos_h = kf_std_weight_position_ * h;

    float std_vel_x = kf_std_weight_velocity_ * w;
    float std_vel_y = kf_std_weight_velocity_ * h;
    float std_vel_w = kf_std_weight_velocity_ * w;
    float std_vel_h = kf_std_weight_velocity_ * h;
    
    cv::Mat motion_cov = cv::Mat::zeros(8, 8, CV_32F);
    motion_cov.at<float>(0, 0) = std_pos_x * std_pos_x;
    motion_cov.at<float>(1, 1) = std_pos_y * std_pos_y;
    motion_cov.at<float>(2, 2) = std_pos_w * std_pos_w;
    motion_cov.at<float>(3, 3) = std_pos_h * std_pos_h;
    motion_cov.at<float>(4, 4) = std_vel_x * std_vel_x;
    motion_cov.at<float>(5, 5) = std_vel_y * std_vel_y;
    motion_cov.at<float>(6, 6) = std_vel_w * std_vel_w;
    motion_cov.at<float>(7, 7) = std_vel_h * std_vel_h;
    
    // Update mean
    mean = kf_motion_mat_ * mean;
    
    // Update covariance
    covariance = kf_motion_mat_ * covariance * kf_motion_mat_.t() + motion_cov;
    
    return {mean, covariance};
}

std::vector<cv::Point2f> STrack::predictFutureCenters(int n) const {
    std::vector<cv::Point2f> centers;
    if (!kf_initialized_ || n <= 0) {
        return centers;
    }
    
    cv::Mat mean = kf_mean_.clone();
    cv::Mat covariance = kf_covariance_.clone();
    for (int i = 0; i < n; ++i) {
        // 预测一步
        auto [next_mean, next_cov] = predictKalman(mean, covariance);
        mean = next_mean;
        covariance = next_cov;
        float x = mean.at<float>(0);
        float y = mean.at<float>(1);
        centers.emplace_back(x, y);
    }
    return centers;
}

// Overloaded predictKalman for future prediction
std::pair<cv::Mat, cv::Mat> STrack::predictKalman(const cv::Mat& mean, const cv::Mat& covariance) const {
    float w = mean.at<float>(2);
    float h = mean.at<float>(3);
    float std_pos_x = kf_std_weight_position_ * w;
    float std_pos_y = kf_std_weight_position_ * h;
    float std_pos_w = kf_std_weight_position_ * w;
    float std_pos_h = kf_std_weight_position_ * h;
    float std_vel_x = kf_std_weight_velocity_ * w;
    float std_vel_y = kf_std_weight_velocity_ * h;
    float std_vel_w = kf_std_weight_velocity_ * w;
    float std_vel_h = kf_std_weight_velocity_ * h;
    cv::Mat motion_cov = cv::Mat::zeros(8, 8, CV_32F);
    motion_cov.at<float>(0, 0) = std_pos_x * std_pos_x;
    motion_cov.at<float>(1, 1) = std_pos_y * std_pos_y;
    motion_cov.at<float>(2, 2) = std_pos_w * std_pos_w;
    motion_cov.at<float>(3, 3) = std_pos_h * std_pos_h;
    motion_cov.at<float>(4, 4) = std_vel_x * std_vel_x;
    motion_cov.at<float>(5, 5) = std_vel_y * std_vel_y;
    motion_cov.at<float>(6, 6) = std_vel_w * std_vel_w;
    motion_cov.at<float>(7, 7) = std_vel_h * std_vel_h;
    cv::Mat next_mean = kf_motion_mat_ * mean;
    cv::Mat next_cov = kf_motion_mat_ * covariance * kf_motion_mat_.t() + motion_cov;
    return {next_mean, next_cov};
}

std::pair<cv::Mat, cv::Mat> STrack::projectKalman(const cv::Mat& mean, const cv::Mat& covariance) {
    // Project to measurement space
    float w = mean.at<float>(2);
    float h = mean.at<float>(3);
    
    // Measurement noise standard deviation
    float std_x = kf_std_weight_position_ * w;
    float std_y = kf_std_weight_position_ * h;
    float std_w = kf_std_weight_position_ * w;
    float std_h = kf_std_weight_position_ * h;
    
    // Create measurement noise covariance matrix
    cv::Mat innovation_cov = cv::Mat::zeros(4, 4, CV_32F);
    innovation_cov.at<float>(0, 0) = std_x * std_x;
    innovation_cov.at<float>(1, 1) = std_y * std_y;
    innovation_cov.at<float>(2, 2) = std_w * std_w;
    innovation_cov.at<float>(3, 3) = std_h * std_h;
    
    // Project mean
    cv::Mat projected_mean = kf_update_mat_ * mean;
    
    // Project covariance
    cv::Mat projected_cov = kf_update_mat_ * covariance * kf_update_mat_.t() + innovation_cov;
    
    return {projected_mean, projected_cov};
}

std::pair<cv::Mat, cv::Mat> STrack::updateKalman(const cv::Mat& mean, const cv::Mat& covariance, 
                                                         const cv::Rect2f& measurement) {
    // Convert measurement value to center point and width/height
    float center_x = measurement.x + measurement.width / 2;
    float center_y = measurement.y + measurement.height / 2;
    
    // Create measurement vector [x, y, w, h]
    cv::Mat measurement_vec = cv::Mat::zeros(4, 1, CV_32F);
    measurement_vec.at<float>(0) = center_x;
    measurement_vec.at<float>(1) = center_y;
    measurement_vec.at<float>(2) = measurement.width;
    measurement_vec.at<float>(3) = measurement.height;
    
    // Project state to measurement space
    auto [projected_mean, projected_cov] = projectKalman(mean, covariance);
    
    // Calculate Kalman gain
    cv::Mat kalman_gain = covariance * kf_update_mat_.t() * projected_cov.inv();
    
    // Calculate innovation
    cv::Mat innovation = measurement_vec - projected_mean;
    
    // Update state mean
    cv::Mat new_mean = mean + kalman_gain * innovation;
    
    // Update state covariance
    cv::Mat new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.t();
    
    return {new_mean, new_covariance};
}

// 计算轨迹相似度
float STrack::calculateTrajectorySimilarity(const STrack* other, int num_frames) const {
    if (!other) return 0.0f;
    
    // Get position history of two trajectories
    const auto& pos1 = this->position_history;
    const auto& pos2 = other->position_history;
    
    if (pos1.empty() || pos2.empty()) return 0.0f;
    
    int n1 = std::min(static_cast<int>(pos1.size()), num_frames);
    int n2 = std::min(static_cast<int>(pos2.size()), num_frames);
    
    cv::Mat dtw_matrix = cv::Mat::zeros(n1 + 1, n2 + 1, CV_32F);

    for (int i = 0; i <= n1; ++i) dtw_matrix.at<float>(i, 0) = std::numeric_limits<float>::max();
    for (int j = 0; j <= n2; ++j) dtw_matrix.at<float>(0, j) = std::numeric_limits<float>::max();
    dtw_matrix.at<float>(0, 0) = 0.0f;
    
    // Dynamic programming to calculate DTW distance
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            float cost = cv::norm(pos1[pos1.size() - n1 + i - 1] - pos2[pos2.size() - n2 + j - 1]);
            dtw_matrix.at<float>(i, j) = cost + std::min({
                dtw_matrix.at<float>(i-1, j),
                dtw_matrix.at<float>(i, j-1),
                dtw_matrix.at<float>(i-1, j-1)
            });
        }
    }
    
    float dtw_distance = dtw_matrix.at<float>(n1, n2);
    
    // Normalize to [0,1] range, distance smaller similarity higher
    float similarity = std::exp(-dtw_distance / 100.0f);
    
    return similarity;
}


// GMM modeling function implementation (motion + space + history + ROI four types of features)
cv::Ptr<cv::ml::EM> fitGMM(
    const std::deque<cv::Point2f>& positions,
    const std::deque<cv::Point2f>& velocities,
    const std::deque<float>& directions,
    const std::deque<float>& speeds,
    const std::deque<int>& frame_ids,
    const cv::Rect& roi_bbox,
    int current_frame,
    int n_components) {
    
    if (positions.empty() || positions.size() < 3) {
        return cv::Ptr<cv::ml::EM>();
    }
    
    // Adaptive number of components: adjust according to data amount
    int adaptive_components = std::min(n_components, static_cast<int>(positions.size() / 3));
    if (adaptive_components < 1) adaptive_components = 1;
    
    // Create 8-dimensional feature vector: [x_abs, y_abs, x_rel, y_rel, vx, vy, direction, time_weight]
    cv::Mat samples(positions.size(), 8, CV_32F);
    
    for (size_t i = 0; i < positions.size(); ++i) {
        // 1. Absolute position feature
        samples.at<float>(i, 0) = positions[i].x;
        samples.at<float>(i, 1) = positions[i].y;
        
        // 2. ROI relative position feature (normalized)
        float rel_x = (positions[i].x - roi_bbox.x) / std::max(1.0f, static_cast<float>(roi_bbox.width));
        float rel_y = (positions[i].y - roi_bbox.y) / std::max(1.0f, static_cast<float>(roi_bbox.height));
        rel_x = std::max(-0.5f, std::min(1.5f, rel_x));
        rel_y = std::max(-0.5f, std::min(1.5f, rel_y));
        samples.at<float>(i, 2) = rel_x;
        samples.at<float>(i, 3) = rel_y;
        
        // 3. Motion information feature
        if (i < velocities.size()) {
            samples.at<float>(i, 4) = velocities[i].x;
            samples.at<float>(i, 5) = velocities[i].y;
        } else {
            samples.at<float>(i, 4) = 0.0f;
            samples.at<float>(i, 5) = 0.0f;
        }
        
        // 4. Motion direction feature
        if (i < directions.size()) {
            samples.at<float>(i, 6) = directions[i];
        } else {
            samples.at<float>(i, 6) = 0.0f;
        }
        
        // 5. Time weight feature (exponential decay)
        if (i < frame_ids.size()) {
            float time_weight = std::exp(-0.1f * (current_frame - frame_ids[i]));
            samples.at<float>(i, 7) = time_weight;
        } else {
            samples.at<float>(i, 7) = 1.0f;
        }
    }
    
    // Create and train GMM model
    auto em = cv::ml::EM::create();
    em->setClustersNumber(adaptive_components);
    em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
    em->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.01));
    
    if (samples.rows >= adaptive_components) {
        try {
            em->trainEM(samples);
            return em;
        } catch (const cv::Exception& e) {
            return cv::Ptr<cv::ml::EM>();
        }
    }
    
    return cv::Ptr<cv::ml::EM>();
}

// GMM probability calculation function
float GMMProbability(
    const cv::Ptr<cv::ml::EM>& gmm,
    const cv::Point2f& position,
    const cv::Point2f& velocity,
    float direction,
    float speed,
    const cv::Rect& roi_bbox) {
    
    if (!gmm || gmm->getMeans().empty()) return 0.0f;
    
    // Create 8-dimensional sample feature vector
    cv::Mat sample(1, 8, CV_32F);
    
    // 1. Spatial position feature
    sample.at<float>(0, 0) = position.x;
    sample.at<float>(0, 1) = position.y;
    
    // 2. ROI relative position feature (normalized)
    float rel_x = (position.x - roi_bbox.x) / std::max(1.0f, static_cast<float>(roi_bbox.width));
    float rel_y = (position.y - roi_bbox.y) / std::max(1.0f, static_cast<float>(roi_bbox.height));
    // Allow slightly out of ROI boundary
    rel_x = std::max(-0.5f, std::min(1.5f, rel_x));
    rel_y = std::max(-0.5f, std::min(1.5f, rel_y));
    sample.at<float>(0, 2) = rel_x;
    sample.at<float>(0, 3) = rel_y;
    
    // 3. Motion information feature
    sample.at<float>(0, 4) = velocity.x;
    sample.at<float>(0, 5) = velocity.y;
    sample.at<float>(0, 6) = direction;
    
    // 4. Current time weight (set to 1.0, representing current frame)
    sample.at<float>(0, 7) = 1.0f;
    
    try {
        // Use GMM to calculate probability density
        cv::Vec2d result = gmm->predict2(sample, cv::noArray());
        float log_likelihood = static_cast<float>(result[1]);
        
        // Convert to probability value, and handle numerical stability
        float probability = std::exp(std::max(-50.0f, log_likelihood));
        
        return probability;
    } catch (const cv::Exception& e) {
        return 0.0f;
    }
}

// GMM functionality has been moved to TPTrack's memory recovery feature
// Individual track GMM modeling is no longer used

// Optical flow feature update method
void STrack::updateOpticalFlowFeatures(const cv::Mat& current_gray_patch, 
                                               const std::vector<cv::Point2f>& flow_points) {
    // Update previous frame's gray image patch
    prev_gray_patch_ = current_gray_patch.clone();
    
    // Update optical flow feature points
    optical_flow_points_ = flow_points;
}

} // namespace tracking 