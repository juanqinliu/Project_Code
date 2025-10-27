#include "tracking/TPTrack.h"
#include "tracking/STrack.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>                     
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>    
#include <opencv2/video/tracking.hpp> 
#include <limits>
#include <opencv2/ml.hpp>
#include "common/Flags.h"
#include "tracking/Hungarian.h"

// Silence LOG_INFO outputs in this file
#ifdef LOG_INFO
#undef LOG_INFO
#endif
#define LOG_INFO(msg) do { } while (0)

namespace tracking {

// ===== 🔥 Geometric Constraints Implementation =====

bool GeometricConstraints::checkConstraints(
    const STrack* track,
    const STrack* detection,
    const MatchingConstraints& constraints) {
    
    cv::Point2f last_pos = track->center();
    cv::Point2f pred_pos = track->getPredictedCenter();
    cv::Point2f det_pos = detection->center();
    
    float dist_to_last = cv::norm(last_pos - det_pos);
    float dist_to_pred = cv::norm(pred_pos - det_pos);
    
    // Hard constraint: distance to last position
    if (dist_to_last > constraints.max_distance_to_last) {
        return false;
    }
    
    // Adaptive constraint: distance to predicted position
    float avg_size = (track->tlwh.width + track->tlwh.height +
                     detection->tlwh.width + detection->tlwh.height) / 4.0f;
    float gating_dist = constraints.getGatingDistance(avg_size);
    
    if (dist_to_pred > gating_dist) {
        return false;
    }
    
    return true;
}

void GeometricConstraints::applyGeometricGating(
    cv::Mat& cost_matrix,
    const std::vector<STrack*>& tracks,
    const std::vector<STrack*>& detections,
    const MatchingConstraints& constraints,
    bool verbose) {
    
    if (cost_matrix.empty()) return;
    
    int rejected_count = 0;
    
    for (int i = 0; i < cost_matrix.rows; ++i) {
        for (int j = 0; j < cost_matrix.cols; ++j) {
            if (!checkConstraints(tracks[i], detections[j], constraints)) {
                cost_matrix.at<float>(i, j) = 1.0f;  // Reject match
                rejected_count++;
                
                if (verbose) {
                    cv::Point2f last_pos = tracks[i]->center();
                    cv::Point2f det_pos = detections[j]->center();
                    float dist = cv::norm(last_pos - det_pos);
                    std::cout << "  [Gating] Rejected: Track ID-" << tracks[i]->displayId()
                              << " vs Det(" << det_pos.x << "," << det_pos.y << ")"
                              << " | dist=" << dist << std::endl;
                }
            }
        }
    }
    
    if (verbose && rejected_count > 0) {
        std::cout << "[Gating] Total rejected: " << rejected_count 
                  << " / " << (cost_matrix.rows * cost_matrix.cols) << std::endl;
    }
}

MatchingConstraints GeometricConstraints::getStandardConstraints() {
    return MatchingConstraints{
        .max_distance_to_last = 100.0f,
        .gating_distance_factor = 3.0f,
        .base_gating_distance = 50.0f,
        .strict_mode = false
    };
}

MatchingConstraints GeometricConstraints::getStrictConstraints() {
    return MatchingConstraints{
        .max_distance_to_last = 80.0f,
        .gating_distance_factor = 2.0f,
        .base_gating_distance = 40.0f,
        .strict_mode = true
    };
}

// ===== Original ByteTrack Implementation =====

// Original ByteTrack cost matrix builder (IoU-only)
cv::Mat TPTrack::buildOriginalByteTrackCostMatrix(
    const std::vector<STrack*>& tracks,
    const std::vector<STrack*>& detections,
    CostMatrixType matrix_type) {
    if (tracks.empty() || detections.empty()) {
        return cv::Mat();
    }
    cv::Mat iou_dists = calculateIoUMatrix(tracks, detections);
    if (iou_dists.empty() || iou_dists.rows == 0 || iou_dists.cols == 0) {
        return iou_dists;
    }
    cv::Mat cost_matrix = iou_dists.clone();
    if (matrix_type == RESCUE_MATCH) {
        // Can add specific adjustments here, but original ByteTrack mainly relies on different thresholds
    }
    for (int i = 0; i < cost_matrix.rows; ++i) {
        for (int j = 0; j < cost_matrix.cols; ++j) {
            cost_matrix.at<float>(i, j) = std::max(0.0f, std::min(1.0f, cost_matrix.at<float>(i, j)));
        }
    }
    return cost_matrix;
}

// JCMA implementation (IoU + distance + relative + motion)
cv::Mat TPTrack::buildEnhancedCostMatrix(
    const std::vector<STrack*>& tracks,
    const std::vector<STrack*>& detections,
    CostMatrixType matrix_type) {
    

    if (tracks.empty() || detections.empty()) {
        return cv::Mat();
    }
    
    // Create cost matrix
    cv::Mat cost_matrix(tracks.size(), detections.size(), CV_32F);
    
    // Iterate over all tracks and detections, calculate comprehensive cost
    for (size_t i = 0; i < tracks.size(); ++i) {
        for (size_t j = 0; j < detections.size(); ++j) {
            
            // 1. IoU cost - geometric overlap
            float iou = calculateIoU(tracks[i]->tlwh, detections[j]->tlwh);
            float iou_cost = 1.0f - iou; // Convert to distance metric
            
            // 2. Euclidean distance cost - spatial position distance
            cv::Point2f track_center = tracks[i]->getPredictedCenter();
            cv::Point2f detection_center = detections[j]->center();
            float distance = cv::norm(track_center - detection_center);
            
            // Adaptive normalization based on target size
            float size_factor = (tracks[i]->tlwh.width + tracks[i]->tlwh.height + 
                               detections[j]->tlwh.width + detections[j]->tlwh.height) / 4.0f;
            float distance_cost = std::min(1.0f, distance / (size_factor * 2.0f));
            
            // 3. Appearance cost - color histogram + HOG features
            // float appearance_cost = calculateAppearanceCost(tracks[i], detections[j]);
            
            // 4. Relative position cost - relative position between multiple targets
            float relative_position_cost = calculateRelativePositionCost(tracks[i], detections[j], tracks);
            
            // 5. Motion cost - solve the problem of drone mutation
            float motion_cost = calculateMotionCost(tracks[i], detections[j]);

            // Adjust weights based on matching type - add motion feature constraint
            float iou_weight = 0.5f;        
            float distance_weight = 0.1f;  
            // float appearance_weight = 0.00f; 
            float relative_weight = 0.3f;  
            float motion_weight = 0.1f;  
            
            // float appearance_cost = 0.0f;
            // Calculate comprehensive cost - add motion feature constraint
            float final_cost = iou_weight * iou_cost + 
                              distance_weight * distance_cost + 
                            //   appearance_weight * appearance_cost + 
                              relative_weight * relative_position_cost +
                              motion_weight * motion_cost;

            // Add small penalty to unconfirmed/lost tracks, prioritize matching confirmed and stable tracks
            if (!tracks[i]->is_confirmed) {
                final_cost = std::min(1.0f, final_cost + 0.05f);
            }
            if (tracks[i]->state == STrack::Lost) {
                final_cost = std::min(1.0f, final_cost + 0.08f);
            }
            
            // Normalize to [0,1] range
            final_cost = std::max(0.0f, std::min(1.0f, final_cost));
            
            // Store in cost matrix
            cost_matrix.at<float>(i, j) = final_cost;
        }
    }
    
    return cost_matrix;
}

// Reset tracker state implementation
void TPTrack::resetTrackerState() {
    // Clear all track containers
    tracked_stracks_.clear();
    lost_stracks_.clear();
    removed_stracks_.clear();
    
    frame_id_ = 0;
    LOG_INFO("⚠️ Tracker state reset");
}


// TPTrack implementation
TPTrack::TPTrack(const Config& config, int frame_rate, 
                                        std::shared_ptr<ROIManager> roi_manager)
    : config_(config), frame_id_(0), roi_manager_(roi_manager) {}


static void removeDuplicateWithinTracked(std::vector<std::unique_ptr<tracking::STrack>>& tracked);

std::vector<std::unique_ptr<STrack>> TPTrack::update(
    const std::vector<cv::Rect2f>& bboxes, 
    const std::vector<float>& scores, 
    const std::vector<int>& classes) {
    
    std::vector<Detection> detections;
    for (size_t i = 0; i < bboxes.size(); ++i) {
        Detection det;
        det.bbox = bboxes[i];
        det.confidence = scores[i];
        det.class_id = classes[i];
        detections.push_back(det);
    }
    
    return update(detections);
}

std::vector<std::unique_ptr<STrack>> TPTrack::update(
    const std::vector<cv::Rect2f>& bboxes, 
    const std::vector<float>& scores, 
    const std::vector<int>& classes, 
    const cv::Mat& frame) {
    
    std::vector<Detection> detections;
    for (size_t i = 0; i < bboxes.size(); ++i) {
        Detection det;
        det.bbox = bboxes[i];
        det.confidence = scores[i];
        det.class_id = classes[i];
        detections.push_back(det);
    }
    
    return update(detections);
}

std::vector<std::unique_ptr<STrack>> TPTrack::update(
    const std::vector<Detection>& detections,
    const std::unordered_map<int, std::vector<Detection>>& roi_detections) {
    
    frame_id_++;
    std::vector<std::unique_ptr<STrack>> activated_stracks;
    std::vector<std::unique_ptr<STrack>> refind_stracks;
    std::vector<std::unique_ptr<STrack>> lost_stracks;
    std::vector<std::unique_ptr<STrack>> removed_stracks;
    std::set<int> processed_track_ids; 
    
    // 1. Initialize detections and group by confidence
    auto det_stracks = initTrack(detections);
    
    // Separate high confidence and low confidence detections
    std::vector<std::unique_ptr<STrack>> dets_high, dets_low;
    for (auto& track : det_stracks) {
        if (track->score >= config_.track_high_thresh) {
            dets_high.push_back(std::move(track));
        } else if (track->score >= config_.track_low_thresh) {
            dets_low.push_back(std::move(track));
        }
    }
    
    // 2. Separate unconfirmed and tracked targets
    std::vector<STrack*> unconfirmed, tracked_stracks_raw;
    for (auto& track : tracked_stracks_) {
        if (!track->is_activated) {
            unconfirmed.push_back(track.get());
        } else {
            tracked_stracks_raw.push_back(track.get());
        }
    }
    
    // 3. Prepare tracking pool and predict
    std::vector<STrack*> strack_pool;
    for (auto& track : tracked_stracks_) {
        strack_pool.push_back(track.get());
    }
    for (auto& track : lost_stracks_) {
        strack_pool.push_back(track.get());
    }
    
    multiPredict(strack_pool);
    
    // 4. First stage: main matching - high confidence detection with predicted tracks
    // Check if in global mode
    bool is_global_mode = false;
    if (roi_manager_) {
        is_global_mode = (roi_manager_->getROIs().size() <= 1);
    }
    
    std::vector<STrack*> dets_high_ptr;
    for (auto& det : dets_high) {
            dets_high_ptr.push_back(det.get());
    }

    std::vector<int> u_track, u_detection;
    
    if (!strack_pool.empty() && !dets_high_ptr.empty()) {
        // Build cost matrix for main matching
    cv::Mat cost_matrix;
    if (use_original_bytetrack_ || !config_.use_enhanced_matching || !roi_manager_) {
        cost_matrix = calculateIoUMatrix(strack_pool, dets_high_ptr);
        LOG_INFO("Use original ByteTrack IoU cost matrix for first stage matching");
    } else {
        cost_matrix = buildEnhancedCostMatrix(strack_pool, dets_high_ptr, PRIMARY_MATCH);
        LOG_INFO("Use enhanced cost matrix for first stage matching (IoU + distance + appearance + relative position)");
    }
    
    // 🔥 Apply geometric gating (unified and clean)
    if (!cost_matrix.empty()) {
        MatchingConstraints constraints = GeometricConstraints::getStandardConstraints();
        GeometricConstraints::applyGeometricGating(
            cost_matrix, strack_pool, dets_high_ptr, constraints, false);
    }

    // Print statistics of first stage cost matrix
    if (!cost_matrix.empty()) {
        double min_val, max_val, mean_val, std_val;
        cv::minMaxLoc(cost_matrix, &min_val, &max_val);
        cv::Scalar mean_scalar = cv::mean(cost_matrix);
        cv::Scalar std_scalar, mean_scalar_for_std;
        cv::meanStdDev(cost_matrix, mean_scalar_for_std, std_scalar);
        mean_val = mean_scalar[0];
        std_val = std_scalar[0];

    
    }

        
        float primary_match_threshold;
        if (use_original_bytetrack_) {
            primary_match_threshold = config_.match_thresh;
        } else if (is_global_mode) {
            primary_match_threshold = 0.75f;
        } else {
            primary_match_threshold = 0.60f;
        }
        
        LOG_INFO("First stage matching threshold: " << primary_match_threshold);
        LOG_INFO("Global mode: " << (is_global_mode ? "Yes" : "No"));
        
        if (!cost_matrix.empty() && cost_matrix.rows > 0 && cost_matrix.cols > 0) {
        auto result = hungarianAssignment(cost_matrix, primary_match_threshold);
        auto primary_matches = std::get<0>(result);
        u_track = std::get<1>(result); 
        u_detection = std::get<2>(result);
            
            LOG_INFO("First stage matching result: " << primary_matches.size() << " matches");
        
        for (auto& [track_idx, det_idx] : primary_matches) {
            auto* track = strack_pool[track_idx];
            auto* det = dets_high_ptr[det_idx];
            float match_cost = cost_matrix.at<float>(track_idx, det_idx);
            
                LOG_INFO("First stage matching: track ID-" << track->displayId() 
                          << " matches detection (confidence: " << det->score 
                          << " match cost: " << match_cost << " threshold: " << primary_match_threshold << ")");

            if (track->state == STrack::Tracked) {
                track->update(det->tlwh, det->score, frame_id_);
                track->updateConfirmationStatus();
                
                for (auto it = tracked_stracks_.begin(); it != tracked_stracks_.end(); ++it) {
                    if (it->get() == track) {
                        activated_stracks.push_back(std::make_unique<STrack>(*track));
                        break;
                    }
                }
            } else {
                // Reactivate lost track
                track->update(det->tlwh, det->score, frame_id_);
                track->state = STrack::Tracked;
                track->updateConfirmationStatus();
                
                for (auto it = lost_stracks_.begin(); it != lost_stracks_.end(); ++it) {
                    if (it->get() == track) {
                        refind_stracks.push_back(std::make_unique<STrack>(*track));
                        break;
                    }
                }
            }
            
            // Mark used detection as processed
            for (auto it = dets_high.begin(); it != dets_high.end(); ++it) {
                if (it->get() == det) {
                    dets_high.erase(it);
                    break;
                }
            }
        }
    } else {
        // Matrix is empty, all tracks and detections are not matched
            u_track.resize(strack_pool.size());
            u_detection.resize(dets_high_ptr.size());
            std::iota(u_track.begin(), u_track.end(), 0);
            std::iota(u_detection.begin(), u_detection.end(), 0);
        }
    } else {
        // No tracks or detections, all are not matched
        u_track.resize(strack_pool.size());
        u_detection.resize(dets_high_ptr.size());
        std::iota(u_track.begin(), u_track.end(), 0);
        std::iota(u_detection.begin(), u_detection.end(), 0);
    }
    
    std::vector<STrack*> r_tracked_stracks;
    for (int i : u_track) {
        if (i < strack_pool.size() && (strack_pool[i]->state == STrack::Tracked || strack_pool[i]->state == STrack::Lost)) {
            r_tracked_stracks.push_back(strack_pool[i]);
        }
    }
    
    // Print detailed information of tracks involved in rescue matching
    for (size_t i = 0; i < r_tracked_stracks.size(); ++i) {
        auto* track = r_tracked_stracks[i];

    }
    
    // Print detailed information of low confidence detections
    for (size_t i = 0; i < dets_low.size(); ++i) {
        auto* det = dets_low[i].get();
        LOG_INFO("Low confidence detection[" << i << "]: confidence:" << det->score 
                  << " position:(" << det->center().x << "," << det->center().y << ")"
                  << " size:(" << det->tlwh.width << "x" << det->tlwh.height << ")");
    }
    
    if (!r_tracked_stracks.empty() && !dets_low.empty()) {
        std::vector<STrack*> dets_low_ptr;
        for (auto& det : dets_low) {
            dets_low_ptr.push_back(det.get());
        }
        
        // Build cost matrix for rescue matching
        cv::Mat rescue_cost_matrix;
        if (use_original_bytetrack_ || !config_.use_enhanced_matching || !roi_manager_) {
            rescue_cost_matrix = calculateIoUMatrix(r_tracked_stracks, dets_low_ptr);
            LOG_INFO("Use original ByteTrack IoU cost matrix for rescue matching");
        } else {
            rescue_cost_matrix = buildEnhancedCostMatrix(r_tracked_stracks, dets_low_ptr, RESCUE_MATCH);
            LOG_INFO("Use enhanced cost matrix for rescue matching (IoU + distance + appearance + relative position)");
        }
        
        // Apply geometric gating before matching (prevent distant false matches)
        if (!rescue_cost_matrix.empty()) {
            for (int i = 0; i < rescue_cost_matrix.rows; ++i) {
                const auto* track = r_tracked_stracks[i];
                const cv::Point2f pred_center = track->getPredictedCenter();
                const float track_w = track->tlwh.width;
                const float track_h = track->tlwh.height;
                for (int j = 0; j < rescue_cost_matrix.cols; ++j) {
                    const auto* det = dets_low_ptr[j];
                    const cv::Point2f det_center = det->center();
                    const float det_w = det->tlwh.width;
                    const float det_h = det->tlwh.height;

                    const float center_distance = cv::norm(pred_center - det_center);
                    const float avg_size = (track_w + track_h + det_w + det_h) / 4.0f;
                    float gating_distance = std::max(30.0f, avg_size * 2.5f);
                    if (track->state == STrack::Lost) {
                        gating_distance *= 1.5f; // 
                    }

                    const float iou = calculateIoU(track->tlwh, det->tlwh);

                    if (center_distance > gating_distance) {
                        rescue_cost_matrix.at<float>(i, j) = 1.0f;
                        continue;
                    }

                    if (iou < 0.01f && center_distance > gating_distance * 0.7f) {
                        rescue_cost_matrix.at<float>(i, j) = 1.0f;
                        continue;
                    }
                }
            }
        }
        
        // LOG_INFO("Rescue matching cost matrix size: [" << rescue_cost_matrix.rows << " x " << rescue_cost_matrix.cols << "]");
        
        if (!rescue_cost_matrix.empty()) {
            double min_val, max_val, mean_val, std_val;
            cv::minMaxLoc(rescue_cost_matrix, &min_val, &max_val);
            cv::Scalar mean_scalar = cv::mean(rescue_cost_matrix);
            cv::Scalar std_scalar, mean_scalar_for_std;
            cv::meanStdDev(rescue_cost_matrix, mean_scalar_for_std, std_scalar);
            mean_val = mean_scalar[0];
            std_val = std_scalar[0];
            
            // LOG_INFO("救援匹配代价矩阵统计:");
            // LOG_INFO("  最小值: " << min_val << " 最大值: " << max_val);
            // LOG_INFO("  平均值: " << mean_val << " 标准差: " << std_val);
        }
        
        if (!rescue_cost_matrix.empty() && rescue_cost_matrix.rows > 0 && rescue_cost_matrix.cols > 0) {
            bool is_global_mode = false;
            if (roi_manager_) {
                is_global_mode = (roi_manager_->getROIs().size() <= 1);
            }
            
            float rescue_thresh;
            if (use_original_bytetrack_) {
                rescue_thresh = 0.6f;
            } else if (is_global_mode) {
                rescue_thresh = 0.85f;
            } else {
                rescue_thresh = 0.8f;
            }
            
            
            auto result = hungarianAssignment(rescue_cost_matrix, rescue_thresh);
            auto rescue_matches = std::get<0>(result);
            std::vector<int> u_r_track = std::get<1>(result);
            std::vector<int> u_r_detection = std::get<2>(result);
            
            // Process rescue matching results
            for (auto& [track_idx, det_idx] : rescue_matches) {
                auto* track = r_tracked_stracks[track_idx];
                auto* det = dets_low_ptr[det_idx];
                float match_cost = rescue_cost_matrix.at<float>(track_idx, det_idx);

                if (track->state == STrack::Tracked) {
                    track->update(det->tlwh, det->score, frame_id_);
                    track->updateConfirmationStatus();
                    
                    for (auto it = tracked_stracks_.begin(); it != tracked_stracks_.end(); ++it) {
                        if (it->get() == track) {
                            activated_stracks.push_back(std::make_unique<STrack>(*track));
                            break;
                        }
                    }
                } else {
                    track->update(det->tlwh, det->score, frame_id_);
                    track->state = STrack::Tracked;
                    track->updateConfirmationStatus();
                    
                    for (auto it = lost_stracks_.begin(); it != lost_stracks_.end(); ++it) {
                        if (it->get() == track) {
                            refind_stracks.push_back(std::make_unique<STrack>(*track));
                            break;
                        }
                    }
                }
            
                for (auto it = dets_low.begin(); it != dets_low.end(); ++it) {
                    if (it->get() == det) {
                        dets_low.erase(it);
                        break;
                    }
                }
            }
            

            
            // Update unmatched track indices
            std::vector<int> new_u_track;
            for (int i : u_track) {
                bool still_unmatched = true;
                for (size_t j = 0; j < r_tracked_stracks.size(); ++j) {
                    if (strack_pool[i] == r_tracked_stracks[j]) {
                        bool found = false;
                        for (size_t k = 0; k < u_r_track.size(); ++k) {
                            if (j == static_cast<size_t>(u_r_track[k])) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            still_unmatched = false;
                            break;
                        }
                    }
                }
                if (still_unmatched) {
                    new_u_track.push_back(i);
                }
            }
            u_track = std::move(new_u_track);
        } else {
            // LOG_INFO("⚠️ Rescue matching cost matrix is empty or invalid, skip rescue matching");
        }
    } else {
        // LOG_INFO("⚠️ Rescue matching condition not met:");
        // LOG_INFO("  number of tracks involved in rescue matching: " << r_tracked_stracks.size());
        // LOG_INFO("  number of low confidence detections: " << dets_low.size());
    }
    

    std::vector<std::unique_ptr<STrack>> u_high_detection;
    std::vector<STrack*> u_high_detection_ptr;
    
    // Collect unmatched high confidence detections
    for (int i : u_detection) {
        if (i < static_cast<int>(dets_high_ptr.size())) {
            for (auto it = dets_high.begin(); it != dets_high.end(); ++it) {
                if (it->get() == dets_high_ptr[i]) {
                    u_high_detection.push_back(std::move(*it));
                    u_high_detection_ptr.push_back(u_high_detection.back().get());
                    dets_high.erase(it);
                    break;
                }
            }
        }
    }
    
    // LOG_INFO("Collected unmatched high confidence detections number: " << u_high_detection.size());
    
    // Perform GMM memory recovery (delete ROI dependency)
    if (FLAGS_enable_gmm_memory_recovery && !u_high_detection_ptr.empty()) {
        // Generate current frame lost tracks for GMM memory recovery
        std::vector<std::unique_ptr<STrack>> current_frame_lost_stracks;
        std::set<int> processed_track_ids;  
        for (int i : u_track) {
            if (i < static_cast<int>(strack_pool.size()) && strack_pool[i]->state == STrack::Tracked) {
                auto* lost_track = strack_pool[i];
                auto lost_track_copy = std::make_unique<STrack>(*lost_track);
                lost_track_copy->markLost();
                current_frame_lost_stracks.push_back(std::move(lost_track_copy));
                processed_track_ids.insert(lost_track->displayId());
            }
        }
        
        std::vector<std::unique_ptr<STrack>> high_detections;
        for (auto& det : u_high_detection) {
            high_detections.push_back(std::move(det));
        }
        
        auto [remaining_dets, recovered] = processGMMMemoryRecovery(high_detections, roi_detections, current_frame_lost_stracks);
        
        for (auto& recovered_track : recovered) {
            activated_stracks.push_back(std::move(recovered_track));
        }
        
        u_high_detection = std::move(remaining_dets);
        u_high_detection_ptr.clear();
        for (auto& det : u_high_detection) {
            u_high_detection_ptr.push_back(det.get());
        }
    }
    
    // 7. Fourth stage: unmatched unconfirmed tracks matching
    // LOG_INFO("=== Fourth stage: unmatched unconfirmed tracks matching ===");
    // LOG_INFO("Unconfirmed tracks number: " << unconfirmed.size());
    // LOG_INFO("Remaining high confidence detections number: " << u_high_detection_ptr.size());
    
    if (!unconfirmed.empty() && !u_high_detection_ptr.empty()) {
        // Build cost matrix for unmatched unconfirmed tracks matching
        cv::Mat unconfirmed_cost_matrix;
        if (use_original_bytetrack_ || !config_.use_enhanced_matching || !roi_manager_) {
            unconfirmed_cost_matrix = calculateIoUMatrix(unconfirmed, u_high_detection_ptr);
        } else {
            unconfirmed_cost_matrix = buildEnhancedCostMatrix(unconfirmed, u_high_detection_ptr, UNCONFIRMED_MATCH);
        }
        
        bool is_global_mode = false;
        if (roi_manager_) {
            is_global_mode = (roi_manager_->getROIs().size() <= 1);
        }
        
        float unconfirmed_thresh;
        if (use_original_bytetrack_) {
            unconfirmed_thresh = 0.7f;
        } else if (is_global_mode) {
            unconfirmed_thresh = 0.8f;
        } else {
            unconfirmed_thresh = 0.60f;
        }
        
        // LOG_INFO("Fourth stage matching threshold: " << unconfirmed_thresh);
        
        auto [u_matches, u_unconfirmed, u_detection_final] = 
            hungarianAssignment(unconfirmed_cost_matrix, unconfirmed_thresh);
        
        // LOG_INFO("Fourth stage matching result: " << u_matches.size() << " matches");
        
        // Process unmatched unconfirmed tracks matching result
    for (auto& [track_idx, det_idx] : u_matches) {
        if (track_idx >= 0 && track_idx < static_cast<int>(unconfirmed.size()) && 
            det_idx >= 0 && det_idx < static_cast<int>(u_high_detection_ptr.size())) {
            
            auto* track = unconfirmed[track_idx];
            auto* det = u_high_detection_ptr[det_idx];

            track->update(det->tlwh, det->score, frame_id_);
            track->updateConfirmationStatus();
            
            for (auto it = tracked_stracks_.begin(); it != tracked_stracks_.end(); ++it) {
                if (it->get() == track) {
                    activated_stracks.push_back(std::make_unique<STrack>(*track));
                    break;
                }
            }
                
            for (auto it = u_high_detection.begin(); it != u_high_detection.end(); ++it) {
                if (it->get() == det) {
                    u_high_detection.erase(it);
                    u_high_detection_ptr.clear();
                    for (auto& det : u_high_detection) {
                        u_high_detection_ptr.push_back(det.get());
                    }
                    break;
                }
            }
        }
    }
        
        // Remove unmatched unconfirmed tracks
        for (int i : u_unconfirmed) {
            if (i >= 0 && i < static_cast<int>(unconfirmed.size())) {
                unconfirmed[i]->state = STrack::Removed;
                for (auto it = tracked_stracks_.begin(); it != tracked_stracks_.end(); ++it) {
                    if (it->get() == unconfirmed[i]) {
                        removed_stracks.push_back(std::make_unique<STrack>(*unconfirmed[i]));
                        break;
                    }
                }
            }
        }
        
        // Collect unmatched high confidence detections
        std::vector<std::unique_ptr<STrack>> remaining_high_dets;
        for (int i : u_detection_final) {
            if (i >= 0 && i < static_cast<int>(u_high_detection.size())) {
                remaining_high_dets.push_back(std::move(u_high_detection[i]));
            }
        }
        u_high_detection = std::move(remaining_high_dets);
    }
    
    // 8. Mark unmatched tracks as lost
    for (int i : u_track) {
        if (i < static_cast<int>(strack_pool.size()) && strack_pool[i]->state == STrack::Tracked) {
            auto* lost_track = strack_pool[i];
            // 🔥 Check if the track has been processed in the memory recovery stage
            if (processed_track_ids.find(lost_track->displayId()) == processed_track_ids.end()) {
            lost_track->markLost();
            lost_stracks.push_back(std::make_unique<STrack>(*lost_track));
                // LOG_INFO("Mark track ID-" << lost_track->displayId() << " as lost state");
            } else {
                // 🔥 If the track has been processed in the memory recovery stage, mark it as lost
                lost_track->markLost();
                lost_stracks.push_back(std::make_unique<STrack>(*lost_track));
                // LOG_INFO("Track ID-" << lost_track->displayId() << " has been processed in the memory recovery stage, mark it as lost state");
            }
        }
    }
    
    // 9. Initialize new tracks - process all unmatched detections
    std::vector<std::unique_ptr<STrack>> unmatched_detections;
    
    // Merge all unmatched detections
    for (auto& det : u_high_detection) {
        unmatched_detections.push_back(std::move(det));
    }
    for (auto& det : dets_low) {
        unmatched_detections.push_back(std::move(det));
    }
    
    // Directly initialize new tracks
    for (auto& track : unmatched_detections) {
        if (track->score >= 0.6f && !track->is_recovered) {
            // LOG_INFO("Create new track: detection confidence " << track->score << " satisfies high confidence requirement");
            track->activate(frame_id_);
            activated_stracks.push_back(std::move(track));
        } else if (track->score >= config_.new_track_thresh && track->score < 0.6f) {
            // LOG_INFO("Skip medium confidence detection to create new track: confidence " << track->score << " does not satisfy high confidence requirement (≥0.6)");
        } else {
            // LOG_INFO("Discard low confidence detection: confidence " << track->score << " is below new track threshold " << config_.new_track_thresh);
        }
    }
    
    // 10. Update track state
    for (auto& track : lost_stracks_) {
        if (frame_id_ - track->frame_id > config_.track_buffer) {
            track->state = STrack::Removed;
            removed_stracks.push_back(std::make_unique<STrack>(*track));
        }
    }
    
    // 11. Merge all activated tracks
    auto all_activated = jointSTracks(activated_stracks, refind_stracks);
    
    // 12. Update tracker state
    std::vector<std::unique_ptr<STrack>> active_tracks;
    for (auto& track : tracked_stracks_) {
        if (track->state == STrack::Tracked) {
            active_tracks.push_back(std::make_unique<STrack>(*track));
        }
    }
    
    tracked_stracks_ = jointSTracks(active_tracks, all_activated);
    
    // Update lost tracks
    std::vector<std::unique_ptr<STrack>> current_lost;
    for (auto& track : lost_stracks_) {
        if (track->state != STrack::Removed) {
            current_lost.push_back(std::make_unique<STrack>(*track));
        }
    }
    
    // Add new lost tracks to the current lost tracks list
    for (auto& track : lost_stracks) {
        bool already_in_tracked = false;
        for (const auto& active_track : tracked_stracks_) {
            if (active_track->displayId() == track->displayId()) {
                already_in_tracked = true;
                break;
            }
        }
        
        if (!already_in_tracked) {
            current_lost.push_back(std::move(track));
        }
    }
    
    lost_stracks_ = std::move(current_lost);
    lost_stracks_ = subSTracks(lost_stracks_, tracked_stracks_);
    
    // Remove duplicates
    auto [final_tracked, final_lost] = removeDuplicateSTracks(tracked_stracks_, lost_stracks_);
    tracked_stracks_ = std::move(final_tracked);
    lost_stracks_ = std::move(final_lost);

    // Remove duplicates within tracked set (solve the problem of the same target being tracked by two IDs)
    removeDuplicateWithinTracked(tracked_stracks_);
    
    // Update ROI memory
    if (roi_manager_) {
        roi_manager_->updateROITrackMemories(tracked_stracks_, frame_id_);
    }
    
    // Only return confirmed tracks
    std::vector<std::unique_ptr<STrack>> output_stracks;
    for (auto& track : tracked_stracks_) {
        bool should_keep = false;
        
        if (track->is_activated && track->is_confirmed) {
            if (track->tracklet_len > 1) {
                should_keep = true;
            }
        } else if (track->is_activated && !track->is_confirmed) {
            if (track->tracklet_len > 3) {
                should_keep = true;
            }
        }
        
        if (should_keep) {
            output_stracks.push_back(std::make_unique<STrack>(*track));
        }
    }
    
    return output_stracks;
}

std::vector<std::unique_ptr<STrack>> TPTrack::initTrack(const std::vector<Detection>& detections) {
    std::vector<std::unique_ptr<STrack>> stracks;
    
    for (const auto& det : detections) {
        auto track = std::make_unique<STrack>(det.bbox, det.confidence, det.class_id);
        
        // if detection contains appearance feature, set it to track
        if (!det.appearance.empty()) {
            track->setAppearance(det.appearance);
        }
        
        stracks.push_back(std::move(track));
    }
    
    return stracks;
}





cv::Mat TPTrack::calculateIoUMatrix(const std::vector<STrack*>& tracks,
                                               const std::vector<STrack*>& detections) {
    // check if input is empty
    if (tracks.empty() || detections.empty()) {
        return cv::Mat();  // return empty matrix
    }
    
    cv::Mat iou_matrix(tracks.size(), detections.size(), CV_32F);
    
    for (size_t i = 0; i < tracks.size(); ++i) {
        for (size_t j = 0; j < detections.size(); ++j) {
            if (tracks[i] && detections[j]) {  
                float iou = calculateIoU(tracks[i]->tlwh, detections[j]->tlwh);
                iou_matrix.at<float>(i, j) = 1.0f - iou; 
            } else {
                iou_matrix.at<float>(i, j) = 1.0f;  
            }
        }
    }
    return iou_matrix;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
TPTrack::hungarianAssignment(const cv::Mat& cost_matrix, float thresh) {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks, unmatched_dets;
    
    if (cost_matrix.empty() || cost_matrix.rows == 0 || cost_matrix.cols == 0) {
        for (int i = 0; i < cost_matrix.rows; ++i) {
            unmatched_tracks.push_back(i);
        }
        for (int j = 0; j < cost_matrix.cols; ++j) {
            unmatched_dets.push_back(j);
        }
        return {matches, unmatched_tracks, unmatched_dets};
    }
    
    // create the matrix required for Hungarian algorithm
    cv::Mat cost_matrix_copy = cost_matrix.clone();
    
    // ensure the matrix is square, if not, pad it
    int rows = cost_matrix_copy.rows;
    int cols = cost_matrix_copy.cols;
    int size = std::max(rows, cols);
    
    if (rows != cols) {
        cv::Mat padded_cost(size, size, CV_32F, cv::Scalar(1000.0)); // 大值表示不匹配
        cv::Rect roi(0, 0, cols, rows);
        cost_matrix_copy.copyTo(padded_cost(roi));
        cost_matrix_copy = padded_cost;
    }
    
    // apply Hungarian algorithm
    std::vector<int> assignment(size, -1);
    
    cv::Mat cost_8u;
    cost_matrix_copy.convertTo(cost_8u, CV_8U, 255);

    cv::Mat assignment_mat;
    float min_cost = tracking::Hungarian(cost_8u, assignment_mat);
    
    // process the matching results
    for (int i = 0; i < rows; i++) {
        int j = -1;
        for (int k = 0; k < assignment_mat.cols; k++) {
            if (assignment_mat.at<unsigned char>(i, k) == 1) {
                j = k;
                break;
            }
        }
        
        if (j >= 0 && j < cols && cost_matrix.at<float>(i, j) <= thresh) {
            matches.emplace_back(i, j);
        } else {
            unmatched_tracks.push_back(i);
        }
    }
    
    // determine unmatched detections
    for (int j = 0; j < cols; j++) {
        bool matched = false;
        for (const auto& [track_idx, det_idx] : matches) {
            if (det_idx == j) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            unmatched_dets.push_back(j);
        }
    }
    
    return {matches, unmatched_tracks, unmatched_dets};
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
TPTrack::linearAssignment(const cv::Mat& cost_matrix, float thresh) {
    // directly use Hungarian algorithm
    return hungarianAssignment(cost_matrix, thresh);
}

void TPTrack::multiPredict(const std::vector<STrack*>& stracks) {
    for (auto* track : stracks) {
        track->predict();
    }
}

std::vector<std::unique_ptr<STrack>> TPTrack::jointSTracks(
    const std::vector<std::unique_ptr<STrack>>& tlista,
    const std::vector<std::unique_ptr<STrack>>& tlistb) {
    std::set<int> exists;
    std::vector<std::unique_ptr<STrack>> result;

    for (const auto& track : tlista) {
        exists.insert(track->displayId());
        result.push_back(std::make_unique<STrack>(*track));
    }
    for (const auto& track : tlistb) {
        if (exists.find(track->displayId()) == exists.end()) {
            result.push_back(std::make_unique<STrack>(*track));
        }
    }
    return result;
}

std::vector<std::unique_ptr<STrack>> TPTrack::subSTracks(
    const std::vector<std::unique_ptr<STrack>>& tlista,
    const std::vector<std::unique_ptr<STrack>>& tlistb) {
    
    std::set<int> remove_ids;
    for (const auto& track : tlistb) {
        remove_ids.insert(track->displayId());
    }
    
    std::vector<std::unique_ptr<STrack>> result;
    for (const auto& track : tlista) {
        if (remove_ids.find(track->displayId()) == remove_ids.end()) {
            result.push_back(std::make_unique<STrack>(*track));
        }
    }
    
    return result;
}

std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>>
TPTrack::removeDuplicateSTracks(std::vector<std::unique_ptr<STrack>>& stracksa,
                                           std::vector<std::unique_ptr<STrack>>& stracksb) {
    if (stracksa.empty() || stracksb.empty()) {
        return {std::move(stracksa), std::move(stracksb)};
    }
    
    // convert to pointer vector to calculate IoU
    std::vector<STrack*> ptra, ptrb;
    for (auto& track : stracksa) ptra.push_back(track.get());
    for (auto& track : stracksb) ptrb.push_back(track.get());
    
    cv::Mat pdist = calculateIoUMatrix(ptra, ptrb);
    
    std::set<int> dupa, dupb;
    for (int i = 0; i < pdist.rows; ++i) {
        for (int j = 0; j < pdist.cols; ++j) {
           
            if (pdist.at<float>(i, j) < 0.15f) { // IoU > 0.85
                float distance = pdist.at<float>(i, j);
                float iou = 1.0f - distance;
                
                float trajectory_similarity = stracksa[i]->calculateTrajectorySimilarity(stracksb[j].get(), 5);
                float center_distance = cv::norm(stracksa[i]->center() - stracksb[j]->center());
                
                // LOG_INFO("detected duplicate track: stracksa[" << i << "] ID-" << stracksa[i]->displayId() 
                //           << " 与 stracksb[" << j << "] ID-" << stracksb[j]->displayId() 
                //           << " distance: " << distance << " IoU: " << iou 
                //           << " trajectory similarity: " << trajectory_similarity 
                //           << " center distance: " << center_distance
                //           << " trajectory length difference: " << std::abs((stracksa[i]->frame_id - stracksa[i]->start_frame) - 
                //                                            (stracksb[j]->frame_id - stracksb[j]->start_frame))
                //           << " if different ID: " << (stracksa[i]->displayId() != stracksb[j]->displayId()));
                
                // improved logic: combine IoU and trajectory similarity
                bool is_high_iou = iou > 0.85f;
                bool is_high_trajectory_similarity = trajectory_similarity > 0.8f;  
                bool is_close_center = center_distance < 30.0f;  
                
                // check if it is a different ID track
                bool is_different_ids = stracksa[i]->displayId() != stracksb[j]->displayId();
                
                // check if it is an ID switch problem caused by position overlap
                bool is_position_overlap = center_distance < 50.0f && iou > 0.7f; 
                bool is_same_roi = stracksa[i]->roi_id == stracksb[j]->roi_id;  
                
                // only when IoU is high and trajectory similarity is also high, it is considered a real duplicate track
                // check trajectory length difference to avoid deleting long-term tracks
                bool is_similar_length = std::abs((stracksa[i]->frame_id - stracksa[i]->start_frame) - 
                                                 (stracksb[j]->frame_id - stracksb[j]->start_frame)) < 10;
                
                // for different ID tracks, more strict conditions are needed
                if (is_high_iou && (is_high_trajectory_similarity || is_close_center) && 
                    (is_similar_length || !is_different_ids)) {
                    
                    // check if it is an ID switch problem caused by position overlap
                    if (is_position_overlap && is_same_roi && is_different_ids) {
                        LOG_INFO("detected ID switch problem caused by position overlap: ID-" << stracksa[i]->displayId() 
                                  << " and ID-" << stracksb[j]->displayId() 
                                  << " center distance: " << center_distance << " IoU: " << iou);
                        
                        // keep the track with longer length
                        if ((stracksa[i]->frame_id - stracksa[i]->start_frame) > 
                            (stracksb[j]->frame_id - stracksb[j]->start_frame)) {
                            // keep stracksa[i] ID, remove stracksb[j]
                            dupb.insert(j);
                            LOG_INFO("keep longer track ID-" << stracksa[i]->displayId() 
                                      << " remove shorter track ID-" << stracksb[j]->displayId());
                        } else {
                            // keep stracksb[j] ID, remove stracksa[i]
                            dupa.insert(i);
                            LOG_INFO("keep longer track ID-" << stracksb[j]->displayId() 
                                      << " remove shorter track ID-" << stracksa[i]->displayId());
                        }
                        continue;  // skip the subsequent duplicate processing logic
                    }
                    
                // improved duplicate logic: keep confirmed long-term tracks
                bool a_is_confirmed_long = stracksa[i]->is_confirmed && 
                    (stracksa[i]->frame_id - stracksa[i]->start_frame) > 50;  
                bool b_is_confirmed_long = stracksb[j]->is_confirmed && 
                    (stracksb[j]->frame_id - stracksb[j]->start_frame) > 50;  
                
                // check if it is the same track recovery (through ID and position)
                bool is_same_track_recovery = false;
                if (stracksa[i]->displayId() == stracksb[j]->displayId()) {
                    // if it is the same ID, check if the position is close
                    float pos_distance = cv::norm(stracksa[i]->center() - stracksb[j]->center());
                    is_same_track_recovery = pos_distance < 100.0f;  
                    // LOG_INFO("detected same ID track recovery: ID-" << stracksa[i]->displayId() 
                    //           << " position distance: " << pos_distance);
                }
                
                if (is_same_track_recovery) {
                    // if it is the same track recovery, keep the longer track
                    if ((stracksa[i]->frame_id - stracksa[i]->start_frame) > 
                        (stracksb[j]->frame_id - stracksb[j]->start_frame)) {
                        dupb.insert(j);
                        // LOG_INFO("remove duplicate short track stracksb[" << j << "] ID-" << stracksb[j]->displayId());
                    } else {
                        dupa.insert(i);
                        // LOG_INFO("remove duplicate short track stracksa[" << i << "] ID-" << stracksa[i]->displayId());
                    }
                } else if (a_is_confirmed_long && !b_is_confirmed_long) {
                    dupb.insert(j);
                    // LOG_INFO("remove short track stracksb[" << j << "] ID-" << stracksb[j]->displayId() 
                    //           << " keep long track stracksa[" << i << "] ID-" << stracksa[i]->displayId());
                } else if (b_is_confirmed_long && !a_is_confirmed_long) {
                    // keep confirmed long-term track b, remove short-term track a
                    dupa.insert(i);
                    // LOG_INFO("remove short track stracksa[" << i << "] ID-" << stracksa[i]->displayId() 
                    //           << " keep long track stracksb[" << j << "] ID-" << stracksb[j]->displayId());
                } else {
                    // if both are long-term tracks or both are short-term tracks, use the original logic
                    if ((stracksa[i]->frame_id - stracksa[i]->start_frame) > 
                        (stracksb[j]->frame_id - stracksb[j]->start_frame)) {
                        dupb.insert(j);
                        // LOG_INFO("remove stracksb[" << j << "] ID-" << stracksb[j]->displayId());
                    } else {
                        dupa.insert(i);
                        // LOG_INFO("remove stracksa[" << i << "] ID-" << stracksa[i]->displayId());
                    }
                }
                } // end if (is_high_iou && ...)
            }
        }
    }
    
    std::vector<std::unique_ptr<STrack>> result_a, result_b;
    for (size_t i = 0; i < stracksa.size(); ++i) {
        if (dupa.find(i) == dupa.end()) {
            result_a.push_back(std::move(stracksa[i]));
        }
    }
    for (size_t i = 0; i < stracksb.size(); ++i) {
        if (dupb.find(i) == dupb.end()) {
            result_b.push_back(std::move(stracksb[i]));
        }
    }
    
    return {std::move(result_a), std::move(result_b)};
}

float TPTrack::calculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;
    
    return intersection / union_area;
}

void TPTrack::associateDetectionsToTracks(
    const std::vector<STrack*>& tracks,
    const std::vector<STrack*>& detections,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_dets,
    float thresh) {

    cv::Mat iou_matrix = calculateIoUMatrix(tracks, detections);
    auto [m, ut, ud] = hungarianAssignment(iou_matrix, thresh);
    matches = m;
    unmatched_tracks = ut;
    unmatched_dets = ud;
}


// calculate motion feature cost
float TPTrack::calculateMotionCost(const STrack* track, const STrack* detection) {
    // if the track history information is insufficient, return the default cost
    if (track->velocity_history.size() < 2) {
        return 0.2f; // medium cost, not completely prevent matching
    }
    
    // 1. calculate the average history speed (recent frames)
    cv::Point2f avg_velocity(0.0f, 0.0f);
    float avg_speed = 0.0f;
    float avg_direction = 0.0f;
    
    int history_frames = std::min(static_cast<int>(track->velocity_history.size()), 5);
    for (int i = track->velocity_history.size() - history_frames; i < track->velocity_history.size(); ++i) {
        avg_velocity += track->velocity_history[i];
        avg_speed += track->speed_history[i];
    }
    avg_velocity /= history_frames;
    avg_speed /= history_frames;
    
    // calculate the average direction
    float sin_sum = 0.0f, cos_sum = 0.0f;
    for (int i = track->direction_history.size() - history_frames; i < track->direction_history.size(); ++i) {
        sin_sum += std::sin(track->direction_history[i]);
        cos_sum += std::cos(track->direction_history[i]);
    }
    avg_direction = std::atan2(sin_sum / history_frames, cos_sum / history_frames);
    
    // 2. calculate the current expected speed (the motion from the track prediction center to the detection center)
    cv::Point2f track_center = track->center(); // use the current position
    cv::Point2f detection_center = detection->center();
    cv::Point2f expected_velocity = detection_center - track_center;
    float expected_speed = cv::norm(expected_velocity);
    float expected_direction = std::atan2(expected_velocity.y, expected_velocity.x);
    
    // 3. speed consistency check
    float speed_diff = std::abs(expected_speed - avg_speed);
    float max_reasonable_speed = std::max(50.0f, avg_speed * 2.0f); // maximum reasonable speed change
    float speed_cost = std::min(1.0f, speed_diff / max_reasonable_speed);
    
    // 4. direction consistency check
    float direction_diff = std::abs(expected_direction - avg_direction);
    // handle the angle loop (-π to π)
    if (direction_diff > M_PI) {
        direction_diff = 2 * M_PI - direction_diff;
    }
    // allow the maximum direction change (e.g. 90 degrees = π/2)
    float max_direction_change = M_PI / 2.0f; // 90 degrees
    float direction_cost = std::min(1.0f, direction_diff / max_direction_change);
    
    // 5. acceleration constraint check
    float acceleration_cost = 0.0f;
    if (track->velocity_history.size() >= 2) {
        cv::Point2f recent_velocity = track->velocity_history.back();
        cv::Point2f acceleration = expected_velocity - recent_velocity;
        float acceleration_magnitude = cv::norm(acceleration);
        
        // maximum reasonable acceleration (pixels/frame²)
        float max_acceleration = 30.0f;
        acceleration_cost = std::min(1.0f, acceleration_magnitude / max_acceleration);
    }
    
    // 6. speed stability weight adjustment
    float stability_weight = 1.0f;
    if (track->speed_history.size() >= 3) {
        // calculate the speed stability
        float speed_variance = 0.0f;
        for (float speed : track->speed_history) {
            float diff = speed - avg_speed;
            speed_variance += diff * diff;
        }
        speed_variance /= track->speed_history.size();
        float speed_std = std::sqrt(speed_variance);
        
        // if the track speed is very stable, strictly constrain the motion
        if (speed_std < avg_speed * 0.3f && avg_speed > 5.0f) {
            stability_weight = 1.5f; // increase the motion constraint weight
        }
    }
    
    // 7. comprehensive motion cost
    float motion_cost = 0.4f * speed_cost + 0.4f * direction_cost + 0.2f * acceleration_cost;
    motion_cost = std::min(1.0f, motion_cost * stability_weight);
    
    // 8. special case handling: if the track is almost stationary recently, reduce the motion constraint
    if (avg_speed < 3.0f) {
        motion_cost *= 0.5f; // for almost stationary targets, reduce the motion constraint
    }
    
    return motion_cost;
}

// save the current tracking state
void TPTrack::saveTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks) {
    saved_tracks.clear();
    
    // only save the activated tracks
    for (const auto& track : tracked_stracks_) {
        if (track->is_activated) {
            saved_tracks.push_back(std::make_unique<STrack>(*track));
        }
    }
}

// restore the tracker from the saved state
void TPTrack::restoreTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks) {
    // clear the current tracking state
    tracked_stracks_.clear();
    lost_stracks_.clear();
    
    // restore the saved tracks
    for (auto& track : saved_tracks) {
        if (track->is_activated) {
            tracked_stracks_.push_back(std::make_unique<STrack>(*track));
        }
    }
    
    // keep the current frame ID unchanged
    // update the frame ID of all restored tracks to match the current frame
    for (auto& track : tracked_stracks_) {
        track->frame_id = frame_id_;
    }
}

// state transfer when mode switching
void TPTrack::transferState(TPTrack& other_tracker) {
    // temporarily store the tracking state
    std::vector<std::unique_ptr<STrack>> temp_tracks;
    
    // save the current tracking state
    saveTrackerState(temp_tracks);
    
    // get the frame ID of the other tracker
    int other_frame_id = other_tracker.getFrameId();
    
    // restore to another tracker
    other_tracker.restoreTrackerState(temp_tracks);
    
    // synchronize the frame ID to maintain continuity
    if (other_frame_id > 0) {
        other_tracker.frame_id_ = other_frame_id;
    } else {
        other_tracker.frame_id_ = frame_id_;
    }
}

// GMM memory recovery implementation
std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>> 
TPTrack::processGMMMemoryRecovery(
    std::vector<std::unique_ptr<STrack>>& detections,
    const std::unordered_map<int, std::vector<Detection>>& roi_detections,
    const std::vector<std::unique_ptr<STrack>>& current_frame_lost_stracks) {
    std::vector<std::unique_ptr<STrack>> recovered_stracks;
    std::vector<std::unique_ptr<STrack>> remaining_detections;

    // 1. Fit ROI-Aware GMM for all lost tracks (history + current frame)
    std::vector<STrack*> all_lost_tracks;
    for (auto& t : lost_stracks_) all_lost_tracks.push_back(t.get());
    for (auto& t : current_frame_lost_stracks) all_lost_tracks.push_back(t.get());
    
    struct SimplifiedTrackRecoveryInfo {
        cv::Ptr<cv::ml::EM> gmm;           // GMM model (motion + spatial + history + ROI)
        cv::Rect roi_bbox;                 // ROI bbox
        cv::Point2f last_position;
        cv::Point2f predicted_position;
        cv::Point2f last_velocity;        // last velocity
        float last_direction;              // last movement direction
        int lost_frames;
        int roi_id;
        float gmm_threshold;               // GMM probability threshold
        bool has_valid_roi;
    };
    
    // Helper function to setup recovery track properties
    auto setupRecoveryTrack = [&](std::unique_ptr<STrack>& detection, int track_id, float confidence) {
        detection->permanent_id = track_id;
        detection->is_confirmed = true;
        detection->is_real_target = true;
        detection->track_id = track_id;
        detection->recovery_confidence = confidence;
        detection->is_recovered = true;
        detection->is_activated = true;
        detection->state = STrack::Tracked;
        detection->tracklet_len = 1;
        detection->confirmation_frames = 3;
        detection->activate(frame_id_);
    };
    
    // Helper function to check position constraint with multi-level strategy
    auto checkPositionConstraint = [](const cv::Point2f& det_center, 
                                     const SimplifiedTrackRecoveryInfo& info) -> bool {
        float velocity_magnitude = cv::norm(info.last_velocity);
        float distance = cv::norm(det_center - info.predicted_position);
        float base_threshold = std::max(30.0f, velocity_magnitude * 3.0f + info.lost_frames * 10.0f);
        
        // Strategy 1: Basic distance matching (relatively lenient)
        if (distance <= base_threshold) return true;
        
        // Strategy 2: Medium distance matching (requires velocity direction consistency)
        if (distance <= base_threshold * 1.5f && velocity_magnitude > 1.0f) {
            cv::Point2f direction_to_detection = det_center - info.predicted_position;
            float direction_dot = direction_to_detection.dot(info.last_velocity);
            if (direction_dot > 0) {
                float direction_similarity = direction_dot / (cv::norm(direction_to_detection) * velocity_magnitude);
                if (direction_similarity > 0.5f) return true;
            }
        }
        
        // Strategy 3: Large distance matching (only for high-speed long trajectory targets)
        if (distance <= base_threshold * 2.0f && velocity_magnitude > 10.0f && info.lost_frames <= 2) {
            cv::Point2f direction_to_detection = det_center - info.predicted_position;
            float direction_dot = direction_to_detection.dot(info.last_velocity);
            if (direction_dot > 0) {
                float direction_similarity = direction_dot / (cv::norm(direction_to_detection) * velocity_magnitude);
                if (direction_similarity > 0.7f) return true;
            }
        }
        
        // Strategy 4: Close distance matching (for very close detections, accept even if prediction is inaccurate)
        if (distance <= 25.0f) return true;
        
        return false;
    };
    
    std::unordered_map<int, SimplifiedTrackRecoveryInfo> track_recovery_info;
    
    // Build recovery info for all lost tracks
    for (auto* lost_track : all_lost_tracks) {
        if (lost_track->position_history.size() >= 3 && 
            lost_track->velocity_history.size() >= 2 &&
            lost_track->direction_history.size() >= 2) {
            
            SimplifiedTrackRecoveryInfo info;
            info.last_position = lost_track->position_history.back();
            info.lost_frames = frame_id_ - lost_track->frame_id;
            info.roi_id = lost_track->roi_id;
            info.has_valid_roi = (info.roi_id > 0 && roi_manager_);
            
            // Get the last movement information
            info.last_velocity = lost_track->velocity_history.back();
            info.last_direction = lost_track->direction_history.back();
            
            // Get the ROI bbox
            if (info.has_valid_roi) {
                const ROI* roi = roi_manager_->getROI(info.roi_id);
                if (roi) {
                    info.roi_bbox = roi->bbox;
                } else {
                    info.has_valid_roi = false;
                    info.roi_bbox = cv::Rect(0, 0, 1920, 1080);
                }
            } else {
                info.roi_bbox = cv::Rect(0, 0, 1920, 1080);
            }
            
            // Simplified predicted position calculation: use the motion model prediction
            cv::Point2f predicted_velocity;
            if (info.lost_frames <= 5) {
                // Short-term lost: use the last known velocity
                predicted_velocity = info.last_velocity;
            } else {
                // Long-term lost: use the history speed trend prediction
                if (lost_track->velocity_history.size() >= 3) {
                    // Calculate the speed trend (the average acceleration of the last 3 frames)
                    cv::Point2f velocity_trend(0, 0);
                    for (size_t i = lost_track->velocity_history.size() - 3; 
                         i < lost_track->velocity_history.size() - 1; ++i) {
                        velocity_trend += (lost_track->velocity_history[i+1] - lost_track->velocity_history[i]);
                    }
                    velocity_trend /= 2.0f; 
                    
                    // Predict the current velocity based on the trend
                    predicted_velocity = info.last_velocity + velocity_trend * info.lost_frames;
                } else {
                    predicted_velocity = info.last_velocity;
                }
            }
            
            // Predict the position
            info.predicted_position = info.last_position + predicted_velocity * info.lost_frames;
            
            // Set the GMM threshold: only adjust based on the ROI state
            info.gmm_threshold = info.has_valid_roi ? 1e-4f : 1e-5f;
            
            // Train the GMM model
            info.gmm = fitGMM(
                lost_track->position_history,
                lost_track->velocity_history,
                lost_track->direction_history,
                lost_track->speed_history,
                lost_track->frame_history,
                info.roi_bbox,
                frame_id_,
                2  // Fixed using 2 components
            );
            
            track_recovery_info[lost_track->displayId()] = info;
        }
    }

    // 2. Process each detection for GMM matching (simplified version)
    for (auto& detection : detections) {
        cv::Point2f det_center = detection->center();
        int detection_roi_id = detection->roi_id;
        
        int best_track_id = -1;
        float best_gmm_prob = 0.0f;
        
        for (const auto& [track_id, info] : track_recovery_info) {
            // Step 1: ROI consistency check (the only constraint item)
            bool roi_compatible = false;
            if (info.has_valid_roi && detection_roi_id > 0) {
                // Both have ROI info, must match
                roi_compatible = (detection_roi_id == info.roi_id);
            } else if (info.has_valid_roi && detection_roi_id <= 0) {
                // Track has ROI, detection has no ROI, check if within ROI range
                roi_compatible = info.roi_bbox.contains(cv::Point(det_center.x, det_center.y));
            } else if (!info.has_valid_roi && detection_roi_id > 0) {
                // Track has no ROI, detection has ROI, allow matching
                roi_compatible = true;
            } else {
                // Neither has ROI info, allow matching
                roi_compatible = true;
            }
            
            if (!roi_compatible) continue;
            
            // Step 2: Position constraint check using multi-level strategy
            if (!checkPositionConstraint(det_center, info)) continue;
            
            // Step 3: Calculate enhanced GMM probability (core matching metric)
            float gmm_prob = 0.0f;
            if (info.gmm) {
                // Calculate current detection's motion information
                cv::Point2f current_velocity;
                float current_direction;
                
                if (info.lost_frames <= 5) {
                    // Short-term lost: use predicted motion information
                    current_velocity = info.last_velocity;
                    current_direction = info.last_direction;
                } else {
                    // Long-term lost: estimate motion based on detection position vs predicted position difference
                    cv::Point2f position_diff = det_center - info.predicted_position;
                    current_velocity = position_diff / std::max(1.0f, static_cast<float>(info.lost_frames));
                    current_direction = std::atan2(current_velocity.y, current_velocity.x);
                }
                
                float current_speed = cv::norm(current_velocity);
                
                // Calculate the GMM probability
                gmm_prob = GMMProbability(
                    info.gmm, 
                    det_center, 
                    current_velocity,
                    current_direction,
                    current_speed,
                    info.roi_bbox
                );
            }
            
            // Step 4: Simple threshold judgment (remove complex fusion scoring)
            if (gmm_prob > info.gmm_threshold && gmm_prob > best_gmm_prob) {
                best_gmm_prob = gmm_prob;
                best_track_id = track_id;
            }
        }
        
        // Recover the best matching track
        if (best_track_id != -1) {
            LOG_INFO("✅ Simplified GMM memory recovery: Detection(" << det_center.x << "," << det_center.y 
                     << ") matches Track ID-" << best_track_id << " GMM prob:" << best_gmm_prob);
            
            // Search and recover track
            bool found_and_recovered = false;
            auto it = lost_stracks_.begin();
            while (it != lost_stracks_.end()) {
                if ((*it)->displayId() == best_track_id) {
                    // Fix: save necessary info before deletion to avoid accessing deleted objects
                    int track_id = (*it)->displayId();
                    
                    setupRecoveryTrack(detection, best_track_id, best_gmm_prob);
                    recovered_stracks.push_back(std::move(detection));
                    found_and_recovered = true;
                    
                    // Fix: mark for deletion instead of immediate deletion
                    (*it)->markForDeletion();
                    
                    LOG_INFO("✅ Recovered from historical lost tracks ID-" << track_id);
                    break;
                } else {
                    ++it;
                }
            }
            
            // If not found in historical lost tracks, check current frame lost tracks
            if (!found_and_recovered) {
                for (auto& lost_track : current_frame_lost_stracks) {
                    if (lost_track->displayId() == best_track_id) {
                        setupRecoveryTrack(detection, best_track_id, best_gmm_prob);
                        recovered_stracks.push_back(std::move(detection));
                        found_and_recovered = true;
                        LOG_INFO("✅ Recovered from current frame lost tracks ID-" << best_track_id);
                        break;
                    }
                }
            }
            
            if (!found_and_recovered) {
                remaining_detections.push_back(std::move(detection));
            }
        } else {
            remaining_detections.push_back(std::move(detection));
        }
    }

    return {std::move(remaining_detections), std::move(recovered_stracks)};
}

// Extract trajectory features
TrajectoryFeatures TPTrack::extractTrajectoryFeatures(const STrack& track, 
                                                                 const std::vector<STrack*>& all_tracks) {
    TrajectoryFeatures features;
    
    // Extract position history
    features.position_history = track.position_history;
    
    // If position history is empty, add current position (for detection objects)
    if (features.position_history.empty()) {
        features.position_history.push_back(track.center());
    }
    
    // Limit history length
    if (features.position_history.size() > HISTORY_FRAMES) {
        features.position_history.erase(features.position_history.begin(), 
                                     features.position_history.end() - HISTORY_FRAMES);
    }
    
    // Calculate velocity history
    features.velocity_history.clear();
    for (size_t i = 1; i < features.position_history.size(); ++i) {
        cv::Point2f velocity = features.position_history[i] - features.position_history[i-1];
        features.velocity_history.push_back(velocity);
    }
    
    // Calculate interaction features
    features.relative_positions = calculateInteractionFeatures(track, all_tracks);
    
    // Other features
    features.avg_confidence = track.score;
    features.track_length = track.tracklet_len;
    features.bbox_area = track.tlwh.width * track.tlwh.height;
    
    return features;
}

// Calculate interaction features - only calculate relative positions with other detection targets
std::vector<cv::Point2f> TPTrack::calculateInteractionFeatures(const STrack& track,
                                                                         const std::vector<STrack*>& all_tracks) {
    std::vector<cv::Point2f> relative_positions;
    cv::Point2f track_center = track.center();
    
    LOG_INFO("        🔍 Calculate interaction features:");
    LOG_INFO("          Target center: (" << track_center.x << ", " << track_center.y << ")");
    LOG_INFO("          Total tracks count: " << all_tracks.size());
    
    int valid_interactions = 0;
    for (const auto* other_track : all_tracks) {
        if (other_track == &track) continue;
        
        cv::Point2f other_center = other_track->center();
        cv::Point2f relative_pos = other_center - track_center;
        
        // Calculate distance
        float distance = cv::norm(relative_pos);
        if (distance > 0) {
            // Normalize relative position
            relative_pos = relative_pos / distance;
        relative_positions.push_back(relative_pos);
            valid_interactions++;
            
            LOG_INFO("          Relative position to other target: (" << relative_pos.x << ", " << relative_pos.y << ") distance: " << distance);
        }
    }
    
    LOG_INFO("          Valid interactions count: " << valid_interactions);
    return relative_positions;
}

// Build three-stage memory recovery model
ThreeStageRecoveryModel TPTrack::buildThreeStageRecoveryModel(const TrajectoryFeatures& lost_features,
                                                   const TrajectoryFeatures& detection_features) {
    ThreeStageRecoveryModel model;
    
    // First stage: build motion prediction model
    if (!lost_features.position_history.empty() && !lost_features.velocity_history.empty()) {
        cv::Point2f last_position = lost_features.position_history.back();
        
        // Improved motion prediction method
        cv::Point2f predicted_position = last_position;
        cv::Point2f avg_velocity(0, 0);
        cv::Point2f avg_acceleration(0, 0);
    
        // 1. Calculate weighted average speed (recent frames have higher weight)
        float total_weight = 0.0f;
        for (size_t i = 0; i < lost_features.velocity_history.size(); ++i) {
            float weight = static_cast<float>(i + 1); // Frames closer have higher weight
            avg_velocity += lost_features.velocity_history[i] * weight;
            total_weight += weight;
        }
        if (total_weight > 0) {
            avg_velocity = avg_velocity * (1.0f / total_weight);
        }
        
        // 2. Calculate trend acceleration (based on recent speed changes)
        if (lost_features.velocity_history.size() >= 3) {
            size_t recent_count = std::min(static_cast<size_t>(3), lost_features.velocity_history.size());
            cv::Point2f recent_avg_vel(0, 0);
            cv::Point2f older_avg_vel(0, 0);
            
            // Average speed of recent frames
            for (size_t i = lost_features.velocity_history.size() - recent_count; i < lost_features.velocity_history.size(); ++i) {
                recent_avg_vel += lost_features.velocity_history[i];
            }
            recent_avg_vel = recent_avg_vel * (1.0f / recent_count);
            
            // Average speed of previous frames
            size_t older_start = lost_features.velocity_history.size() - 2 * recent_count;
            if (older_start < lost_features.velocity_history.size()) {
                for (size_t i = older_start; i < lost_features.velocity_history.size() - recent_count; ++i) {
                    older_avg_vel += lost_features.velocity_history[i];
                }
                older_avg_vel = older_avg_vel * (1.0f / recent_count);
                
                // Calculate acceleration trend
                avg_acceleration = (recent_avg_vel - older_avg_vel) * (1.0f / recent_count);
            }
        }
        
        // 3. Adaptive prediction frame count (based on track length and speed stability)
        int lost_frames = 1; // Default to 1 frame
        if (lost_features.track_length > 5) {
            // Calculate speed stability
            float velocity_std = 0.0f;
            for (const auto& vel : lost_features.velocity_history) {
                float diff = cv::norm(vel - avg_velocity);
                velocity_std += diff * diff;
            }
            velocity_std = std::sqrt(velocity_std / lost_features.velocity_history.size());
            
            // The more stable the speed, the more frames are predicted
            float velocity_magnitude = cv::norm(avg_velocity);
            if (velocity_magnitude > 0 && velocity_std < velocity_magnitude * 0.5f) {
                lost_frames = std::min(5, static_cast<int>(velocity_magnitude / 10.0f));
        }
    }
    
        // 4. Predict position (using improved model)
        predicted_position = last_position + avg_velocity * lost_frames + 
                           0.5f * avg_acceleration * lost_frames * lost_frames;
        
        model.motion.predicted_position = predicted_position;
        model.motion.avg_velocity = avg_velocity;
        model.motion.avg_acceleration = avg_acceleration;
        model.motion.lost_frames = lost_frames;
        
        // 5. Improved prediction radius calculation
        float velocity_magnitude = cv::norm(avg_velocity);
        
        // Calculate base radius based on speed and prediction frame count
        float prediction_radius = 20.0f; // Reduce base radius
        
        // Adjust based on speed (faster speeds result in larger prediction error)
        prediction_radius += velocity_magnitude * 1.5f;
        
        // Adjust based on prediction frame count (more frames result in larger error)
        prediction_radius += lost_frames * 5.0f;
        
        // Adjust based on track length (longer tracks result in more accurate prediction, smaller radius)
        if (lost_features.track_length > 10) {
            prediction_radius *= 0.7f;
        } else if (lost_features.track_length < 5) {
            prediction_radius *= 1.2f;
        }
        
        // Adjust based on speed stability
        float velocity_std = 0.0f;
        for (const auto& vel : lost_features.velocity_history) {
            float diff = cv::norm(vel - avg_velocity);
            velocity_std += diff * diff;
        }
        velocity_std = std::sqrt(velocity_std / lost_features.velocity_history.size());
        
        if (velocity_magnitude > 0) {
            float stability_ratio = velocity_std / velocity_magnitude;
            if (stability_ratio < 0.3f) {
                prediction_radius *= 0.6f; // Speed stable, further reduce radius
            } else if (stability_ratio > 0.7f) {
                prediction_radius *= 1.2f; // Speed unstable, increase radius
            }
        }
        
        // Ensure minimum and maximum radius
        prediction_radius = std::max(15.0f, std::min(150.0f, prediction_radius));
        
        model.motion.prediction_radius = prediction_radius;
        
        LOG_INFO("      🔧 First stage motion prediction model construction completed");
        LOG_INFO("        Predicted position: (" << model.motion.predicted_position.x << ", " << model.motion.predicted_position.y << ")");
        LOG_INFO("        Average velocity: (" << avg_velocity.x << ", " << avg_velocity.y << ")");
        LOG_INFO("        Prediction frames: " << lost_frames);
        LOG_INFO("        Prediction radius: " << model.motion.prediction_radius);
        LOG_INFO("        Prediction radius calculation details:");
        LOG_INFO("          Base radius: 20.0");
        LOG_INFO("          Velocity adjustment: +" << (velocity_magnitude * 1.5f) << " (velocity: " << velocity_magnitude << ")");
        LOG_INFO("          Frames adjustment: +" << (lost_frames * 5.0f) << " (frames: " << lost_frames << ")");
        LOG_INFO("          Track length adjustment: " << (lost_features.track_length > 10 ? "0.7x" : (lost_features.track_length < 5 ? "1.2x" : "1.0x")));
        LOG_INFO("          Velocity stability adjustment: " << (velocity_magnitude > 0 ? (velocity_std / velocity_magnitude < 0.3f ? "0.6x" : (velocity_std / velocity_magnitude > 0.7f ? "1.2x" : "1.0x")) : "1.0x"));
    }
    
    // Second stage: appearance feature model removed (not needed)
    LOG_INFO("      🔧 Second stage: appearance feature model skipped");
    
    // Third stage: build interaction feature model
    if (!lost_features.relative_positions.empty()) {
        model.interaction.relative_positions = lost_features.relative_positions;
        
        // Improved consistency calculation of interaction pattern
        float consistency_sum = 0.0f;
        int valid_patterns = 0;
        
        if (lost_features.relative_positions.size() > 1) {
            for (size_t i = 1; i < lost_features.relative_positions.size(); ++i) {
                cv::Point2f current_rel = lost_features.relative_positions[i];
                cv::Point2f prev_rel = lost_features.relative_positions[i-1];
                
                // Use Gaussian function to calculate similarity, avoiding negative values
                float distance = cv::norm(current_rel - prev_rel);
                float max_distance = std::max(cv::norm(current_rel), cv::norm(prev_rel));
                
                if (max_distance > 0) {
                    // Use Gaussian function, ensuring similarity is within 0-1 range
                    float pattern_similarity = std::exp(-distance * distance / (2.0f * 50.0f * 50.0f));
                    consistency_sum += pattern_similarity;
                    valid_patterns++;
                }
            }
        }
        
        // Ensure consistency is within a reasonable range
        model.interaction.interaction_consistency = valid_patterns > 0 ? 
            std::max(0.0f, std::min(1.0f, consistency_sum / valid_patterns)) : 0.5f;
        
        LOG_INFO("      🔧 Third stage interaction feature model construction completed");
        LOG_INFO("        Relative positions count: " << model.interaction.relative_positions.size());
        LOG_INFO("        Interaction consistency: " << model.interaction.interaction_consistency);
        LOG_INFO("        Valid patterns count: " << valid_patterns);
    } else {
        // If there is no relative position data, set default value
        model.interaction.interaction_consistency = 0.5f;
        LOG_INFO("      🔧 Third stage interaction feature model construction completed: no relative position data, using default value");
    }
    
    return model;
}

// First stage: motion prediction screening
bool TPTrack::stageOneMotionPrediction(const ThreeStageRecoveryModel& model,
                                                   const cv::Point2f& detection_position) {
    // Calculate distance from detection position to predicted position
    float distance = cv::norm(detection_position - model.motion.predicted_position);
    float velocity_magnitude = cv::norm(model.motion.avg_velocity);
    
    // Improved multi-level distance determination strategy
    bool in_prediction_range = false;
    std::string match_strategy = "";
    
    // Strategy 1: basic distance matching (relatively lenient)
    float base_threshold = std::max(30.0f, velocity_magnitude * 3.0f + model.motion.lost_frames * 10.0f);
    if (distance <= base_threshold) {
        in_prediction_range = true;
                    match_strategy = "Basic distance matching";
    }
    // Strategy 2: medium distance matching (requires speed direction consistency)
    else if (distance <= base_threshold * 1.5f && velocity_magnitude > 1.0f) {
        cv::Point2f direction_to_detection = detection_position - model.motion.predicted_position;
        float direction_dot = direction_to_detection.dot(model.motion.avg_velocity);
        
        // If direction is basically consistent (angle less than 60 degrees)
        if (direction_dot > 0) {
            float direction_similarity = direction_dot / (cv::norm(direction_to_detection) * velocity_magnitude);
            if (direction_similarity > 0.5f) {
                in_prediction_range = true;
                match_strategy = "Direction consistent matching";
            }
        }
    }
    // Strategy 3: large distance matching (only for high-speed long trajectory targets)
    else if (distance <= base_threshold * 2.0f && velocity_magnitude > 10.0f && model.motion.lost_frames <= 2) {
        cv::Point2f direction_to_detection = detection_position - model.motion.predicted_position;
        float direction_dot = direction_to_detection.dot(model.motion.avg_velocity);
        
        if (direction_dot > 0) {
            float direction_similarity = direction_dot / (cv::norm(direction_to_detection) * velocity_magnitude);
            if (direction_similarity > 0.7f) {
                in_prediction_range = true;
                match_strategy = "High-speed long trajectory matching";
            }
        }
    }
    // Strategy 4: close distance matching (for very close detections, accept even if prediction is inaccurate)
    else if (distance <= 25.0f) {
        in_prediction_range = true;
        match_strategy = "Close distance matching";
    }
    
    return in_prediction_range;
}

// Second stage: appearance feature matching (removed - not needed)
float TPTrack::stageTwoAppearanceMatching(const ThreeStageRecoveryModel& model,
                                                 const TrajectoryFeatures& detection_features) {
    // Appearance features removed - return default high similarity to skip this stage
    LOG_INFO("      🔧 Second stage: appearance feature matching skipped (features removed)");
    return 1.0f;  // Return high similarity to pass this stage
}

// Third stage: interaction feature verification - only calculate similarity between current frame and previous frame
float TPTrack::stageThreeInteractionVerification(const ThreeStageRecoveryModel& model,
                                                            const TrajectoryFeatures& detection_features,
                                                            const std::vector<STrack*>& all_tracks) {
    if (all_tracks.size() <= 1) {
        return 0.5f; // Default medium similarity
    }
    
    // Calculate relative position between current detection and other targets
    std::vector<cv::Point2f> current_relative_positions;
    cv::Point2f detection_center = detection_features.position_history.back();
    
    for (const auto* track : all_tracks) {
        if (track->center() != detection_center) { // Exclude self
            cv::Point2f rel_pos = track->center() - detection_center;
            current_relative_positions.push_back(rel_pos);
        }
    }
    
    if (current_relative_positions.empty()) {
        return 0.5f;
    }
    
    // Only calculate similarity between current frame and previous frame
    float interaction_similarity = 0.0f;
    int valid_comparisons = 0;
    
    // Get relative position of previous frame (if exists)
    if (!model.interaction.relative_positions.empty()) {
        // Ensure consistency in number, only compare interaction features of the same number
        size_t compare_count = std::min(model.interaction.relative_positions.size(), current_relative_positions.size());
        
        for (size_t i = 0; i < compare_count; ++i) {
            const auto& last_frame_rel = model.interaction.relative_positions[i];
            const auto& current_rel = current_relative_positions[i];
            
            // Calculate similarity of relative position between current frame and previous frame
            float distance = cv::norm(current_rel - last_frame_rel);
            float max_distance = std::max(cv::norm(current_rel), cv::norm(last_frame_rel));
            
            if (max_distance > 0) {
                // Use Gaussian function to calculate similarity
                float similarity = std::exp(-distance * distance / (2.0f * 100.0f * 100.0f));
                interaction_similarity += similarity;
                valid_comparisons++;
            
            }
        }
        
        if (valid_comparisons > 0) {
            interaction_similarity /= valid_comparisons;
        }
    }
    
    // If there is no previous frame data, use current interaction consistency as a fallback
    if (valid_comparisons == 0) {
        if (current_relative_positions.size() > 1) {
            float consistency_sum = 0.0f;
            int consistency_count = 0;
            
            // Calculate similarity between current relative positions
            for (size_t i = 0; i < current_relative_positions.size(); ++i) {
                for (size_t j = i + 1; j < current_relative_positions.size(); ++j) {
                    float distance = cv::norm(current_relative_positions[i] - current_relative_positions[j]);
                    float max_distance = std::max(cv::norm(current_relative_positions[i]), cv::norm(current_relative_positions[j]));
                    
                    if (max_distance > 0) {
                        float similarity = std::exp(-distance * distance / (2.0f * 50.0f * 50.0f));
                        consistency_sum += similarity;
                        consistency_count++;
                    }
                }
            }
            
            if (consistency_count > 0) {
                interaction_similarity = consistency_sum / consistency_count;
            }
        }
    }
    
    // Ensure score is within a reasonable range
    float final_interaction_score = std::max(0.0f, std::min(1.0f, interaction_similarity));
    
    return final_interaction_score;
}

// Calculate three-stage comprehensive similarity (appearance stage removed)
float TPTrack::calculateThreeStageSimilarity(const ThreeStageRecoveryModel& model,
                                                         const TrajectoryFeatures& lost_features,
                                                         const TrajectoryFeatures& detection_features,
                                                         const std::vector<STrack*>& all_tracks) {
    // First stage: motion prediction screening
    cv::Point2f detection_position = detection_features.position_history.back();
    bool motion_passed = stageOneMotionPrediction(model, detection_position);
    
    if (!motion_passed) {
        LOG_INFO("      ❌ First stage motion prediction failed, skip subsequent stages");
        return 0.0f;
    }
    
    // Second stage: appearance feature matching (removed - always pass)
    LOG_INFO("      ✅ Second stage: appearance feature matching skipped (always pass)");
    
    // Third stage: interaction feature verification
    float interaction_score = stageThreeInteractionVerification(model, detection_features, all_tracks);
    
    // Comprehensive calculation of final similarity (based on motion + interaction only)
    // Since appearance is removed, give more weight to interaction features
    float motion_confidence = 0.8f;  // High confidence if motion prediction passed
    float final_similarity = 0.6f * motion_confidence + 0.4f * interaction_score;
    
    return final_similarity;
}



// Calculate motion similarity
float TPTrack::calculateMotionSimilarity(const std::deque<cv::Point2f>& pos1,
                                                   const std::deque<cv::Point2f>& pos2) {
    if (pos1.empty() || pos2.empty()) return 0.0f;
    
    // Calculate DTW distance
    int n = pos1.size(), m = pos2.size();
    cv::Mat dtw_matrix = cv::Mat::zeros(n + 1, m + 1, CV_32F);
    
    // Initialize
    for (int i = 0; i <= n; ++i) dtw_matrix.at<float>(i, 0) = std::numeric_limits<float>::max();
    for (int j = 0; j <= m; ++j) dtw_matrix.at<float>(0, j) = std::numeric_limits<float>::max();
    dtw_matrix.at<float>(0, 0) = 0.0f;
    
    // Dynamic programming
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = cv::norm(pos1[i-1] - pos2[j-1]);
            dtw_matrix.at<float>(i, j) = cost + std::min({
                dtw_matrix.at<float>(i-1, j),
                dtw_matrix.at<float>(i, j-1),
                dtw_matrix.at<float>(i-1, j-1)
            });
        }
    }
    
    float dtw_distance = dtw_matrix.at<float>(n, m);
    return std::exp(-dtw_distance / 100.0f); // Normalize to [0,1]
}

// Calculate interaction similarity
float TPTrack::calculateInteractionSimilarity(const std::vector<cv::Point2f>& rel1,
                                                        const std::vector<cv::Point2f>& rel2) {
    if (rel1.empty() || rel2.empty()) return 0.0f;
    
    // Calculate similarity of relative position vectors
    float similarity = 0.0f;
    int min_size = std::min(rel1.size(), rel2.size());
    
    for (int i = 0; i < min_size; ++i) {
        float dot_product = rel1[i].x * rel2[i].x + rel1[i].y * rel2[i].y;
        float norm1 = cv::norm(rel1[i]);
        float norm2 = cv::norm(rel2[i]);
        
        if (norm1 > 0 && norm2 > 0) {
            similarity += dot_product / (norm1 * norm2);
        }
    }
    
    return similarity / min_size;
}

// New: process memory recovery for failed tracks
std::pair<std::vector<std::unique_ptr<STrack>>, std::vector<std::unique_ptr<STrack>>> 
TPTrack::processGMMMemoryRecoveryWithFailedTracks(
    std::vector<std::unique_ptr<STrack>>& detections,
    const std::vector<STrack*>& failed_tracks,
    const std::unordered_map<int, std::vector<Detection>>& roi_detections) {
    
    std::vector<std::unique_ptr<STrack>> recovered_stracks;
    std::vector<std::unique_ptr<STrack>> remaining_detections;

    // Collect all active tracks for interaction feature calculation
    std::vector<STrack*> all_tracks;
    for (auto& track : tracked_stracks_) {
        all_tracks.push_back(track.get());
    }
    for (auto& track : lost_stracks_) {
        all_tracks.push_back(track.get());
    }

    // Process each detection
    for (auto& detection : detections) {
        cv::Point2f det_center = detection->center();
        bool recovered = false;

        // Extract features of the detection
        TrajectoryFeatures detection_features = extractTrajectoryFeatures(*detection, all_tracks);
        
        LOG_INFO("🔥 Detection feature extraction completed:");
        LOG_INFO("   Position history count: " << detection_features.position_history.size());
        LOG_INFO("   Velocity history count: " << detection_features.velocity_history.size());
        LOG_INFO("   Relative positions count: " << detection_features.relative_positions.size());
        LOG_INFO("   Detection box area: " << detection_features.bbox_area);
        
        // Search for GMM matching in all lost tracks
        float best_similarity = 0.0f;
        int best_lost_track_id = -1;
        
        // Check historical lost tracks
        for (auto& lost_track : lost_stracks_) {
            // Check if lost time is reasonable (not more than 30 frames)
            if (frame_id_ - lost_track->frame_id > HISTORY_FRAMES) {
        
                continue;
            }
            
            // Extract features of the lost track
            TrajectoryFeatures lost_features = extractTrajectoryFeatures(*lost_track, all_tracks);
            

            LOG_INFO("    Lost track features:");
            LOG_INFO("      Position history count: " << lost_features.position_history.size());
            LOG_INFO("      Velocity history count: " << lost_features.velocity_history.size());
            LOG_INFO("      Relative positions count: " << lost_features.relative_positions.size());
            
            // Build three-stage memory recovery model
            ThreeStageRecoveryModel recovery_model = buildThreeStageRecoveryModel(lost_features, detection_features);
            
            // Calculate three-stage comprehensive similarity
            float similarity = calculateThreeStageSimilarity(recovery_model, lost_features, detection_features, all_tracks);
            
            
            if (similarity > best_similarity && similarity >= FINAL_RECOVERY_THRESHOLD) {
            
                best_similarity = similarity;
                best_lost_track_id = lost_track->displayId();
            } else if (similarity >= FINAL_RECOVERY_THRESHOLD) {

            } else {

            }
        }
        
        // If a best match is found, recover the track
        if (best_lost_track_id != -1) {
            bool found_and_recovered = false;
            
            // First, search in historical lost tracks
            auto it = lost_stracks_.begin();
            while (it != lost_stracks_.end()) {
                if ((*it)->displayId() == best_lost_track_id) {
                    // 🔥 Fix: save necessary information before deletion to avoid accessing deleted objects
                    int track_id = (*it)->displayId();
                    
                    // Recover track - directly use lost track information
                    detection->permanent_id = best_lost_track_id;
                    detection->is_confirmed = true;
                    detection->is_real_target = true;
                    detection->track_id = best_lost_track_id;
                    // Delete ROI dependency: detection->roi_id = roi_id;
                    detection->recovery_confidence = best_similarity;
                    detection->is_recovered = true;
                    detection->is_activated = true;
                    detection->state = STrack::Tracked;
                    detection->tracklet_len = 1;
                    detection->confirmation_frames = 3;
                    detection->activate(frame_id_); // 确保激活
                    int display_id = detection->displayId();
                    recovered_stracks.push_back(std::move(detection));
                    recovered = true;
                    found_and_recovered = true;
                    
                    // 🔥 Fix: safely delete and update iterator
                    it = lost_stracks_.erase(it);

                    break;
                } else {
                    ++it;
                }
            }
            
            // If not found in historical lost tracks, check current frame lost tracks
            if (!found_and_recovered) {
            }

            if (!found_and_recovered) {
                LOG_INFO("❌ Warning: found best match but cannot recover track ID-" << best_lost_track_id);
            }
        } else {
            LOG_INFO("❌ Warning: no suitable lost track found for recovery");
        }
        
        if (!recovered) {
            remaining_detections.push_back(std::move(detection));
        }
    }
    
    
    // clean all marked for deletion tracks
    auto it = lost_stracks_.begin();
    while (it != lost_stracks_.end()) {
        if ((*it)->isMarkedForDeletion()) {
            it = lost_stracks_.erase(it);
        } else {
            ++it;
        }
    }
    
    return {std::move(remaining_detections), std::move(recovered_stracks)};
}



// New: calculate relative position feature cost - relative position between multiple targets
float TPTrack::calculateRelativePositionCost(const STrack* track, 
                                                       const STrack* detection,
                                                       const std::vector<STrack*>& all_tracks) {
    if (all_tracks.size() <= 1) {
        return 0.5f;  // Return medium cost when there is only one target
    }
    
    cv::Point2f track_center = track->center();
    cv::Point2f det_center = detection->center();
    
    // Calculate relative position between track and other targets
    std::vector<cv::Point2f> track_relative_positions;
    std::vector<cv::Point2f> det_relative_positions;
    
    for (const auto* other_track : all_tracks) {
        if (other_track == track) continue;
        
        cv::Point2f other_center = other_track->center();
        
        // Relative position of track
        cv::Point2f track_rel = other_center - track_center;
        if (cv::norm(track_rel) > 0) {
            track_rel = track_rel / cv::norm(track_rel);  // Normalize
        }
        track_relative_positions.push_back(track_rel);
        
        // Relative position of detection
        cv::Point2f det_rel = other_center - det_center;
        if (cv::norm(det_rel) > 0) {
            det_rel = det_rel / cv::norm(det_rel);  // Normalize
        }
        det_relative_positions.push_back(det_rel);
    }
    
    // Calculate similarity of relative positions
    float similarity = 0.0f;
    int min_size = std::min(track_relative_positions.size(), det_relative_positions.size());
    
    if (min_size > 0) {
        for (int i = 0; i < min_size; ++i) {
            float dot_product = track_relative_positions[i].x * det_relative_positions[i].x + 
                               track_relative_positions[i].y * det_relative_positions[i].y;
            similarity += dot_product;
        }
        similarity /= min_size;
    }
    
    // Convert to cost (1 - similarity)
    return 1.0f - similarity;
}

// Add a helper function at the end of the file:
// Local IoU calculation function (used for internal deduplication)
static float computeIoURect(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = a.width * a.height + b.width * b.height - inter;
    if (union_area <= 0.0f) return 0.0f;
    return inter > 0.0f ? inter / union_area : 0.0f;
}
static void removeDuplicateWithinTracked(std::vector<std::unique_ptr<STrack>>& tracked) {
    if (tracked.size() < 2) return;
    // Mark indices to be removed
    std::vector<bool> remove_flag(tracked.size(), false);
    for (size_t i = 0; i < tracked.size(); ++i) {
        if (remove_flag[i]) continue;
        for (size_t j = i + 1; j < tracked.size(); ++j) {
            if (remove_flag[j]) continue;
            float iou = computeIoURect(tracked[i]->tlwh, tracked[j]->tlwh);
            float center_dist = cv::norm(tracked[i]->center() - tracked[j]->center());
            bool same_roi = (tracked[i]->roi_id > 0 && tracked[i]->roi_id == tracked[j]->roi_id);
            if (iou > 0.85f || (same_roi && center_dist < 15.0f)) {
                // Choose the retainer: first confirmed, then track length
                auto length_i = tracked[i]->frame_id - tracked[i]->start_frame;
                auto length_j = tracked[j]->frame_id - tracked[j]->start_frame;
                bool keep_i = (tracked[i]->is_confirmed && !tracked[j]->is_confirmed) ||
                              (tracked[i]->is_confirmed == tracked[j]->is_confirmed && length_i >= length_j);
                if (keep_i) {
                    remove_flag[j] = true;
                } else {
                    remove_flag[i] = true;
                    break;
                }
            }
        }
    }
    // Filter
    std::vector<std::unique_ptr<STrack>> kept;
    kept.reserve(tracked.size());
    for (size_t i = 0; i < tracked.size(); ++i) {
        if (!remove_flag[i]) kept.push_back(std::move(tracked[i]));
    }
    tracked = std::move(kept);
}

} // namespace tracking 
