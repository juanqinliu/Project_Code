#include "roi/ROIManager.h"
#include "tracking/STrack.h"  // Include full definition for implementation
#include <algorithm>
#include <iostream>
#include <cmath>
#include <unordered_set>

namespace tracking {

// ROIManager Implementation
ROIManager::ROIManager(const Config& config) 
    : config_(config), next_roi_id_(1), current_frame_id_(0) {
    // Remove ROIOptimizer dependency, directly use ROI class methods
}


int ROIManager::allocateROIId() {
    if (!available_roi_ids_.empty()) {
        int id = available_roi_ids_.front();
        available_roi_ids_.pop_front();
        return id;
    }
    return next_roi_id_++;
}

void ROIManager::recycleROIId(int roi_id) {
    available_roi_ids_.push_back(roi_id);
}

int ROIManager::createROIForTrack(const STrack& track, int frame_width, int frame_height, int frame_id) {
    // Use predicted center point from Kalman filter, not current center point
    cv::Point2f track_center = track.getPredictedCenter();
    
    int roi_size = config_.roi_size;
    int roi_x = static_cast<int>(track_center.x - roi_size / 2);
    int roi_y = static_cast<int>(track_center.y - roi_size / 2);
    
    // Boundary constraints
    roi_x = std::max(0, std::min(roi_x, frame_width - roi_size));
    roi_y = std::max(0, std::min(roi_y, frame_height - roi_size));
    
    int actual_width = std::min(roi_size, frame_width - roi_x);
    int actual_height = std::min(roi_size, frame_height - roi_y);
    
    cv::Rect roi_rect(roi_x, roi_y, actual_width, actual_height);
    
    // Check overlap
    for (const auto& [existing_id, existing_roi] : rois_) {
        cv::Rect existing_rect = existing_roi->bbox;
        cv::Rect intersection = roi_rect & existing_rect;
        float overlap_ratio = static_cast<float>(intersection.area()) / 
                            std::min(roi_rect.area(), existing_rect.area());
        
        if (overlap_ratio > config_.roi_overlap_threshold) {
            return existing_id;
        }
    }
    
    int roi_id = allocateROIId();
    auto new_roi = std::make_unique<ROI>(roi_id, roi_rect);
    new_roi->track_ids.push_back(track.displayId());
    new_roi->last_updated = frame_id;
    
    rois_[roi_id] = std::move(new_roi);
    addTrackMemory(roi_id, track, frame_id);
    
    return roi_id;
}

void ROIManager::addTrackMemory(int roi_id, const STrack& track, int frame_id) {
    if (rois_.find(roi_id) == rois_.end()) return;
    
    int track_id = track.displayId();
    auto& memories = rois_[roi_id]->track_memories;
    if (memories.find(track_id) == memories.end()) {
        memories[track_id] = std::make_unique<ROIMemory>(track_id, track.center(), frame_id);
    }
    memories[track_id]->updateMemory(track.center(), track.score, frame_id);
}

void ROIManager::updateROIPositions(std::vector<std::unique_ptr<STrack>>& tracks, 
                                           int frame_width, int frame_height) {
    fixTrackROIAssociations(tracks);
    
    // LOG_INFO("=== ROI positions update start ===");
    
    for (auto& [roi_id, roi] : rois_) {
        std::vector<STrack*> roi_tracks = getTracksInROI(tracks, roi_id, *roi);
        if (roi_tracks.empty()) {
            roi_tracks = findTracksByPosition(tracks, *roi);
        }
        if (!roi_tracks.empty()) {
            // LOG_INFO("UpdateROI-" << roi_id << " position");
            // LOG_INFO("  Original position: (" << roi->bbox.x << ", " << roi->bbox.y  << ", " << roi->bbox.width << ", " << roi->bbox.height << ")");
            
            std::vector<cv::Point2f> track_centers;
            std::vector<std::pair<cv::Point2f, cv::Size2f>> track_info; 
            std::unordered_set<int> active_ids;
            for (auto* track : roi_tracks) {
                cv::Point2f predicted_center = track->getPredictedCenter();
                cv::Size2f track_size(track->tlwh.width, track->tlwh.height);
                
                // 🔥 Grace Period expansion: if track is in Grace Period (using prediction), expand uncertainty range
                if (track->miss_count_in_grace > 0) {
                    // Expand uncertainty based on miss count linearly
                    // miss 1次：+20%，miss 2次：+40%，miss 3次：+60%
                    float uncertainty_factor = 1.0f + (track->miss_count_in_grace * 0.2f);
                    track_size.width *= uncertainty_factor;
                    track_size.height *= uncertainty_factor;
                    
                    // LOG_INFO("  🔥 Target ID-" << track->displayId() << " in Grace Period (miss: " 
                    //           << track->miss_count_in_grace << "), expanding size by " 
                    //           << (uncertainty_factor * 100 - 100) << "%");
                }
                
                track_centers.push_back(predicted_center);
                track_info.emplace_back(predicted_center, track_size);
                active_ids.insert(track->displayId());
                // LOG_INFO("  Target ID-" << track->displayId() << " predicted center: (" << predicted_center.x << ", " << predicted_center.y << "), size: (" << track_size.width << "x" << track_size.height << ")");
            }

            // Memory enhancement: Include recently lost but reliable target positions into ROI update, avoid sudden ROI jump/shrink due to brief missing detection
            for (const auto& kv : roi->track_memories) {
                int mem_track_id = kv.first;
                const auto& memory = kv.second;
                if (active_ids.find(mem_track_id) != active_ids.end()) continue; 
                if (!memory) continue;
                if (memory->isReliable() && memory->lost_duration <= config_.roi_memory_frames) {
                    track_centers.push_back(memory->last_position);
                    
                    cv::Size2f memory_size(50.0f, 50.0f);   
                    if (!memory->size_history.empty()) {
                        memory_size = memory->size_history.back();
                    }
                    track_info.emplace_back(memory->last_position, memory_size);
                }
            }
            
            // Check target number, if more than 1 target, may need to split instead of update
            if (track_centers.size() > 1) {
                // LOG_INFO("  ⚠️ ROI-" << roi_id << " contains " << track_centers.size() << " targets, may need to split");
                // Calculate distance between targets
                float max_distance = 0.0f;
                for (size_t i = 0; i < track_centers.size(); ++i) {
                    for (size_t j = i + 1; j < track_centers.size(); ++j) {
                        float distance = cv::norm(track_centers[i] - track_centers[j]);
                        max_distance = std::max(max_distance, distance);
                    }
                }
                // LOG_INFO("  Max distance between targets: " << max_distance << " pixels");
                
                // If target distance is too far, skip position update, wait for split
                if (max_distance > 500.0f) {
                    // LOG_INFO("  ❌ Target distance is too far, skip position update, wait for split");
                    continue;
                }
            }
            
            // Use new adaptive ROI update method, include track size information
            bool updated = roi->adaptiveUpdateWithTrackInfo(track_info, frame_width, frame_height, config_, false);
            if (updated) {
                roi->last_updated = current_frame_id_;
                
                // LOG_INFO("  New position: (" << roi->bbox.x << ", " << roi->bbox.y 
                //         << ", " << roi->bbox.width << ", " << roi->bbox.height << ")");
                // LOG_INFO("  ✅ ROI-" << roi_id << " position updated (adaptive size)");
            } else {
                // LOG_INFO("  ❌ ROI-" << roi_id << " position not updated");
            }
        } else {
            // LOG_INFO("ROI-" << roi_id << " no associated targets, skip position update");
        }
    }
    
    // LOG_INFO("=== ROI positions update end ===");
}

std::vector<STrack*> ROIManager::getTracksInROI(std::vector<std::unique_ptr<STrack>>& tracks, 
                                                               int roi_id, const ROI& roi) {
    std::vector<STrack*> roi_tracks_by_pos;
    std::vector<STrack*> roi_tracks_by_id;
    std::set<int> all_track_ids;
    std::vector<STrack*> result_tracks;
    
    // LOG_INFO("🔍 Check the track association of ROI-" << roi_id);
    for (auto& track : tracks) {
        // LOG_INFO("  Track ID-" << track->displayId() << " roi_id=" << track->roi_id << " position:(" << track->center().x << "," << track->center().y << ")");
        
        // 🔥 Use position matching instead of roi_id matching
        if (roi.containsPoint(track->center(), 50)) {
            roi_tracks_by_pos.push_back(track.get());
        } else if (track->roi_id == roi_id) {
            roi_tracks_by_id.push_back(track.get());
        }
    }
    
    // 🔥 Use position matching instead of roi_id matching
    for (auto* track : roi_tracks_by_pos) {
        if (all_track_ids.find(track->displayId()) == all_track_ids.end()) {
            all_track_ids.insert(track->displayId());
            result_tracks.push_back(track);
            // Update track's roi_id to match current ROI
            if (track->roi_id != roi_id) {
                int old_roi_id = track->roi_id;
                track->roi_id = roi_id;
                // LOG_INFO("    🔥 Update track ID-" << track->displayId() << " roi_id: " << old_roi_id << " -> " << roi_id);
            }
        }
    }
    
    // Then process tracks matched by roi_id (if not already processed by position matching)
    for (auto* track : roi_tracks_by_id) {
        if (all_track_ids.find(track->displayId()) == all_track_ids.end()) {
            all_track_ids.insert(track->displayId());
            result_tracks.push_back(track);
            if (track->roi_id != roi_id) {
                track->roi_id = roi_id;
            }
        }
    }
    
    return result_tracks;
}

std::vector<STrack*> ROIManager::findTracksByPosition(const std::vector<std::unique_ptr<STrack>>& tracks, 
                                                                     const ROI& roi) {
    std::vector<STrack*> result;
    for (const auto& track : tracks) {
        if (roi.containsPoint(track->getPredictedCenter(), 50)) {
            result.push_back(track.get());
        }
    }
    return result;
}

    std::optional<int> ROIManager::findBestROIForTrack(cv::Point2f track_center) {
    std::vector<std::pair<int, float>> candidates;
    
    for (const auto& [roi_id, roi] : rois_) {
        bool contains = roi->containsPoint(track_center, 100);
        float distance = cv::norm(track_center - roi->center());

        if (contains) {
            candidates.emplace_back(roi_id, distance);
        }
    }
    
    if (!candidates.empty()) {
        auto min_it = std::min_element(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        return min_it->first;
    }
    
    return std::nullopt;
}

void ROIManager::fixTrackROIAssociations(std::vector<std::unique_ptr<STrack>>& tracks) {
    for (const auto& [roi_id, roi] : rois_) {
        // LOG_INFO("  Existing ROI-" << roi_id << " position: (" << roi->bbox.x << ", " << roi->bbox.y << ", " << roi->bbox.width << ", " << roi->bbox.height << ")");
    }
    
    for (auto& track : tracks) {
        // LOG_INFO("🔥 Check track ID-" << track->displayId() << " (current roi_id=" << track->roi_id << ")");
        
        // 🔥 Use position matching instead of roi_id matching
        cv::Point2f track_center = track->getPredictedCenter();
        bool found_by_position = false;
        
        // First try to find a suitable ROI by position matching
        for (const auto& [roi_id, roi] : rois_) {
            if (roi->containsPoint(track_center, 50)) {
                if (track->roi_id != roi_id) {
                    int old_roi_id = track->roi_id;
                    track->roi_id = roi_id;
                    // LOG_INFO("  🔥 Update track ID-" << track->displayId() << " roi_id: " << old_roi_id << " -> " << roi_id);
                    
                    // Remove track ID from old ROI (if exists)
                    if (old_roi_id > 0 && rois_.find(old_roi_id) != rois_.end()) {
                        auto& old_track_ids = rois_[old_roi_id]->track_ids;
                        old_track_ids.erase(std::remove(old_track_ids.begin(), old_track_ids.end(), track->displayId()), old_track_ids.end());
                        // LOG_INFO("     Remove track ID-" << track->displayId() << " from ROI-" << old_roi_id);
                    }
                    
                    // Add to new ROI
                    if (std::find(roi->track_ids.begin(), roi->track_ids.end(), track->displayId()) == roi->track_ids.end()) {
                        roi->track_ids.push_back(track->displayId());
                        // LOG_INFO("     Add to ROI-" << roi_id << " track_ids list");
                    }
                }
                found_by_position = true;
                break;
            }
        }
        
        // If not found by position matching, check if roi_id is valid
        if (!found_by_position) {
            bool roi_exists = false;
            if (track->roi_id > 0) {
                roi_exists = (rois_.find(track->roi_id) != rois_.end());
                // LOG_INFO("  Track roi_id=" << track->roi_id << " existence check: " << (roi_exists ? "exists" : "not exists"));
            }
            
            if (track->roi_id == -1 || !roi_exists) {
                // LOG_INFO("  Need to reallocate ROI, track center: (" << track_center.x << ", " << track_center.y << ")");
                
                auto best_roi_id = findBestROIForTrack(track_center);
                if (best_roi_id) {
                    int old_roi_id = track->roi_id;
                    track->roi_id = *best_roi_id;
                    // LOG_INFO("  Find best ROI-" << *best_roi_id << "，update track roi_id from " << old_roi_id << " to " << *best_roi_id);
                    
                    // Remove track ID from old ROI (if exists)
                    if (old_roi_id > 0 && rois_.find(old_roi_id) != rois_.end()) {
                        auto& old_track_ids = rois_[old_roi_id]->track_ids;
                        old_track_ids.erase(std::remove(old_track_ids.begin(), old_track_ids.end(), track->displayId()), old_track_ids.end());
                        // LOG_INFO("     Remove track ID-" << track->displayId() << " from ROI-" << old_roi_id);
                    }
                    
                    if (std::find(rois_[*best_roi_id]->track_ids.begin(), 
                                 rois_[*best_roi_id]->track_ids.end(), 
                                 track->displayId()) == rois_[*best_roi_id]->track_ids.end()) {
                        rois_[*best_roi_id]->track_ids.push_back(track->displayId());
                        
                    }
                } else {
                    track->roi_id = -1;
                }
            } else {
                // LOG_INFO("  roi_id is valid, no need to repair");
            }
        }
    }

}

int ROIManager::mergeOverlappingROIs(int frame_width, int frame_height) {
    if (rois_.size() < 2) return 0;
    
    int merged_count = 0;
    std::vector<int> roi_ids;
    for (const auto& [id, _] : rois_) {
        roi_ids.push_back(id);
    }
    
    
    for (size_t i = 0; i < roi_ids.size(); ++i) {
        int roi_id1 = roi_ids[i];
        if (rois_.find(roi_id1) == rois_.end()) continue;
        
        for (size_t j = i + 1; j < roi_ids.size(); ++j) {
            int roi_id2 = roi_ids[j];
            if (rois_.find(roi_id2) == rois_.end()) continue;
            
            float overlap_ratio = rois_[roi_id1]->calculateOverlapRatio(*rois_[roi_id2]);

            if (rois_[roi_id1]->shouldMergeWith(*rois_[roi_id2])) {
                mergeROIs(roi_id1, roi_id2, frame_width, frame_height);
                merged_count++;
                break; 
            }
        }
    }
    
    return merged_count;
}

void ROIManager::mergeROIs(int roi_id1, int roi_id2, int frame_width, int frame_height) {
    auto& roi1 = rois_[roi_id1];
    auto& roi2 = rois_[roi_id2];
    
    if (!roi1 || !roi2) return;
    
    cv::Rect new_rect = roi1->calculateMergedBbox(*roi2, frame_width, frame_height, 0);

    roi1->updatePosition(new_rect.x, new_rect.y, new_rect.width, new_rect.height);
    roi1->last_updated = current_frame_id_;
    roi1->is_merged = true;
    
    for (int track_id : roi2->track_ids) {
        if (std::find(roi1->track_ids.begin(), roi1->track_ids.end(), track_id) == roi1->track_ids.end()) {
            roi1->track_ids.push_back(track_id);
        }
    }
    

    for (auto& [track_id, memory] : roi2->track_memories) {
        if (roi1->track_memories.find(track_id) == roi1->track_memories.end()) {
            roi1->track_memories[track_id] = std::move(memory);
        } else {
            if (memory->last_seen_frame > roi1->track_memories[track_id]->last_seen_frame) {
                roi1->track_memories[track_id] = std::move(memory);
            }
        }
    }

    roi1->no_detection_count = std::min(roi1->no_detection_count, roi2->no_detection_count);
    roi1->no_tracking_count = std::min(roi1->no_tracking_count, roi2->no_tracking_count);
    

    rois_.erase(roi_id2);
    recycleROIId(roi_id2);
}

int ROIManager::splitOversizedROIs(std::vector<std::unique_ptr<STrack>>& tracks, 
                                          int frame_width, int frame_height, int frame_id) {

    if (rois_.size() > 8) {
        return 0;  // Avoid producing too many ROI to affect performance
    }
    
    int split_count = 0;
    std::vector<int> roi_ids;
    for (const auto& [id, _] : rois_) {
        roi_ids.push_back(id);
    }
    
    
    // Sort ROI by target number, prioritize splitting ROI with multiple targets
    std::sort(roi_ids.begin(), roi_ids.end(), [this, &tracks](int id1, int id2) {
        int count1 = 0, count2 = 0;
        for (const auto& track : tracks) {
            if (track->roi_id == id1) count1++;
            if (track->roi_id == id2) count2++;
        }
        return count1 > count2; // Descending order
    });
    
    for (int roi_id : roi_ids) {
        if (rois_.find(roi_id) == rois_.end()) continue;
        
        auto& roi = rois_[roi_id];
        std::vector<STrack*> roi_tracks;
        for (const auto& track : tracks) {
            if (track->roi_id == roi_id) {
                roi_tracks.push_back(track.get());
            }
        }
        
        // LOG_INFO("Check if ROI-" << roi_id << " should be split");
        // LOG_INFO("ROI-" << roi_id << " position: (" << roi->bbox.x << ", " << roi->bbox.y  << ", " << roi->bbox.width << ", " << roi->bbox.height << ")");
        // LOG_INFO("ROI-" << roi_id << " target number: " << roi_tracks.size());
        
        // Print actual tracking IDs in ROI
        // LOG_INFO("ROI-" << roi_id << " tracking IDs list:");
        for (auto* track : roi_tracks) {
            // LOG_INFO("  Target ID-" << track->displayId() << " position: (" << track->getPredictedCenter().x << ", " << track->getPredictedCenter().y << ")");
        }
        
        // Print track_ids in ROI
        // LOG_INFO("ROI-" << roi_id << " 记录的track_ids: [");
        for (size_t i = 0; i < roi->track_ids.size(); ++i) {
            // LOG_INFO("  " << roi->track_ids[i] << (i < roi->track_ids.size() - 1 ? "," : ""));
        }
        // LOG_INFO("]");
        
        // Unified split condition: at least 2 targets
        if (roi_tracks.size() < 2) {
            // LOG_INFO("❌ Target number is less than 2, skip split");
            continue;
        }
        
        // Collect target centers and corresponding track_ids
        std::vector<cv::Point2f> track_centers;
        std::vector<int> track_ids;
        for (auto* track : roi_tracks) {
            track_centers.push_back(track->getPredictedCenter());
            track_ids.push_back(track->displayId());
        }
        
        // Calculate distance between targets
        float max_distance = 0.0f;
        std::pair<int, int> max_distance_pair = {-1, -1};
        for (size_t i = 0; i < track_centers.size(); ++i) {
            for (size_t j = i + 1; j < track_centers.size(); ++j) {
                float distance = cv::norm(track_centers[i] - track_centers[j]);
                if (distance > max_distance) {
                    max_distance = distance;
                    max_distance_pair = {roi_tracks[i]->displayId(), roi_tracks[j]->displayId()};
                }
            }
        }
        
        // LOG_INFO("Max distance between targets: " << max_distance << " pixels (ID-" << max_distance_pair.first << " and ID-" << max_distance_pair.second << ")");
        // LOG_INFO("Split threshold: 500 pixels");
        
        // Use ROI class method to determine if split is needed
        if (roi->shouldSplit(track_centers, config_.roi_size)) {
            // LOG_INFO("✅ Split condition met, execute split");
            if (executeROISplit(roi_id, track_centers, track_ids, frame_width, frame_height, frame_id, tracks)) {
                split_count++;
                // LOG_INFO("✅ ROI-" << roi_id << " split successfully");
                
                // Limit single split count, avoid excessive splitting
                if (split_count >= 2) break;
            } else {
                // LOG_INFO("❌ ROI-" << roi_id << " split failed");
            }
        } else {
            // LOG_INFO("❌ Split condition not met");
        }
    }
    

    return split_count;
}

bool ROIManager::executeROISplit(int roi_id, const std::vector<cv::Point2f>& track_centers, 
                                        const std::vector<int>& track_ids, int frame_width, int frame_height, int frame_id,
                                        std::vector<std::unique_ptr<STrack>>& tracks) {
    auto old_roi = rois_[roi_id].get();
    if (!old_roi) return false;
    
    for (size_t i = 0; i < old_roi->track_ids.size(); ++i) {
        // LOG_INFO("    " << old_roi->track_ids[i] << (i < old_roi->track_ids.size() - 1 ? "," : ""));
    }

    
    int base_size = config_.roi_size;

    std::vector<cv::Rect> split_configs = old_roi->generateSplitConfigs(track_centers, frame_width, frame_height, base_size);

    if (split_configs.size() < 2) {
        return false;
    }
    
    // Get target clustering information for correct allocation
    std::vector<std::vector<int>> clusters = old_roi->clusterTrackCenters(track_centers, 500.0f);

    // Before creating new ROIs after splitting, record the track information in the old ROI
    std::vector<int> old_track_ids = old_roi->track_ids;
    std::unordered_map<int, std::unique_ptr<ROIMemory>> old_memories;
    
    // Move memory ownership
    for (auto& [track_id, memory] : old_roi->track_memories) {
        old_memories[track_id] = std::move(memory);
    }
    
    rois_.erase(roi_id);
    recycleROIId(roi_id);

    for (size_t i = 0; i < split_configs.size(); ++i) {
        int new_roi_id = allocateROIId();
        auto new_roi = std::make_unique<ROI>(new_roi_id, split_configs[i]);
        new_roi->last_updated = frame_id;
        
        if (i < clusters.size()) {
            const auto& cluster = clusters[i];

            
            for (int target_idx : cluster) {

                if (target_idx < track_ids.size()) {
                    int track_id = track_ids[target_idx];
                    new_roi->track_ids.push_back(track_id);

                    for (auto& track : tracks) {
                        if (track->displayId() == track_id) {
                            int old_roi_id = track->roi_id;
                            track->roi_id = new_roi_id;
                            
                            break;
                        }
                    }
                    

                    if (old_memories.find(track_id) != old_memories.end()) {
                        new_roi->track_memories[track_id] = std::move(old_memories[track_id]);

                    }
                }
            }
        }
        
        rois_[new_roi_id] = std::move(new_roi);
    }
    
    
    return true;
}

void ROIManager::updateCandidates(const std::vector<Detection>& detections_outside_roi, int frame_id) {
    for (const auto& detection : detections_outside_roi) {
        cv::Point2f center = detection.center();
        CandidateTarget* closest_candidate = nullptr;
        float min_distance = std::numeric_limits<float>::infinity();
        
        for (auto& [id, candidate] : candidates_) {
            float distance = cv::norm(center - candidate->avgCenter());
            if (distance < min_distance && distance < 50) {
                min_distance = distance;
                closest_candidate = candidate.get();
            }
        }
        
        if (closest_candidate) {
            closest_candidate->update(center, frame_id);
        } else {
            int candidate_id = candidates_.size() + 1;
            candidates_[candidate_id] = std::make_unique<CandidateTarget>(candidate_id, center, frame_id);
        }
    }
}

int ROIManager::createROIsForConfirmedCandidates(int frame_width, int frame_height, int frame_id) {
    int confirmed_count = 0;
    std::vector<int> to_remove;
    
    for (auto& [candidate_id, candidate] : candidates_) {
        if (candidate->shouldConfirm(config_)) {
            cv::Point2f center = candidate->avgCenter();
            int roi_size = config_.roi_size;
            int roi_x = static_cast<int>(center.x - roi_size / 2);
            int roi_y = static_cast<int>(center.y - roi_size / 2);
            
            roi_x = std::max(0, std::min(roi_x, frame_width - roi_size));
            roi_y = std::max(0, std::min(roi_y, frame_height - roi_size));
            int roi_width = std::min(roi_size, frame_width - roi_x);
            int roi_height = std::min(roi_size, frame_height - roi_y);
            
            int roi_id = allocateROIId();
            auto roi = std::make_unique<ROI>(roi_id, cv::Rect(roi_x, roi_y, roi_width, roi_height));
            roi->last_updated = frame_id;
            rois_[roi_id] = std::move(roi);
            
            to_remove.push_back(candidate_id);
            confirmed_count++;
        }
    }
    
    for (int id : to_remove) {
        candidates_.erase(id);
    }
    return confirmed_count;
}

int ROIManager::cleanupInactiveROIs(int current_frame) {
    std::vector<int> inactive_rois;
    for (const auto& [roi_id, roi] : rois_) {
        if (roi->no_detection_count >= config_.roi_max_no_detection &&
            roi->no_tracking_count >= config_.roi_max_no_detection) {
            inactive_rois.push_back(roi_id);
        }
    }
    
    for (int roi_id : inactive_rois) {
        rois_.erase(roi_id);
        recycleROIId(roi_id);
    }
    

    return inactive_rois.size();
}

void ROIManager::updateROIDetectionStatus(const std::unordered_map<int, std::vector<Detection>>& roi_detections) {
    for (auto& [roi_id, roi] : rois_) {
        auto it = roi_detections.find(roi_id);
        if (it != roi_detections.end() && !it->second.empty()) {
            roi->no_detection_count = 0;
        } else {
            roi->no_detection_count++;
        }
    }
}

void ROIManager::updateROITrackingStatus(std::vector<std::unique_ptr<STrack>>& tracks, int frame_id) {
    for (auto& [roi_id, roi] : rois_) {
        roi->no_tracking_count++;
    }
    
    for (const auto& track : tracks) {
        int roi_id = track->roi_id;
        if (roi_id != -1 && rois_.find(roi_id) != rois_.end()) {
            rois_[roi_id]->no_tracking_count = 0;
            addTrackMemory(roi_id, *track, frame_id);
        }
    }
}

void ROIManager::updateROITrackMemories(std::vector<std::unique_ptr<STrack>>& tracks, int frame_id) {
    std::set<int> active_track_ids;
    for (const auto& track : tracks) {
        active_track_ids.insert(track->displayId());
    }
    
    for (auto& [roi_id, roi] : rois_) {
        std::vector<int> to_remove;
        for (auto& [track_id, memory] : roi->track_memories) {
            if (active_track_ids.find(track_id) != active_track_ids.end()) {
                auto it = std::find_if(tracks.begin(), tracks.end(), 
                    [track_id](const auto& t) { return t->displayId() == track_id; });
                if (it != tracks.end()) {
                    memory->updateMemory((*it)->center(), (*it)->score, frame_id);
                }
            } else {
                memory->incrementLost();
                if (memory->lost_duration > 30) {
                    to_remove.push_back(track_id);
                }
            }
        }
        for (int track_id : to_remove) {
            roi->track_memories.erase(track_id);
        }
    }
}

std::unordered_map<std::string, int> ROIManager::performROIManagement(
    std::vector<std::unique_ptr<STrack>>& tracks, 
    int frame_width, int frame_height, int frame_id) {
    std::unordered_map<std::string, int> stats;
    stats["merged"] = 0;
    stats["split"] = 0;
    stats["cleaned"] = 0;
    
    updateROIPositions(tracks, frame_width, frame_height);
    
    int expanded_count = expandROIsForSafety(tracks, frame_width, frame_height);
    stats["expanded"] = expanded_count;

    stats["cleaned"] = cleanupInactiveROIs(frame_id);

    bool should_check_merge = (frame_id % 5 == 0);

    
    if (should_check_merge && rois_.size() > 1) {
        stats["merged"] = mergeOverlappingROIs(frame_width, frame_height);
    }
    

    bool should_check_split = (frame_id % 5 == 0);

    
    if (should_check_split) {
        stats["split"] = splitOversizedROIs(tracks, frame_width, frame_height, frame_id);
    }
    

    if (stats["expanded"] > 0) {
        // LOG_INFO("Expanded: " << stats["expanded"] << " ROIs due to safety zone violation");
    }
    
    return stats;
}

std::unordered_map<std::string, int> ROIManager::dynamicROIManagement(
    std::vector<std::unique_ptr<STrack>>& tracks, 
    int frame_width, int frame_height, int frame_id) {

    auto stats = performROIManagement(tracks, frame_width, frame_height, frame_id);
    
    int confirmed = createROIsForConfirmedCandidates(frame_width, frame_height, frame_id);
    
    return stats;
}

std::unordered_map<std::string, int> ROIManager::localPhaseROIManagement(
    std::vector<std::unique_ptr<STrack>>& tracks, 
    int frame_width, int frame_height, int frame_id) {
    // 🔥 Local phase: perform the same ROI management
    return performROIManagement(tracks, frame_width, frame_height, frame_id);
}

int ROIManager::expandROIsForSafety(std::vector<std::unique_ptr<STrack>>& tracks, 
                                           int frame_width, int frame_height) {

    int expanded_count = 0;
    std::unordered_map<int, std::vector<std::pair<cv::Point2f, cv::Size2f>>> roi_track_info;
    

    for (const auto& track : tracks) {
        if (track->is_activated && track->roi_id > 0 && rois_.find(track->roi_id) != rois_.end()) {
            cv::Point2f predicted_center = track->getPredictedCenter();
            cv::Size2f track_size(track->tlwh.width, track->tlwh.height);
            roi_track_info[track->roi_id].emplace_back(predicted_center, track_size);
        }
    }

    for (auto& [roi_id, track_info] : roi_track_info) {
        if (track_info.empty()) continue;
        
        auto& roi = rois_[roi_id];
        

        std::vector<cv::Point2f> track_centers;
        for (const auto& info : track_info) {
            track_centers.push_back(info.first);
        }
        

        auto violations = roi->getSafetyZoneViolations(track_centers);
        if (!violations.empty()) {

            bool updated = roi->adaptiveUpdateWithTrackInfo(track_info, frame_width, frame_height, config_, true);
            if (updated) {
                roi->last_updated = current_frame_id_;
                expanded_count++;

            }
        }
    }
    
    return expanded_count;
}

std::optional<ROIMemory*> ROIManager::findCandidateForRecovery(cv::Point2f detection_center, int roi_id) {
    auto it = rois_.find(roi_id);
    if (it == rois_.end()) return std::nullopt;
    
    return it->second->findCandidateForRecovery(detection_center, config_);
}

std::optional<int> ROIManager::findROIIdByPoint(cv::Point2f point, int margin) const {
    for (const auto& [roi_id, roi] : rois_) {
        if (roi->containsPoint(point, margin)) {
            return roi_id;
        }
    }
    return std::nullopt;
}


ROI* ROIManager::getROI(int roi_id) {
    auto it = rois_.find(roi_id);
    return (it != rois_.end()) ? it->second.get() : nullptr;
}

const ROI* ROIManager::getROI(int roi_id) const {
    auto it = rois_.find(roi_id);
    return (it != rois_.end()) ? it->second.get() : nullptr;
}

void ROIManager::forEachROI(std::function<void(int, ROI&)> func) {
    for (auto& [roi_id, roi] : rois_) {
        func(roi_id, *roi);
    }
}

} // namespace tracking 