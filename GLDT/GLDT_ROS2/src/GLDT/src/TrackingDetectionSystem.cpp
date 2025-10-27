#include "TrackingDetectionSystem.h"
#include "tracking/TPTrack.h"
#include "tracking/STrack.h"
#include "roi/ROIManager.h"
#include "roi/ROI.h"
#include "roi/ROIMemory.h"
#include "roi/CandidateTarget.h"
#include "common/Detection.h"
#include "common/Config.h"
#include "inference/InferenceFactory.h"
#include "common/Logger.h"  
#include "common/Flags.h"
#include "inference/TensorRTGlobalInference.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>  
#include <limits>  
#include <sstream>  
#ifdef _OPENMP
#include <omp.h>
#endif
#define ENABLE_ROI_PERF_TEST

namespace tracking {


// TrackingDetectionSystem implementation
TrackingDetectionSystem::TrackingDetectionSystem(const std::string& global_model_path, 
                          const std::string& local_model_path,
                          const Config& config)
    : config_(config), frame_count_(0), current_frame_id_(0), is_first_frame_(true), total_frames_(0),
      global_model_path_(global_model_path), local_model_path_(local_model_path), 
      force_global_phase_(false), force_global_start_frame_(0),
      use_global_detection_(true) {
    
    // Check if log level has been set via environment variable, if so, don't override
    char* env_log_level = getenv("GLOG_minloglevel");
    if (env_log_level == nullptr) {
        // Only use log level from config when environment variable is not set
        if (config_.log_level >= 0) {
            LOG_INFO("Using log level from config file: " << config_.log_level);
            Logger::setLogLevel(config_.log_level);
        }
    } else {
        LOG_WARNING("Environment variable GLOG_minloglevel is set to " << env_log_level << ", keeping this setting");
    }
    
    // Set whether to enable verbose logging
    if (getenv("GLOG_v") == nullptr) {
        Logger::setVerbose(config_.verbose_logging);
    }
    
    // 🔥 Set detection mode using configuration flags
    detection_mode_ = FLAGS_detection_mode;
    if (detection_mode_ == 0) {
        LOG_INFO("⚠️ Detection mode: Global detection only (no local ROI detection)");
    } else {
        LOG_INFO("✅ Detection mode: Global + Local joint detection (default mode)");
    }
    
    
    LOG_INFO("Initializing tracking detection system...");
    LOG_INFO("Global phase duration: " << config_.global_duration << " frames");
    LOG_INFO("Local phase duration: " << config_.local_duration << " frames");
    LOG_INFO("Total cycle length: " << (config_.global_duration + config_.local_duration) << " frames");
    
    // Initialize ROI manager
    roi_manager_ = std::make_shared<ROIManager>(config);
    
    // Initialize global and local trackers
    global_tracker_ = std::make_unique<TPTrack>(config, 30, roi_manager_);
    local_tracker_ = std::make_unique<TPTrack>(config, 30, roi_manager_);
    
    // Use global tracker as main tracker by default
    tracker_ = std::move(global_tracker_);
    global_tracker_ = std::make_unique<TPTrack>(config, 30, roi_manager_);
    
    // Set whether to use original ByteTrack using configuration flags
    bool use_original_bytetrack = FLAGS_use_original_bytetrack;
    tracker_->setUseOriginalByteTrack(use_original_bytetrack);
    if (use_original_bytetrack) {
        LOG_INFO("⚠️ Using original ByteTrack algorithm (ROI constraints and memory recovery disabled)");
    } else {
        LOG_INFO("✅ Using enhanced ByteTrack algorithm (ROI constraints and memory recovery enabled)");
    }
    
    // Initialize inference engines using factory pattern, global and local use different model files
    try {
        global_inference_ = createGlobalInferenceEngine(global_model_path);
        local_inference_ = createLocalInferenceEngine(local_model_path);
        LOG_INFO("✅ Inference engine initialization successful:");
        LOG_INFO("   - Global detection model: " << global_model_path);
        LOG_INFO("   - Local detection model: " << local_model_path);
    } catch (const std::exception& e) {
        LOG_ERROR("❌ Inference engine initialization failed: " << e.what());
        throw;
    }
    
    LOG_INFO("Tracking detection system initialization complete");
    
    // 🔥 Initialize GPU monitoring
    initializeGPUMonitoring();
    
    // 🔥 Initialize OpenMP parallel detection
    #ifdef _OPENMP
    num_threads_ = omp_get_max_threads();
    LOG_INFO("OpenMP available, max threads: " << num_threads_);
    // Delay initialization of thread-local inference engines, create on first use
    #else
    num_threads_ = 1;
    LOG_INFO("OpenMP not available, using single-thread mode");
    #endif
}
    
std::tuple<std::vector<Detection>, std::vector<std::unique_ptr<STrack>>> 
TrackingDetectionSystem::process(const cv::Mat& frame) {
    // 🔥 Start overall processing time measurement
    auto total_start = std::chrono::high_resolution_clock::now();
    
    frame_count_++;
    current_frame_id_ = frame_count_;
    LOG_INFO("frame_count_: " << frame_count_);
    
    // Set current frame ID for ROI manager
    if (detection_mode_ != 0) { // Only update ROI manager state in non-global detection mode
        roi_manager_->setCurrentFrameId(current_frame_id_);
    }
    
    std::vector<Detection> detections;
    std::vector<std::unique_ptr<STrack>> updated_tracks;
    std::unordered_map<int, std::vector<Detection>> roi_detections;
    
    // 🔥 Decide processing flow based on detection mode
    if (detection_mode_ == 0 || isGlobalPhase()) {
        // Global detection phase (or global detection only mode)
        detections = globalDetectionWithMotion(frame);
        
        // 🔥 Start tracking time measurement
        auto tracking_start = std::chrono::high_resolution_clock::now();
        updated_tracks = tracker_->update(detections);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        double tracking_time = std::chrono::duration<double, std::milli>(tracking_end - tracking_start).count();
        updateTrackingTime(tracking_time);
        
        handleGlobalPhase(updated_tracks, detections, frame.cols, frame.rows);
        
        if (detection_mode_ == 1 && isGlobalPhaseEnd()) {
            finalizeGlobalPhaseEnhanced(frame.cols, frame.rows);
        }
    } else if (detection_mode_ == 1) {
        // Local detection phase - first check if there are available ROIs
        const auto& rois = roi_manager_->getROIs();
        if (rois.empty()) {
            // No ROI available, activate forced global phase, but don't update tracking twice in the same frame
            LOG_WARNING("⚠️ Frame " << current_frame_id_ << ": No ROI in local phase, marking switch to global detection for next frame");
            activateForceGlobalPhase();
            // Keep current frame result empty, handled uniformly by upper layer
        } else {
            // Normal local detection
            auto [local_dets, roi_dets] = localDetection(frame);
            detections = local_dets;
            roi_detections = roi_dets;
            
            // 🔥 Update ROI detection status (included in ROI adjustment time)
            roi_manager_->updateROIDetectionStatus(roi_detections);
            
            // 🔥 Start tracking time measurement
            auto tracking_start = std::chrono::high_resolution_clock::now();
            updated_tracks = tracker_->update(local_dets, roi_detections);
            auto tracking_end = std::chrono::high_resolution_clock::now();
            double tracking_time = std::chrono::duration<double, std::milli>(tracking_end - tracking_start).count();
            updateTrackingTime(tracking_time);
            
            // Check if local detection is empty and no active tracks, switch to global detection
            bool no_detections = detections.empty();
            bool no_active_tracks = true;
            for (const auto& track : updated_tracks) {
                if (track && track->is_activated) {
                    no_active_tracks = false;
                    break;
                }
            }
            
            if (no_detections && no_active_tracks) {
                LOG_WARNING("⚠️ Frame " << current_frame_id_ << ": No targets in local detection and no active tracks, switching to global detection (next frame)");
                // Only activate forced global flag, avoid calling update twice in current frame which would break ID continuity
                activateForceGlobalPhase();
                // Use updated_tracks result from local phase for current frame, don't call update again
            } else {
                // Continue normal local phase processing
                handleLocalPhase(updated_tracks, frame.cols, frame.rows);
                
                // Check if there are still ROIs after cleanup
                if (roi_manager_->getROIs().empty()) {
                    activateForceGlobalPhase();
                }
            }
        }
    }
    
    // Check if forced global phase needs to end
    if (detection_mode_ == 1) {
        handleForceGlobalPhaseEnd();
    }
    
    // 🔥 Start data processing time measurement
    auto data_processing_start = std::chrono::high_resolution_clock::now();
    
    // Save current frame tracking results
    auto save_start = std::chrono::high_resolution_clock::now();
    saveFrameResults(updated_tracks);
    auto save_end = std::chrono::high_resolution_clock::now();
    double save_time = std::chrono::duration<double, std::milli>(save_end - save_start).count();
    
    // 🔥 Print key tracking results (bypasses log_level, always visible)
    printKeyTrackingResults(frame_count_, detections, updated_tracks);
    
    // Print tracking information (configurable verbose logging)
    auto print_start = std::chrono::high_resolution_clock::now();
    if (config_.verbose_logging) {  // 🔥 Only print tracking info in verbose logging mode
        printTrackingInfo(updated_tracks);
    }
    auto print_end = std::chrono::high_resolution_clock::now();
    double print_time = std::chrono::duration<double, std::milli>(print_end - print_start).count();
    
    // Update previous frame (prepare for next global detection)
    auto clone_start = std::chrono::high_resolution_clock::now();
    prev_frame_ = frame.clone();
    auto clone_end = std::chrono::high_resolution_clock::now();
    double clone_time = std::chrono::duration<double, std::milli>(clone_end - clone_start).count();
    
    auto data_processing_end = std::chrono::high_resolution_clock::now();
    double data_processing_time = std::chrono::duration<double, std::milli>(data_processing_end - data_processing_start).count();
    updateDataProcessingTime(data_processing_time);
    
    // 🔥 Optimization: Reduce data processing log print frequency
    if (frame_count_ % 100 == 0) {
        LOG_INFO("📊 [Data Processing Time Details] Frame " << frame_count_ << ":");
        LOG_INFO("  Save frame results time: " << std::fixed << std::setprecision(2) << save_time << "ms (" 
                 << std::fixed << std::setprecision(1) << (save_time/data_processing_time*100) << "%)");
        LOG_INFO("  Print tracking info time: " << std::fixed << std::setprecision(2) << print_time << "ms (" 
                 << std::fixed << std::setprecision(1) << (print_time/data_processing_time*100) << "%)");
        LOG_INFO("  Frame clone time: " << std::fixed << std::setprecision(2) << clone_time << "ms (" 
                 << std::fixed << std::setprecision(1) << (clone_time/data_processing_time*100) << "%)");
        LOG_INFO("  Total data processing time: " << std::fixed << std::setprecision(2) << data_processing_time << "ms");
    }
    
    // 🔥 End overall processing time measurement and statistics
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_processing_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    updateProcessingTime(total_processing_time);
    
    // 🔥 Optimization: Reduce time statistics print frequency to lower I/O overhead
    if (frame_count_ % 100 == 0) {
        printTimeStatistics();
    }
    
    // frame_count_++;
    
    return std::make_tuple(detections, std::move(updated_tracks));
}

// 🔥 Phase determination method implementation
bool TrackingDetectionSystem::isGlobalPhase() const {
    // 🔥 Modification: If set to global detection only mode, always return true
    if (detection_mode_ == 0) {
        return true;
    }
    
    // If forced into global phase, return true directly
    if (force_global_phase_) {
        return true;
    }
    
    // Normal cycle determination
    int cycle_length = config_.global_duration + config_.local_duration;  
    int position_in_cycle = frame_count_ % cycle_length;
    bool is_global = position_in_cycle < config_.global_duration;  
    
    return is_global;
}

// Determine if global phase has ended
bool TrackingDetectionSystem::isGlobalPhaseEnd() const {
    if (force_global_phase_) {
        // Forced global phase: must complete config_.global_duration frames of detection
        return (frame_count_ - force_global_start_frame_) >= config_.global_duration - 1;
    }
    
    // Normal cycle determination
    int cycle_length = config_.global_duration + config_.local_duration;
    int next_position = (frame_count_ + 1) % cycle_length;
    return next_position == config_.global_duration;
}

// 🔥 Detection method implementation
std::vector<Detection> TrackingDetectionSystem::globalDetection(const cv::Mat& frame) {
    std::vector<Detection> detections;
    
    if (global_inference_) {
        auto start_time = std::chrono::steady_clock::now();
        
        detections = global_inference_->detect(frame, config_.global_conf_thres);
        
        // Extract appearance features for each detection (if enabled)
        if (config_.enable_appearance && !detections.empty()) {
            auto appearance_start = std::chrono::high_resolution_clock::now();
            int appearance_count = 0;
            
            for (auto& det : detections) {
                // Ensure boundary is within image
                cv::Rect2f bbox = det.bbox;
                int x = std::max(0, static_cast<int>(bbox.x));
                int y = std::max(0, static_cast<int>(bbox.y));
                int width = std::min(static_cast<int>(bbox.width), frame.cols - x);
                int height = std::min(static_cast<int>(bbox.height), frame.rows - y);
                
                // Check boundary validity
                if (width <= 0 || height <= 0) continue;
                
                // Extract target region
                cv::Rect valid_rect(x, y, width, height);
                cv::Mat roi = frame(valid_rect);
                
                cv::Mat appearance;
                cv::resize(roi, appearance, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
                det.appearance = appearance; // Avoid clone
                appearance_count++;
            }
            
            auto appearance_end = std::chrono::high_resolution_clock::now();
            double appearance_time = std::chrono::duration<double, std::milli>(appearance_end - appearance_start).count();
            
            if (frame_count_ % 50 == 0 && appearance_count > 0) {
                LOG_INFO("📸 [Global Detection Appearance Extraction] Frame " << frame_count_ << ": " 
                         << appearance_count << " targets, time: " << std::fixed << std::setprecision(2) 
                         << appearance_time << "ms (avg " << std::fixed << std::setprecision(2) 
                         << (appearance_time/appearance_count) << "ms/target)");
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // 🔥 Update inference time statistics
        updateInferenceTime(inference_time);
        
        if (config_.log_inference_details) {
            LOG_INFO("Global detection time: " << std::fixed << std::setprecision(2) << inference_time << "ms, "
                  << "detected " << detections.size() << " targets "
                     << "(threshold: " << config_.global_conf_thres << ")");
        }
        
        // 🔥 New: Immediately output confidence statistics
        if (config_.log_inference_details) {
            if (detections.empty()) {
                LOG_INFO("[ConfStats] No detection in current frame (global_conf_thres=" << std::fixed << std::setprecision(3)
                         << config_.global_conf_thres << ")");
            } else {
                double min_conf = 1.0, max_conf = 0.0, sum_conf = 0.0;
                std::vector<float> conf_list;
                conf_list.reserve(detections.size());
                for (const auto& d : detections) {
                    min_conf = std::min(min_conf, static_cast<double>(d.confidence));
                    max_conf = std::max(max_conf, static_cast<double>(d.confidence));
                    sum_conf += d.confidence;
                    conf_list.push_back(d.confidence);
                }
                double avg_conf = (detections.empty() ? 0.0 : sum_conf / detections.size());
                std::sort(conf_list.begin(), conf_list.end(), std::greater<float>());
                size_t topk = std::min<size_t>(5, conf_list.size());
                std::ostringstream topk_ss;
                topk_ss << "[";
                for (size_t i = 0; i < topk; ++i) {
                    if (i) topk_ss << ", ";
                    topk_ss << std::fixed << std::setprecision(3) << conf_list[i];
                }
                topk_ss << "]";
                LOG_INFO("[ConfStats] detections=" << detections.size()
                         << ", min=" << std::fixed << std::setprecision(3) << min_conf
                         << ", max=" << max_conf
                         << ", avg=" << avg_conf
                         << ", top" << topk << "=" << topk_ss.str() << ")");
            }
        }
    } else {
        LOG_WARNING("Warning: Global inference engine not initialized");
    }
        
        return detections;
    }
    
std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>> 
TrackingDetectionSystem::localDetection(const cv::Mat& frame) {
    auto start_time = std::chrono::steady_clock::now();
    
    const auto& rois = roi_manager_->getROIs();
    if (rois.empty()) {
        // ROI is empty, activate forced global phase
        activateForceGlobalPhase();
        
        auto end_time = std::chrono::steady_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // 🔥 Update inference time statistics
        updateInferenceTime(inference_time);
        
        printDetectionSummary({}, inference_time);
        return std::make_tuple(std::vector<Detection>{}, std::unordered_map<int, std::vector<Detection>>{});
    }
    
    int roi_count = rois.size();
    std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>> result;
    
    // 🔥 Add detailed debug information
    // LOG_INFO("🔍 [LocalDetection] Starting local detection, frame size: [" << frame.cols << " x " << frame.rows << "]");
    // LOG_INFO("🔍 [LocalDetection] Current ROI count: " << roi_count);
    
    // 🔥 Optimized detection strategy selection - based on ROI count
    std::string detection_mode;
    
    // if (roi_count == 1) {
    //     // Single ROI: Use single detection (most efficient)
    //     detection_mode = "Single ROI Detection";
    //     LOG_INFO("🔍 [LocalDetection] Using single ROI detection mode, ROI count: " << roi_count);
    //     result = singleROIDetection(frame, false);
    //     auto end_time = std::chrono::steady_clock::now();
    //     double time_taken = std::chrono::duration<double>(end_time - start_time).count();
    //     detection_stats_.single_times.push_back(time_taken);
    //     if (detection_stats_.single_times.size() > DetectionStats::MAX_STATS) {
    //         detection_stats_.single_times.pop_front();
    //     }
    // } else 
    if (roi_count >= 1 && roi_count <= 10) {
        // 2-10 ROIs: Use batch detection
        detection_mode = "Batch Detection";
        // LOG_INFO("🔍 [LocalDetection] Using batch detection mode, ROI count: " << roi_count);
        result = optimizedBatchDetection(frame, false);
        auto end_time = std::chrono::steady_clock::now();
        double time_taken = std::chrono::duration<double>(end_time - start_time).count();
        detection_stats_.batch_times.push_back(time_taken);
        if (detection_stats_.batch_times.size() > DetectionStats::MAX_STATS) {
            detection_stats_.batch_times.pop_front();
        }
    } else {
        // More than 10 ROIs: Use chunked batch processing, without OpenMP parallelization
        detection_mode = "Chunked Batch Detection";
        int batch_size = std::min(roi_count, detection_stats_.adaptive_batch_size);
        LOG_INFO("🔍 [LocalDetection] Using chunked batch detection mode, ROI count: " << roi_count 
                 << ", batch size: " << batch_size);
        result = chunkedBatchDetection(frame, batch_size, false);
        
        auto end_time = std::chrono::steady_clock::now();
        double time_taken = std::chrono::duration<double>(end_time - start_time).count();
        detection_stats_.batch_times.push_back(time_taken);
        if (detection_stats_.batch_times.size() > DetectionStats::MAX_STATS) {
            detection_stats_.batch_times.pop_front();
        }
        
        // Dynamically adjust batch size
        adjustBatchSize();
    }
    
    // 🔥 Output detection result statistics
    auto final_detections = std::get<0>(result);
    auto roi_detections = std::get<1>(result);
    
    // LOG_INFO("🔍 [LocalDetection] Detection complete - total detections: " << final_detections.size() 
    //          << ", ROIs with detections: " << roi_detections.size());
    
    // 🔥 Detailed statistics for each ROI's detection results
    // for (const auto& [roi_id, detections] : roi_detections) {
    //     if (!detections.empty()) {
    //         LOG_INFO("🔍 [LocalDetection] ROI-" << roi_id << " detected " << detections.size() << " targets");
    //         for (size_t i = 0; i < detections.size(); ++i) {
    //             const auto& det = detections[i];
    //             LOG_INFO("  - Target" << (i+1) << ": confidence=" << std::fixed << std::setprecision(3) 
    //                      << det.confidence << ", position=(" << det.bbox.x << "," << det.bbox.y 
    //                      << "," << det.bbox.width << "," << det.bbox.height << ")");
    //         }
    //     } else {
    //         LOG_INFO("🔍 [LocalDetection] ROI-" << roi_id << " no detection results");
    //     }
    // }
    
    auto end_time = std::chrono::steady_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // 🔥 Update inference time statistics
    updateInferenceTime(inference_time);
    
    printDetectionSummary(final_detections, inference_time);
    
    return result;
}

std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>> 
TrackingDetectionSystem::optimizedBatchDetection(const cv::Mat& frame, bool verbose) {
        std::vector<Detection> all_detections;
        std::unordered_map<int, std::vector<Detection>> roi_detections;
        
    const auto& rois = roi_manager_->getROIs();
    
    if (rois.empty()) {
        return std::make_tuple(all_detections, roi_detections);
    }
    
    // 🔥 Execute ROI detection strategy optimization
    optimizeROIDetectionStrategy();
    
    // 🔥 Precompute all ROI boundaries
    precomputeROIBounds(frame);
    
    // 🔥 Pre-allocate memory - more precise estimation
    int estimated_detections = rois.size() * 3;  // Estimate 3 detections per ROI
    all_detections.reserve(estimated_detections);
    
    // 🔥 Start performance timing
    auto batch_start = std::chrono::high_resolution_clock::now();
    
    // 🔥 Intelligent ROI filtering - skip low priority ROIs
    std::vector<std::pair<int, cv::Mat>> roi_images;
    std::vector<std::pair<int, cv::Point2i>> roi_offsets;
    std::vector<std::pair<int, float>> roi_conf_thresholds;  // Adaptive thresholds
    std::vector<int> skipped_rois;
    
    roi_images.reserve(rois.size());
    roi_offsets.reserve(rois.size());
    roi_conf_thresholds.reserve(rois.size());
    
    // 🔥 Phase 1: Intelligent ROI filtering and batch extraction
    auto extraction_start = std::chrono::high_resolution_clock::now();
    
    // LOG_INFO("🔍 [BatchDetection] Starting batch detection, ROI count: " << rois.size());
    
    for (const auto& [roi_id, roi] : rois) {
        // 🔥 Skip condition check
        if (shouldSkipROIDetection(roi_id, *roi)) {
            skipped_rois.push_back(roi_id);
            continue;
        }
        
        // 🔥 Get adaptive confidence threshold
        float conf_threshold = getAdaptiveConfidenceThreshold(roi_id, *roi);
        
        // 🔥 Extract ROI image
        cv::Rect roi_rect = roi->bbox;  // Use bbox member variable
        auto [x, y, width, height] = validateROIBounds(*roi, frame.cols, frame.rows);
        
        if (width <= 0 || height <= 0) {
            skipped_rois.push_back(roi_id);
            continue;
        }
        
        cv::Rect valid_rect(x, y, width, height);
        cv::Mat roi_image = extractROIOptimized(frame, valid_rect, roi_id);
        
        if (roi_image.empty()) {
            skipped_rois.push_back(roi_id);
            continue;
        }
        
        // 🔥 Add to batch processing list
        roi_images.emplace_back(roi_id, roi_image);
        roi_offsets.emplace_back(roi_id, cv::Point2i(x, y));
        roi_conf_thresholds.emplace_back(roi_id, conf_threshold);
        
        // LOG_INFO("🔍 [BatchDetection] ROI-" << roi_id << " extraction successful, size: [" << roi_image.cols << " x " << roi_image.rows << "]");
    }
    
    auto extraction_end = std::chrono::high_resolution_clock::now();
    double extraction_time = std::chrono::duration<double, std::milli>(extraction_end - extraction_start).count();
    
    // LOG_INFO("🔍 [BatchDetection] ROI extraction complete, valid ROIs: " << roi_images.size() 
    //          << ", skipped ROIs: " << skipped_rois.size() << ", time: " << extraction_time << "ms");
    
    if (roi_images.empty()) {
        LOG_WARNING("⚠️ [BatchDetection] No valid ROIs to process");
        return std::make_tuple(all_detections, roi_detections);
    }
    
    // 🔥 Phase 2: Batch inference decision
    auto inference_start = std::chrono::high_resolution_clock::now();
    bool use_batch_inference = true;
    
    // std::cout << "📊 [Optimized Batch Detection] Total ROIs: " << rois.size() 
    //           << ", Processing ROIs: " << roi_images.size() 
    //           << ", Skipped ROIs: " << skipped_rois.size()
    //           << ", Batch inference supported: " << (use_batch_inference ? "yes" : "no") << std::endl;
    
    // if (roi_images.size() < 2) {
    //     use_batch_inference = false;
    //     LOG_INFO("🔍 [BatchDetection] ROI count less than 2, using serial inference");
    // } else 
    if (!local_inference_) {
        use_batch_inference = false;
        LOG_ERROR("❌ [BatchDetection] Inference engine is null, using serial inference");
    } else {
        bool supports_batch = local_inference_->supportsBatchDetection();
        int max_batch_size = local_inference_->getMaxBatchSize();
        // LOG_INFO("🔍 [BatchDetection] Inference engine supports batch: " << (supports_batch ? "yes" : "no") 
        //          << ", max batch size: " << max_batch_size);
        
        if (supports_batch) {
            // LOG_INFO("🔍 [BatchDetection] Conditions met, using batch inference mode");
        } else {
            LOG_WARNING("❌ [BatchDetection] Engine does not support batch, using serial inference mode");
            use_batch_inference = false;
        }
    }
    
    if (use_batch_inference) {
        // Batch inference mode
        try {
            // Prepare batch images
            std::vector<cv::Mat> batch_images;
            batch_images.reserve(roi_images.size());
            for (const auto& [roi_id, roi_image] : roi_images) {
                batch_images.push_back(roi_image);
            }
            
            // LOG_INFO("🔍 [BatchDetection] Starting batch inference, processing " << batch_images.size() << " ROIs");
            
            // Execute batch inference
            auto batch_start = std::chrono::high_resolution_clock::now();
            auto batch_results = local_inference_->detectBatch(batch_images, config_.local_conf_thres);
            auto batch_end = std::chrono::high_resolution_clock::now();
            
            double batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
            double time_per_roi = batch_time / batch_images.size();
            
            // LOG_INFO("🔍 [BatchDetection] Batch inference complete, total time: " << batch_time << "ms, avg per ROI: " 
            //          << std::fixed << std::setprecision(2) << time_per_roi << "ms");
            
            // 🔥 Process batch results
            for (size_t i = 0; i < roi_images.size() && i < batch_results.size(); ++i) {
                int roi_id = roi_images[i].first;
                cv::Point2i offset = roi_offsets[i].second;
                auto& roi_detections_local = batch_results[i];
                
                bool has_detection = !roi_detections_local.empty();
                
                // LOG_INFO("🔍 [BatchDetection] ROI-" << roi_id << " batch inference results: " 
                //          << roi_detections_local.size() << " detections");
                
                if (has_detection) {
                    // 🔥 Batch coordinate transformation
                    roi_detections[roi_id].reserve(roi_detections_local.size());
                    for (auto& det : roi_detections_local) {
                        // Record information before NMS filtering
                        // LOG_INFO("  - Before NMS: confidence=" << std::fixed << std::setprecision(3) 
                        //          << det.confidence << ", original position=(" << det.bbox.x << "," << det.bbox.y 
                        //          << "," << det.bbox.width << "," << det.bbox.height << ")");
                        
                        // Adjust coordinates
                        det.bbox.x += offset.x;
                        det.bbox.y += offset.y;
                        
                        // Extract appearance features
                        // Ensure boundary is within image
                        cv::Rect2f bbox = det.bbox;
                        int x = std::max(0, static_cast<int>(bbox.x));
                        int y = std::max(0, static_cast<int>(bbox.y));
                        int width = std::min(static_cast<int>(bbox.width), frame.cols - x);
                        int height = std::min(static_cast<int>(bbox.height), frame.rows - y);
                        
                        // Check boundary validity
                        if (width > 0 && height > 0) {
                            // Extract target region
                            cv::Rect valid_rect(x, y, width, height);
                            cv::Mat roi = frame(valid_rect);
                            
                            if (config_.enable_appearance) {
                                cv::Mat appearance;
                                cv::resize(roi, appearance, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
                                det.appearance = appearance;
                            }
                        }
                        
                        all_detections.emplace_back(std::move(det));
                        roi_detections[roi_id].push_back(all_detections.back());
                    }
                    
                    // LOG_INFO("🔍 [BatchDetection] ROI-" << roi_id << " final detections: " 
                    //          << roi_detections[roi_id].size() << " targets");
                    
                    // 🔥 Update ROI status
                    const auto& roi_it = rois.find(roi_id);
                    if (roi_it != rois.end()) {
                        roi_it->second->no_detection_count = 0;
                        
                        // Print ROI region size and detection frame count
                        const auto& roi = roi_it->second;
                        // LOG_INFO("🔍 [ROI Info] ROI-" << roi_id << ": region size=" << roi->bbox.width << "x" << roi->bbox.height << " (area=" << roi->bbox.area() << "), detection frames=" << roi->no_detection_count << ", detected targets=" << roi_detections[roi_id].size());
                    }
                } else {
                    roi_detections[roi_id] = {};
                    // LOG_INFO("🔍 [BatchDetection] ROI-" << roi_id << " no detection results");
                    const auto& roi_it = rois.find(roi_id);
                    if (roi_it != rois.end()) {
                        roi_it->second->no_detection_count++;
                    }
                }
                
                // 🔥 Update ROI detection state
                updateROIDetectionState(roi_id, has_detection, 0.0); // Batch inference time calculated separately
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("❌ [BatchDetection] Batch inference error: " << e.what());
            LOG_INFO("🔄 Automatically falling back to serial inference mode");
            use_batch_inference = false;
        }
    }
    
    if (!use_batch_inference) {
        // Serial inference mode
        LOG_INFO("🔄 Using serial inference to process " << roi_images.size() << " ROIs");
        
        auto serial_start = std::chrono::high_resolution_clock::now();
        double total_detection_time = 0.0;
        
        for (size_t i = 0; i < roi_images.size(); ++i) {
            int roi_id = roi_images[i].first;
            const cv::Mat& roi_image = roi_images[i].second;
            float conf_threshold = roi_conf_thresholds[i].second;
            
            // LOG_INFO("🔍 [SerialDetection] Processing ROI-" << roi_id << ", confidence threshold: " 
            //          << std::fixed << std::setprecision(3) << conf_threshold);
            
            auto detection_start = std::chrono::high_resolution_clock::now();
            auto roi_detections_local = local_inference_->detect(roi_image, conf_threshold);
            auto detection_end = std::chrono::high_resolution_clock::now();
            double detection_time = std::chrono::duration<double, std::milli>(detection_end - detection_start).count();
            total_detection_time += detection_time;
            
            bool has_detection = !roi_detections_local.empty();
            
            // LOG_INFO("🔍 [SerialDetection] ROI-" << roi_id << " serial inference results: " 
            //          << roi_detections_local.size() << " detections, time: " << detection_time << "ms");
            
            if (has_detection) {
                // Get corresponding offset
                cv::Point2i offset = roi_offsets[i].second;
                
                // Coordinate transformation
                roi_detections[roi_id].reserve(roi_detections_local.size());
                for (auto& det : roi_detections_local) {
                    // Record information before NMS filtering
                    // LOG_INFO("  - Before NMS: confidence=" << std::fixed << std::setprecision(3) 
                    //          << det.confidence << ", original position=(" << det.bbox.x << "," << det.bbox.y 
                    //          << "," << det.bbox.width << "," << det.bbox.height << ")");
                    
                    // Adjust coordinates
                    det.bbox.x += offset.x;
                    det.bbox.y += offset.y;
                    
                    // Extract appearance features
                    // Ensure boundary is within image
                    cv::Rect2f bbox = det.bbox;
                    int x = std::max(0, static_cast<int>(bbox.x));
                    int y = std::max(0, static_cast<int>(bbox.y));
                    int width = std::min(static_cast<int>(bbox.width), frame.cols - x);
                    int height = std::min(static_cast<int>(bbox.height), frame.rows - y);
                    
                    // Check boundary validity
                    if (width > 0 && height > 0) {
                        // Extract target region
                        cv::Rect valid_rect(x, y, width, height);
                        cv::Mat roi = frame(valid_rect);
                        
                        // Resize to standard size and store
                        cv::Mat appearance;
                        cv::resize(roi, appearance, cv::Size(64, 64));
                        det.appearance = appearance.clone();
                    }
                    
                    all_detections.emplace_back(std::move(det));
                    roi_detections[roi_id].push_back(all_detections.back());
                }
                
                // LOG_INFO("🔍 [SerialDetection] ROI-" << roi_id << " final detections: " 
                //          << roi_detections[roi_id].size() << " targets");
                
                // Update ROI status
                const auto& roi_it = rois.find(roi_id);
                if (roi_it != rois.end()) {
                    roi_it->second->no_detection_count = 0;
                    
                    // Print ROI region size and detection frame count
                    const auto& roi = roi_it->second;
                    // LOG_INFO("🔍 [ROI Info] ROI-" << roi_id << ": region size=" << roi->bbox.width << "x" << roi->bbox.height << " (area=" << roi->bbox.area() << "), detection frames=" << roi->no_detection_count << ", detected targets=" << roi_detections[roi_id].size());
                }
            } else {
                roi_detections[roi_id] = {};
                // LOG_INFO("🔍 [SerialDetection] ROI-" << roi_id << " no detection results");
                const auto& roi_it = rois.find(roi_id);
                if (roi_it != rois.end()) {
                    roi_it->second->no_detection_count++;
                }
            }
            
            // Update ROI detection state
            updateROIDetectionState(roi_id, has_detection, detection_time);
        }
        
        auto serial_end = std::chrono::high_resolution_clock::now();
        double serial_total_time = std::chrono::duration<double, std::milli>(serial_end - serial_start).count();
        LOG_INFO("🔄 Serial inference complete: total time " << serial_total_time << "ms, avg per ROI: " 
                 << std::fixed << std::setprecision(2) << (serial_total_time / roi_images.size()) << "ms");
    }
    
    auto inference_end = std::chrono::high_resolution_clock::now();
    double total_inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    
    auto batch_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
    
    // Update performance metrics
    updatePerformanceMetrics(extraction_time, total_inference_time, 0.0);
    
    // Output overall performance report
    // std::cout << "📈 [Performance Report] Total time: " << total_time << "ms (extraction:" << extraction_time 
    //           << "ms, inference:" << total_inference_time << "ms), detections:" << all_detections.size() 
    //           << " targets" << std::endl;
    
    return std::make_tuple(all_detections, roi_detections);
}

void TrackingDetectionSystem::handleGlobalPhase(std::vector<std::unique_ptr<STrack>>& tracks,
                                              const std::vector<Detection>& detections,
                                              int frame_width, int frame_height) {
    // 🔥 When using global detection only mode, skip ROI creation and management
    if (detection_mode_ == 0) {
        // Global detection only mode, don't create or manage ROI
        LOG_INFO("Global detection only mode: Skip ROI creation and management");
        return;
    }
    
    // Following is the logic for global + local joint detection mode
    
    // 1. Create ROI for newly confirmed tracks
    for (const auto& track : tracks) {
        if (track && track->is_activated && track->roi_id == -1 && track->tracklet_len >= 3) {
            int roi_id = roi_manager_->createROIForTrack(*track, frame_width, frame_height, current_frame_id_);
            if (roi_id != -1) {
                const_cast<STrack*>(track.get())->roi_id = roi_id;
            }
        }
    }
    
    // 2. Unified ROI management (position update, merge, split, cleanup)
    // 🔥 Start ROI adjustment time measurement
    auto roi_adjustment_start = std::chrono::high_resolution_clock::now();
    
    // ROI dynamic management
    roi_manager_->dynamicROIManagement(tracks, frame_width, frame_height, current_frame_id_);
    
    // Update candidate targets
    std::vector<Detection> detections_outside_roi;
    for (const auto& det : detections) {
        bool is_outside = true;
        cv::Point2f det_center = det.center();
        
        for (const auto& [roi_id, roi] : roi_manager_->getROIs()) {
            if (roi->containsPoint(det_center, 50)) {
                is_outside = false;
                break;
            }
        }
        
        if (is_outside) {
            detections_outside_roi.push_back(det);
        }
    }
    
    updateCandidatesFromDetections(detections_outside_roi, current_frame_id_);
    
    auto roi_adjustment_end = std::chrono::high_resolution_clock::now();
    double roi_adjustment_time = std::chrono::duration<double, std::milli>(roi_adjustment_end - roi_adjustment_start).count();
    updateROIAdjustmentTime(roi_adjustment_time);
    

}

void TrackingDetectionSystem::handleLocalPhase(std::vector<std::unique_ptr<STrack>>& updated_tracks, 
                                              int frame_width, int frame_height) {
    // First check if there are valid ROIs and tracks
    if (roi_manager_->getROIs().empty()) {
        LOG_WARNING("⚠️ Frame " << current_frame_id_ << ": No valid ROI in local phase, forcing switch to global phase");
        activateForceGlobalPhase();
        return;
    }
    
    // Check if there are active tracks
    bool has_active_tracks = false;
    for (auto& track : updated_tracks) {
        if (track && track->is_activated) {
            has_active_tracks = true;
            break;
        }
    }
    
    if (!has_active_tracks) {
        LOG_WARNING("⚠️ Frame " << current_frame_id_ << ": No active tracks in local phase, suggesting switch to global detection for next frame");
        // Don't switch immediately here, switch when no targets and no tracks detected in process method
    }
    
    // Normal local phase ROI management
    // 🔥 Start ROI adjustment time measurement
    auto roi_adjustment_start = std::chrono::high_resolution_clock::now();
    
    // Local phase ROI management
    auto stats = roi_manager_->localPhaseROIManagement(updated_tracks, frame_width, frame_height, current_frame_id_);
    
    // Update ROI status and cleanup inactive ROIs
    roi_manager_->updateROITrackingStatus(updated_tracks, current_frame_id_);
    int removed_count = roi_manager_->cleanupInactiveROIs(current_frame_id_);
    
    auto roi_adjustment_end = std::chrono::high_resolution_clock::now();
    double roi_adjustment_time = std::chrono::duration<double, std::milli>(roi_adjustment_end - roi_adjustment_start).count();
    updateROIAdjustmentTime(roi_adjustment_time);
    
    // Record ROI count change
    num_rois_ = roi_manager_->getROIs().size();
    
    // Check ROI count again, if 0 then switch to global phase
    if (num_rois_ == 0) {
        LOG_WARNING("⚠️ Frame " << current_frame_id_ << ": No valid ROI after local phase ROI management, forcing switch to global phase");
        activateForceGlobalPhase();
    }
}

void TrackingDetectionSystem::ensureTargetsInSafetyZones(std::vector<std::unique_ptr<STrack>>& tracks,
                                                        int frame_width, int frame_height) {
    // This is a proactive check method to ensure all targets are within their ROI's safety zones
    std::unordered_map<int, std::vector<cv::Point2f>> roi_violations;
    
    // Check if each tracking target is within its ROI's safety zone
    for (auto& track : tracks) {
        if (track->is_activated && track->roi_id > 0) {
            const ROI* roi = roi_manager_->getROI(track->roi_id);
            if (roi != nullptr) {
                cv::Point2f track_center = track->center();
                if (!roi->isInSafetyZone(track_center)) {
                    roi_violations[track->roi_id].push_back(track_center);
                }
            }
        }
    }
    
    // Execute forced update for ROIs with violations
    if (!roi_violations.empty()) {
        int force_updated = 0;
        for (const auto& [roi_id, violation_centers] : roi_violations) {
            ROI* roi = roi_manager_->getROI(roi_id);
            if (roi != nullptr) {
                // Collect all tracking target centers in this ROI
                std::vector<cv::Point2f> all_centers;
                for (auto& track : tracks) {
                    if (track->roi_id == roi_id && track->is_activated) {
                        all_centers.push_back(track->center());
                    }
                }
                
                if (!all_centers.empty()) {
                    // Force update ROI, use high adjustment factor to ensure targets are fully contained in safety zone
                    bool updated = roi->adaptiveUpdate(all_centers, frame_width, frame_height, true);
                    if (updated) {
                        force_updated++;
                        roi->last_updated = current_frame_id_;
                        
                        // If multiple ROIs update simultaneously, set emergency merge check flag
                        if (force_updated >= 2) {
                            // Only set emergency merge check when there are many ROIs to avoid over-merging
                            if (roi_manager_->getROIs().size() > 3) {
                                // Remove emergency merge check, change to periodic check every 5 frames
                            }
                        }
                    }
                }
            }
        }
    }
}

cv::Mat TrackingDetectionSystem::visualize(const cv::Mat& frame, const std::vector<Detection>& detections,
                                          const std::vector<std::unique_ptr<STrack>>& tracks) {
    cv::Mat vis_frame = frame.clone();
    
    // Draw detection boxes
    for (const auto& det : detections) {
        cv::Rect det_rect(static_cast<int>(det.bbox.x), static_cast<int>(det.bbox.y),
                         static_cast<int>(det.bbox.width), static_cast<int>(det.bbox.height));
        
        cv::Scalar det_color = cv::Scalar(0, 165, 255);
        cv::rectangle(vis_frame, det_rect, det_color, 2);
        
        std::string conf_text = std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::putText(vis_frame, conf_text, cv::Point(det_rect.x, det_rect.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, det_color, 1);
    }
    
    // Draw tracking boxes
    for (const auto& track : tracks) {
        if (!track) continue;
        
        // Check if track is actually lost
        bool is_test_lost = track->isLost();
        
        // 1. Draw current position of all tracks (regardless of state)
        cv::Point2f current_pos = track->center();
        cv::Rect track_rect(static_cast<int>(track->tlwh.x), static_cast<int>(track->tlwh.y),
                           static_cast<int>(track->tlwh.width), static_cast<int>(track->tlwh.height));
        
        // 🔥 Generate unique color for each ID
        cv::Scalar base_color = generateColorForID(track->displayId());
        cv::Scalar color = adjustColorForState(base_color, track->is_confirmed, track->is_recovered, is_test_lost);
        
        int thickness;
        std::string label;
        
        if (track->is_activated && !is_test_lost) {
            if (track->is_recovered) {
                thickness = 3;  // Recovered tracks use thicker lines
                label = "ID-" + std::to_string(track->displayId()) + "(R)";
            } else if (track->is_confirmed) {
                thickness = 2;  // Confirmed tracks use medium thickness
                label = "ID-" + std::to_string(track->displayId());
            } else {
                thickness = 1;  // Temporary tracks use thin lines
                label = "T" + std::to_string(track->displayId());
            }
            cv::rectangle(vis_frame, track_rect, color, thickness);
            cv::putText(vis_frame, label, cv::Point(track_rect.x, track_rect.y + track_rect.height + 15),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        } else {
            // Inactive or test lost tracks displayed in gray
            color = cv::Scalar(128, 128, 128);
            thickness = 1;
            label = is_test_lost ? "Lost-" + std::to_string(track->displayId()) : "U" + std::to_string(track->displayId());
            cv::rectangle(vis_frame, track_rect, color, thickness);
            cv::putText(vis_frame, label, cv::Point(track_rect.x, track_rect.y + track_rect.height + 15),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }
        
        // 2. Draw historical trajectory lines for all tracks (whether lost or not)
        if (track->position_history.size() > 1) {
            // 🔥 Draw trajectory lines using ID-corresponding color
            cv::Scalar line_color = is_test_lost ? cv::Scalar(128, 128, 128) : base_color;
            int line_thickness = track->is_confirmed ? 2 : 1;
            
            for (size_t i = 1; i < track->position_history.size(); ++i) {
                cv::line(vis_frame, track->position_history[i-1], track->position_history[i], 
                        line_color, line_thickness, cv::LINE_8);
            }
        }
        
        // 3. For lost tracks, draw future 10 frames predicted trajectory
        if (track->isLost() || is_test_lost) {
            auto future_centers = track->predictFutureCenters(10);
            
            if (future_centers.size() > 1) {
                //  Draw predicted trajectory using ID-corresponding color but with dashed line style
                cv::Scalar pred_color = base_color;
                for (size_t i = 1; i < future_centers.size(); ++i) {
                    // Draw dashed line effect
                    cv::Point2f start = future_centers[i-1];
                    cv::Point2f end = future_centers[i];
                    cv::Point2f direction = end - start;
                    float length = cv::norm(direction);
                    
                    if (length > 0) {
                        direction = direction / length;
                        float dash_length = 5.0f;
                        float gap_length = 3.0f;
                        
                        for (float pos = 0; pos < length; pos += dash_length + gap_length) {
                            cv::Point2f dash_start = start + direction * pos;
                            cv::Point2f dash_end = start + direction * std::min(pos + dash_length, length);
                            cv::line(vis_frame, dash_start, dash_end, pred_color, 2, cv::LINE_4);
                        }
                    }
                }
                
                // Draw small circles at the start and end of predicted trajectory
                cv::circle(vis_frame, future_centers[0], 4, pred_color, -1); // Start point
                cv::circle(vis_frame, future_centers.back(), 4, pred_color, -1); // End point
                
                // Add predicted trajectory label
                std::string pred_label = "Pred-" + std::to_string(track->displayId());
                cv::putText(vis_frame, pred_label, future_centers.back(), cv::FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 2);
            }
        }
        
        // 4. Draw small circles at current position of all tracks
        if (track->isLost() || is_test_lost) {
            // 🔥 Lost tracks use ID-corresponding color
            cv::circle(vis_frame, current_pos, 6, base_color, -1);
            std::string lost_label = "Lost-" + std::to_string(track->displayId());
            cv::putText(vis_frame, lost_label, cv::Point(current_pos.x + 8, current_pos.y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, base_color, 1);
        } else {
            //  Active track's current position uses ID-corresponding color
            cv::circle(vis_frame, current_pos, 3, base_color, -1);
        }
    }
    
    // 🔥 Draw ROI and safety zones
    for (const auto& [roi_id, roi] : roi_manager_->getROIs()) {
        // ROI outer boundary - uniformly use yellow
        cv::Scalar roi_color = cv::Scalar(0, 255, 255);  // Uniformly use yellow
        int roi_thickness = roi->is_merged ? 3 : 2;      // Still keep different thickness to distinguish status
        cv::rectangle(vis_frame, roi->bbox, roi_color, roi_thickness);
        
        // 🔥 Safety zone drawing - uniformly use yellow
        cv::Rect safety_bbox = roi->safetyBbox();
        cv::Scalar safe_color = cv::Scalar(0, 255, 255);  // Uniformly use yellow
        drawDashedRectangle(vis_frame, safety_bbox.tl(), safety_bbox.br(), safe_color, 1, 5, 3);
        
        // 🔥 ROI information - display memory count and safety zone status
        int lost_count = 0;
        for (const auto& [track_id, memory] : roi->track_memories) {
            if (memory->lost_duration > 0) lost_count++;
        }
        
        // Check if current targets are within safety zone
        std::vector<cv::Point2f> current_track_centers;
        for (const auto& track : tracks) {
            if (track->roi_id == roi_id && track->is_activated) {
                current_track_centers.push_back(track->center());
            }
        }
        
        auto violations = roi->getSafetyZoneViolations(current_track_centers);
        int unsafe_count = violations.size();
        
        std::string roi_info = "ROI-" + std::to_string(roi_id) + 
                              " (M:" + std::to_string(roi->track_memories.size()) + 
                              ", L:" + std::to_string(lost_count);
        if (unsafe_count > 0) {
            roi_info += ", U:" + std::to_string(unsafe_count) + ")";
        } else {
            roi_info += ")";
        }
        
        cv::putText(vis_frame, roi_info, cv::Point(roi->x(), roi->y() - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1);
    }
    
    // 🔥 Display status information - add ID statistics
    int confirmed_tracks = 0, temp_tracks = 0;
    for (const auto& track : tracks) {
        if (track->is_confirmed) confirmed_tracks++;
        else temp_tracks++;
    }
    
    std::string mode = isGlobalPhase() ? "GLOBAL" : "LOCAL";
    cv::Scalar mode_color = cv::Scalar(255, 0, 0);
    std::string first_line = "Frame: " + std::to_string(frame_count_) + 
                           " | Mode: " + mode ;
    
    cv::putText(vis_frame, first_line, cv::Point(10, 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2);
    
 
    return vis_frame;
}

void TrackingDetectionSystem::drawDashedRectangle(cv::Mat& img, cv::Point pt1, cv::Point pt2, 
                                                cv::Scalar color, int thickness, int dash_length, int gap_length) {
       
        drawDashedLine(img, pt1, cv::Point(pt2.x, pt1.y), color, thickness, dash_length, gap_length);
    
        drawDashedLine(img, cv::Point(pt2.x, pt1.y), pt2, color, thickness, dash_length, gap_length);
 
        drawDashedLine(img, pt2, cv::Point(pt1.x, pt2.y), color, thickness, dash_length, gap_length);
 
        drawDashedLine(img, cv::Point(pt1.x, pt2.y), pt1, color, thickness, dash_length, gap_length);
    }
    
void TrackingDetectionSystem::drawDashedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, 
                                           cv::Scalar color, int thickness, int dash_length, int gap_length) {
    float dx = pt2.x - pt1.x;
    float dy = pt2.y - pt1.y;
    float length = std::sqrt(dx * dx + dy * dy);
    
    if (length == 0) return;
    
    float unit_dx = dx / length;
    float unit_dy = dy / length;
    
    float current_length = 0;
    bool draw = true;
    
    while (current_length < length) {
        float segment_length = draw ? dash_length : gap_length;
        segment_length = std::min(segment_length, length - current_length);
        
        cv::Point start(static_cast<int>(pt1.x + current_length * unit_dx),
                       static_cast<int>(pt1.y + current_length * unit_dy));
        cv::Point end(static_cast<int>(pt1.x + (current_length + segment_length) * unit_dx),
                     static_cast<int>(pt1.y + (current_length + segment_length) * unit_dy));
        
        if (draw) {
            cv::line(img, start, end, color, thickness);
        }
        
        current_length += segment_length;
        draw = !draw;
    }
}


// Save results based on video name
void TrackingDetectionSystem::saveResults(const std::string& save_dir, const std::string& video_name) {

    std::string save_path = save_dir + "/" + video_name + ".txt";
    std::ofstream file(save_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to create result file: " << save_path << std::endl;
        return;
    }
    
    for (const auto& result : tracking_results_) {
        file << result.frame_id << ","
             << result.track_id << ","
             << result.x << ","
             << result.y << ","
             << result.width << ","
             << result.height << ","
             << result.confidence << ","
             << (result.is_confirmed ? 1 : 0) << std::endl;
    }
    
    // Use LOG_INFO instead of std::cout to avoid duplicate output
    // The main summary is printed in MultiThreadTrackingApp
}

void TrackingDetectionSystem::setTotalFrames(int total_frames) {
    total_frames_ = total_frames;
    std::cout << "Set total frames: " << total_frames_ << std::endl;
}

void TrackingDetectionSystem::updateTrackROIAssociations(std::vector<std::unique_ptr<STrack>>& tracks) {
    for (auto& track : tracks) {
        if (track && track->is_activated && track->roi_id == -1) {
            auto best_roi_id = roi_manager_->findROIIdByPoint(track->center(), 30);
            if (best_roi_id) {
                track->roi_id = *best_roi_id;

                ROI* roi = roi_manager_->getROI(*best_roi_id);
                if (roi && std::find(roi->track_ids.begin(), roi->track_ids.end(), track->displayId()) == roi->track_ids.end()) {
                    roi->track_ids.push_back(track->displayId());
                    // std::cout << "Update association: Track-" << track->displayId() << " -> ROI-" << *best_roi_id << std::endl;
                }
            }
        }
    }
}

void TrackingDetectionSystem::finalizeGlobalPhase(int frame_width, int frame_height) {

    if (detection_mode_ == 0) {
        // Only global detection mode, no ROI processing
        return;
    }
    

    // std::cout << "End global phase, prepare to enter local phase" << std::endl;
    
    // Clean up inactive ROIs
    int cleaned = roi_manager_->cleanupInactiveROIs(current_frame_id_);
    // if (cleaned > 0) {
    //     std::cout << "Global phase ends, cleaned " << cleaned << " inactive ROIs" << std::endl;
    // }
    
    // Additional cleanup or preparation when ending global phase
    auto& active_tracks = tracker_->getActiveTracks();
    
    // ROI adjustment time start
    auto roi_adjustment_start = std::chrono::high_resolution_clock::now();
    
    // ROI status update
    roi_manager_->updateROITrackingStatus(active_tracks, current_frame_id_);
    roi_manager_->updateROITrackMemories(active_tracks, current_frame_id_);
    
    // Ensure all ROI positions are up to date
    roi_manager_->updateROIPositions(active_tracks, frame_width, frame_height);
    
    auto roi_adjustment_end = std::chrono::high_resolution_clock::now();
    double roi_adjustment_time = std::chrono::duration<double, std::milli>(roi_adjustment_end - roi_adjustment_start).count();
    updateROIAdjustmentTime(roi_adjustment_time);
    
    // std::cout << "Global phase ends, currently there are " << active_tracks.size() << " active tracks and " 
    //           << roi_manager_->getROIs().size() << " ROIs" << std::endl;
}

// Support for global detection with double frame detection
std::vector<Detection> TrackingDetectionSystem::globalDetectionWithMotion(const cv::Mat& frame) {
    auto start_time = std::chrono::steady_clock::now();
    std::vector<Detection> detections;
    
    if (global_inference_) {
        // Check if there is a previous frame available for double frame detection
        if (!prev_frame_.empty() && prev_frame_.size() == frame.size()) {
            try {
                // Use TensorRTGlobalInference's double frame detection interface
                // Note: Here we need to convert global_inference_ to TensorRTGlobalInference type
                // But due to interface design, we directly call detectWithPreviousFrame method
                auto* global_inference_ptr = dynamic_cast<TensorRTGlobalInference*>(global_inference_.get());
                if (global_inference_ptr) {
                    // pass current frame number to support preprocessing cache
                    detections = global_inference_ptr->detectWithPreviousFrame(frame, prev_frame_, config_.global_conf_thres, current_frame_id_);
                } else {
                    // If not TensorRTGlobalInference type, fall back to single frame detection
                    detections = global_inference_->detect(frame, config_.global_conf_thres);
                }
            } catch (const std::exception& e) {
                std::cout << "⚠️ [Global detection] Double frame detection failed: " << e.what() << ", fall back to single frame detection" << std::endl;
                detections = global_inference_->detect(frame, config_.global_conf_thres);
            }
        } else {
            // Single frame detection (first frame or frame size mismatch)
            if (prev_frame_.empty()) {
                // First frame, use black frame as previous frame
                cv::Mat black_frame = cv::Mat::zeros(frame.size(), frame.type());
                auto* global_inference_ptr = dynamic_cast<TensorRTGlobalInference*>(global_inference_.get());
                if (global_inference_ptr) {
                    // First frame uses black frame, pass current frame number (cache is not available, will be reprocessed)
                    detections = global_inference_ptr->detectWithPreviousFrame(frame, black_frame, config_.global_conf_thres, current_frame_id_);
                } else {
                    detections = global_inference_->detect(frame, config_.global_conf_thres);
                }
            } else {
                // Frame size mismatch
                std::cout << "⚠️ [Global detection] Frame size mismatch, use single frame detection" << std::endl;
                detections = global_inference_->detect(frame, config_.global_conf_thres);
            }
        }
        
        // Extract appearance features for each detection (if enabled)
        if (config_.enable_appearance && !detections.empty()) {
            auto appearance_start = std::chrono::high_resolution_clock::now();
            int appearance_count = 0;
            
            for (auto& det : detections) {
                // Ensure boundary is within image
                cv::Rect2f bbox = det.bbox;
                int x = std::max(0, static_cast<int>(bbox.x));
                int y = std::max(0, static_cast<int>(bbox.y));
                int width = std::min(static_cast<int>(bbox.width), frame.cols - x);
                int height = std::min(static_cast<int>(bbox.height), frame.rows - y);
                
                // Check boundary validity
                if (width <= 0 || height <= 0) continue;
                
                // Extract target region
                cv::Rect valid_rect(x, y, width, height);
                cv::Mat roi = frame(valid_rect);
                
                cv::Mat appearance;
                cv::resize(roi, appearance, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
                det.appearance = appearance;
                appearance_count++;
            }
            
            auto appearance_end = std::chrono::high_resolution_clock::now();
            double appearance_time = std::chrono::duration<double, std::milli>(appearance_end - appearance_start).count();
            
            if (frame_count_ % 50 == 0 && appearance_count > 0) {
                LOG_INFO("📸 [Global motion detection appearance extraction] Frame " << frame_count_ << ": " 
                         << appearance_count << " targets, time: " << std::fixed << std::setprecision(2) 
                         << appearance_time << "ms (average" << std::fixed << std::setprecision(2) 
                         << (appearance_time/appearance_count) << "ms/target)");
            }
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Update inference time statistics
    updateInferenceTime(inference_time);
    
    // Print detailed detection summary information
    printGlobalDetectionSummary(detections, inference_time, config_.global_conf_thres);
    
    // Immediately output confidence statistics
    if (config_.log_inference_details) {
        if (detections.empty()) {
            LOG_INFO("[ConfStats] No detections in this frame (global_conf_thres=" << std::fixed << std::setprecision(3)
                     << config_.global_conf_thres << ")");
        } else {
            double min_conf = 1.0, max_conf = 0.0, sum_conf = 0.0;
            std::vector<float> conf_list;
            conf_list.reserve(detections.size());
            for (const auto& d : detections) {
                min_conf = std::min(min_conf, static_cast<double>(d.confidence));
                max_conf = std::max(max_conf, static_cast<double>(d.confidence));
                sum_conf += d.confidence;
                conf_list.push_back(d.confidence);
            }
            double avg_conf = (detections.empty() ? 0.0 : sum_conf / detections.size());
            std::sort(conf_list.begin(), conf_list.end(), std::greater<float>());
            size_t topk = std::min<size_t>(5, conf_list.size());
            std::ostringstream topk_ss;
            topk_ss << "[";
            for (size_t i = 0; i < topk; ++i) {
                if (i) topk_ss << ", ";
                topk_ss << std::fixed << std::setprecision(3) << conf_list[i];
            }
            topk_ss << "]";
            LOG_INFO("[ConfStats] detections=" << detections.size()
                     << ", min=" << std::fixed << std::setprecision(3) << min_conf
                     << ", max=" << max_conf
                     << ", avg=" << avg_conf
                     << ", top" << topk << "=" << topk_ss.str() << ")");
        }
    }
    
    return detections;
}

// Print global detection summary information
void TrackingDetectionSystem::printGlobalDetectionSummary(const std::vector<Detection>& detections, 
                                                         double inference_time, float threshold) {
    // Update GPU metrics
    updateGPUMetrics();
    
    // Count detection results
    std::unordered_map<int, int> class_counts;
    for (const auto& det : detections) {
        class_counts[det.class_id]++;
    }
    
    // Build detection result string
    std::string detection_str;
    if (detections.empty()) {
        detection_str = "(no detections)";
    } else {
        std::vector<std::string> detection_parts;
        for (const auto& [class_id, count] : class_counts) {
            std::string class_name = (class_id == 0) ? "drone" : ("class_" + std::to_string(class_id));
            if (count == 1) {
                detection_parts.push_back("1 " + class_name);
            } else {
                detection_parts.push_back(std::to_string(count) + " " + class_name + "s");
            }
        }
        
        if (!detection_parts.empty()) {
            detection_str = detection_parts[0];
            for (size_t i = 1; i < detection_parts.size(); ++i) {
                detection_str += ", " + detection_parts[i];
            }
        }
    }
    
    // Build GPU information string
    std::string gpu_info;
    if (perf_metrics_.gpu_monitoring_available) {
        gpu_info = ", GPU: " + std::to_string(static_cast<int>(perf_metrics_.gpu_utilization)) + "%";
    }
    
    // Output formatted information
    // std::string total_frames_str = (total_frames_ > 0) ? std::to_string(total_frames_) : "unknown";
    // std::cout << "video 1/1 (frame " << frame_count_ << "/" << total_frames_str 
    //           << ") input_video: 640x640 " << detection_str 
    //           << ", " << std::fixed << std::setprecision(1) << inference_time << "ms" 
    //           << gpu_info << std::endl;
}

// 🔥 Print key tracking results (bypasses log_level, always visible)
void TrackingDetectionSystem::printKeyTrackingResults(int frame_id, const std::vector<Detection>& detections,
                                                      const std::vector<std::unique_ptr<STrack>>& tracks) {
    // Use std::cout instead of LOG_INFO to bypass log_level filtering
    // This ensures output is visible even when log_level=2 (ERROR)
    
    std::ostringstream oss;
    oss << "Frame: " << frame_id;
    
    // Add mode info (changed from "Phase" to "Mode")
    std::string mode = isGlobalPhase() ? "GLOBAL" : "LOCAL";
    oss << ", Mode: " << mode;
    
    // Add detection count
    oss << ", Detections: " << detections.size();
    
    // 🔥 Enhanced: Count tracks by state
    std::vector<int> active_track_ids;
    std::vector<int> recovered_track_ids;
    std::vector<int> temp_track_ids;
    int lost_tracks = 0;
    
    for (const auto& track : tracks) {
        if (track && track->is_activated) {
            if (track->is_confirmed) {
                active_track_ids.push_back(track->displayId());
                if (track->is_recovered) {
                    recovered_track_ids.push_back(track->displayId());
                }
            } else {
                temp_track_ids.push_back(track->displayId());
            }
        }
        if (track && track->isLost()) {
            lost_tracks++;
        }
    }
    
    oss << ", Tracks: " << active_track_ids.size();
    if (!temp_track_ids.empty()) {
        oss << " (+" << temp_track_ids.size() << " temp)";
    }
    if (lost_tracks > 0) {
        oss << " (Lost: " << lost_tracks << ")";
    }
    
    // 🔥 Print recovered tracks if any
    if (!recovered_track_ids.empty()) {
        oss << " [RECOVERED: ";
        for (size_t i = 0; i < recovered_track_ids.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "ID" << recovered_track_ids[i];
        }
        oss << "]";
    }
    
    // Print track details if there are tracks
    if (!active_track_ids.empty()) {
        oss << " | Targets: ";
        
        // Sort track IDs for consistent output
        std::sort(active_track_ids.begin(), active_track_ids.end());
        
        bool first = true;
        for (int track_id : active_track_ids) {
            // Find the track
            const STrack* track_ptr = nullptr;
            for (const auto& track : tracks) {
                if (track && track->displayId() == track_id) {
                    track_ptr = track.get();
                    break;
                }
            }
            
            if (track_ptr) {
                if (!first) oss << ", ";
                first = false;
                
                // Format: ID[x,y,w,h] with recovery marker
                oss << "ID" << track_id;
                if (track_ptr->is_recovered) oss << "(R)";  // Mark recovered tracks
                oss << "["
                    << std::fixed << std::setprecision(0)
                    << track_ptr->tlwh.x << ","
                    << track_ptr->tlwh.y << ","
                    << track_ptr->tlwh.width << ","
                    << track_ptr->tlwh.height << "]";
            }
        }
    } else {
        oss << " | Targets: None";
    }
    
    // 🔥 Print temporary tracks if any
    if (!temp_track_ids.empty()) {
        oss << " | Temp: ";
        std::sort(temp_track_ids.begin(), temp_track_ids.end());
        for (size_t i = 0; i < temp_track_ids.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "ID" << temp_track_ids[i];
        }
    }
    
    // Output to console (bypasses log filtering)
    std::cout << oss.str() << std::endl;
}

void TrackingDetectionSystem::printTrackingInfo(const std::vector<std::unique_ptr<STrack>>& tracks) {
    LOG_INFO("=== Tracking information ===");
    LOG_INFO("Total tracks: " << tracks.size());
    
    // Count different types of tracks
    int confirmed_tracks = 0;
    int temp_tracks = 0;
    int lost_tracks = 0;
    int recovered_tracks = 0;
    int activated_tracks = 0;
    
    for (const auto& track : tracks) {
        if (!track) continue;
        
        if (track->is_confirmed) confirmed_tracks++;
        if (track->is_activated && !track->is_confirmed) temp_tracks++;
        if (track->isLost()) lost_tracks++;
        if (track->is_recovered) recovered_tracks++;
        if (track->is_activated) activated_tracks++;
    }
    
    // Get lost tracks information from tracker
    if (tracker_) {
        const auto& lost_tracks_from_tracker = tracker_->getLostTracks();
        LOG_INFO("Lost tracks in tracker: " << lost_tracks_from_tracker.size());
        
        // Update lost tracks count
        lost_tracks += lost_tracks_from_tracker.size();
        
        // Get all active tracks (including unconfirmed)
        const auto& all_active_tracks = tracker_->getActiveTracks();
        LOG_INFO("Total active tracks in tracker: " << all_active_tracks.size());
        
        // Count unconfirmed tracks
        int unconfirmed_tracks = 0;
        for (const auto& track : all_active_tracks) {
            if (track && track->is_activated && !track->is_confirmed) {
                unconfirmed_tracks++;
            }
        }
        LOG_INFO("Unconfirmed tracks in tracker: " << unconfirmed_tracks);
        
        // Debug information: display all track IDs
        std::string active_ids = "Active track IDs: ";
        for (const auto& track : all_active_tracks) {
            if (track) {
                active_ids += std::to_string(track->displayId()) + " ";
            }
        }
        LOG_INFO(active_ids);
        
        std::string lost_ids = "Lost track IDs: ";
        for (const auto& track : lost_tracks_from_tracker) {
            if (track) {
                lost_ids += std::to_string(track->displayId()) + " ";
            }
        }
        LOG_INFO(lost_ids);
    }
    
    LOG_INFO("Confirmed tracks: " << confirmed_tracks);
    LOG_INFO("Temporary tracks: " << temp_tracks);
    LOG_INFO("Lost tracks: " << lost_tracks);
    LOG_INFO("Recovered tracks: " << recovered_tracks);
    LOG_INFO("Activated tracks: " << activated_tracks);
    
    LOG_INFO("--- Detailed track information ---");
    
    // First display active tracks
    for (const auto& track : tracks) {
        if (!track) continue;
        
        LOG_INFO("Track ID-" << track->displayId() << ":");
        
        if (track->isLost()) {
            LOG_INFO("  State: Lost (lost " << track->lost_frames_count << " frames)");
        } else if (track->is_recovered) {
            LOG_INFO("  State: Recovered");
        } else if (track->is_confirmed) {
            LOG_INFO("  State: Confirmed");
        } else if (track->is_activated) {
            LOG_INFO("  State: Temporary");
        } else {
            LOG_INFO("  State: New created");
        }     
        
        LOG_INFO("  Position: (" << std::fixed << std::setprecision(1) 
                  << track->tlwh.x << ", " << track->tlwh.y << ", "
                  << track->tlwh.width << ", " << track->tlwh.height << ")");
        
        LOG_INFO("  Center point: (" << std::fixed << std::setprecision(1)
                  << track->center().x << ", " << track->center().y << ")");
        
        LOG_INFO("  Confidence: " << std::fixed << std::setprecision(3) << track->score);
        
        LOG_INFO("  Track length: " << track->tracklet_len << " frames");
        
        LOG_INFO("  Start frame: " << track->start_frame);
        
        if (track->roi_id >= 0) {
            LOG_INFO("  Associated ROI: " << track->roi_id);
        }
        
        if (track->is_confirmed) {
            LOG_INFO("  Confirmation frames: " << track->confirmation_frames << "/" << track->min_confirmation_frames);
        }
        
        if (track->is_recovered) {
            LOG_INFO("  Recovery confidence: " << std::fixed << std::setprecision(3) << track->recovery_confidence);
        }
        
        if (track->isLost()) {
            LOG_INFO("  Lost duration: " << track->lost_frames_count << " frames");
        }
        
        LOG_INFO("  Quality score: " << std::fixed << std::setprecision(3) << track->quality_score);
        
        if (!track->position_history.empty()) {
            LOG_INFO("  History track points: " << track->position_history.size() << " points");
        }
        
        // Display prediction information (if it is a lost track)
        if (track->isLost()) {
            auto future_centers = track->predictFutureCenters(5);
            if (!future_centers.empty()) {
                std::stringstream pred_ss;
                pred_ss << "  Predicted trajectory: ";
                for (size_t i = 0; i < std::min(future_centers.size(), size_t(3)); ++i) {
                    pred_ss << "(" << std::fixed << std::setprecision(1) << future_centers[i].x << ", " << future_centers[i].y << ")";
                    if (i < std::min(future_centers.size(), size_t(3)) - 1) pred_ss << " -> ";
                }
                if (future_centers.size() > 3) pred_ss << " ...";
                LOG_INFO(pred_ss.str());
            }
        }
    }
    
    // Then display lost tracks
    if (tracker_) {
        const auto& lost_tracks_from_tracker = tracker_->getLostTracks();
        if (!lost_tracks_from_tracker.empty()) {
            LOG_INFO("--- Lost track detailed information ---");
            for (const auto& track : lost_tracks_from_tracker) {
                if (!track) continue;
                
                LOG_INFO("Lost track ID-" << track->displayId() << ":");
                LOG_INFO("  State: Lost (lost " << track->lost_frames_count << " frames)");
                LOG_INFO("  Position: (" << std::fixed << std::setprecision(1) 
                          << track->tlwh.x << ", " << track->tlwh.y << ", "
                          << track->tlwh.width << ", " << track->tlwh.height << ")");
                LOG_INFO("  Center point: (" << std::fixed << std::setprecision(1)
                          << track->center().x << ", " << track->center().y << ")");
                LOG_INFO("  Confidence: " << std::fixed << std::setprecision(3) << track->score);
                LOG_INFO("  Track length: " << track->tracklet_len << " frames");
                LOG_INFO("  Start frame: " << track->start_frame);
                
                if (track->roi_id >= 0) {
                    LOG_INFO("  Associated ROI: " << track->roi_id);
                }
                
                LOG_INFO("  Quality score: " << std::fixed << std::setprecision(3) << track->quality_score);
                
                if (!track->position_history.empty()) {
                    LOG_INFO("  History track points: " << track->position_history.size() << " points");
                }
                
                // Display prediction information
                auto future_centers = track->predictFutureCenters(5);
                if (!future_centers.empty()) {
                    std::stringstream pred_ss;
                    pred_ss << "  Predicted trajectory: ";
                    for (size_t i = 0; i < std::min(future_centers.size(), size_t(3)); ++i) {
                        pred_ss << "(" << std::fixed << std::setprecision(1) << future_centers[i].x << ", " << future_centers[i].y << ")";
                        if (i < std::min(future_centers.size(), size_t(3)) - 1) pred_ss << " -> ";
                    }
                    if (future_centers.size() > 3) pred_ss << " ...";
                    LOG_INFO(pred_ss.str());
                }
            }
        }
        
        // Display all active tracks in tracker (including unconfirmed)
        const auto& all_active_tracks = tracker_->getActiveTracks();
        if (!all_active_tracks.empty()) {
            LOG_INFO("--- All active tracks in tracker ---");
            for (const auto& track : all_active_tracks) {
                if (!track) continue;
                
                LOG_INFO("Active track ID-" << track->displayId() << ":");
                LOG_INFO("  Confirmation state: " << (track->is_confirmed ? "Confirmed" : "Unconfirmed"));
                LOG_INFO("  Confirmation frames: " << track->confirmation_frames << "/" << track->min_confirmation_frames);
                LOG_INFO("  Position: (" << std::fixed << std::setprecision(1) 
                          << track->tlwh.x << ", " << track->tlwh.y << ", "
                          << track->tlwh.width << ", " << track->tlwh.height << ")");
                LOG_INFO("  Center point: (" << std::fixed << std::setprecision(1)
                          << track->center().x << ", " << track->center().y << ")");
                LOG_INFO("  Confidence: " << std::fixed << std::setprecision(3) << track->score);
                LOG_INFO("  Track length: " << track->tracklet_len << " frames");
                LOG_INFO("  Start frame: " << track->start_frame);
                
                if (track->roi_id >= 0) {
                    LOG_INFO("  Associated ROI: " << track->roi_id);
                }
                
                LOG_INFO("  Quality score: " << std::fixed << std::setprecision(3) << track->quality_score);
                
                if (!track->position_history.empty()) {
                    LOG_INFO("  History track points: " << track->position_history.size() << " points");
                }
            }
        }
    }
    
    LOG_INFO("--- ROI statistics information ---");
    for (const auto& [roi_id, roi] : roi_manager_->getROIs()) {
        LOG_INFO("ROI-" << roi_id << ":");
        LOG_INFO("  Position: (" << roi->x() << ", " << roi->y() << ", " 
                  << roi->width() << ", " << roi->height() << ")");
        LOG_INFO("  Memory count: " << roi->track_memories.size());
        
        int lost_memories = 0;
        for (const auto& [track_id, memory] : roi->track_memories) {
            if (memory->lost_duration > 0) lost_memories++;
        }
        LOG_INFO("  Lost memory: " << lost_memories);
        
        // 统计当前ROI中的轨迹
        int roi_tracks = 0;
        for (const auto& track : tracks) {
            if (track && track->roi_id == roi_id) roi_tracks++;
        }
        LOG_INFO("  Current tracks: " << roi_tracks);
    }
    
    LOG_INFO("====================");
}

// Print detection summary information
void TrackingDetectionSystem::printDetectionSummary(const std::vector<Detection>& detections, double inference_time) {
    // Update GPU metrics
    updateGPUMetrics();
    
    // Count detections by class
    std::unordered_map<int, int> class_counts;
    for (const auto& det : detections) {
        class_counts[det.class_id]++;
    }
    
    // Build detection result string
    std::string detection_str;
    if (detections.empty()) {
        detection_str = "(no detections)";
    } else {
        std::vector<std::string> detection_parts;
        for (const auto& [class_id, count] : class_counts) {
            std::string class_name = (class_id == 0) ? "drone" : ("class_" + std::to_string(class_id));
            if (count == 1) {
                detection_parts.push_back("1 " + class_name);
            } else {
                detection_parts.push_back(std::to_string(count) + " " + class_name + "s");
            }
        }
        
        if (!detection_parts.empty()) {
            detection_str = detection_parts[0];
            for (size_t i = 1; i < detection_parts.size(); ++i) {
                detection_str += ", " + detection_parts[i];
            }
        }
    }
    
    // Print confidence statistics (for diagnosing INT8 threshold issues)
    if (config_.log_inference_details) {
        if (detections.empty()) {
            LOG_INFO("[ConfStats] No detections in this frame (global_conf_thres=" << std::fixed << std::setprecision(3) 
                     << config_.global_conf_thres << ", local_conf_thres=" << config_.local_conf_thres << ")");
        } else {
            double min_conf = 1.0, max_conf = 0.0, sum_conf = 0.0;
            std::vector<float> conf_list;
            conf_list.reserve(detections.size());
            for (const auto& d : detections) {
                min_conf = std::min(min_conf, static_cast<double>(d.confidence));
                max_conf = std::max(max_conf, static_cast<double>(d.confidence));
                sum_conf += d.confidence;
                conf_list.push_back(d.confidence);
            }
            double avg_conf = (detections.empty() ? 0.0 : sum_conf / detections.size());
            std::sort(conf_list.begin(), conf_list.end(), std::greater<float>());
            size_t topk = std::min<size_t>(5, conf_list.size());
            std::ostringstream topk_ss;
            topk_ss << "[";
            for (size_t i = 0; i < topk; ++i) {
                if (i) topk_ss << ", ";
                topk_ss << std::fixed << std::setprecision(3) << conf_list[i];
            }
            topk_ss << "]";
            LOG_INFO("[ConfStats] detections=" << detections.size()
                     << ", min=" << std::fixed << std::setprecision(3) << min_conf
                     << ", max=" << max_conf
                     << ", avg=" << avg_conf
                     << ", top" << topk << "=" << topk_ss.str()
                     << " (global_thres=" << config_.global_conf_thres
                     << ", local_thres=" << config_.local_conf_thres << ")");
        }
    }
    
    // Build GPU information string
    std::string gpu_info;
    if (perf_metrics_.gpu_monitoring_available) {
        gpu_info = ", GPU: " + std::to_string(static_cast<int>(perf_metrics_.gpu_utilization)) + "%";
    }
    
    // Output formatted information
    // std::string total_frames_str = (total_frames_ > 0) ? std::to_string(total_frames_) : "unknown";
    // std::cout << "video 1/1 (frame " << frame_count_ << "/" << total_frames_str 
    //           << ") input_video: 640x640 " << detection_str 
    //           << ", " << std::fixed << std::setprecision(1) << inference_time << "ms" 
    //           << gpu_info << std::endl;
}

// Chunked batch detection
std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>>
TrackingDetectionSystem::chunkedBatchDetection(const cv::Mat& frame, int chunk_size, bool verbose) {
    std::vector<Detection> all_tracking_detections;
    std::unordered_map<int, std::vector<Detection>> all_roi_detections;
    
    const auto& rois = roi_manager_->getROIs();
    
    // Fix data structure to avoid pointer type issues
    std::vector<std::pair<int, const ROI*>> roi_items;
    for (const auto& [roi_id, roi_ptr] : rois) {
        roi_items.emplace_back(roi_id, roi_ptr.get());
    }
    
    // Chunk processing ROI
    for (size_t i = 0; i < roi_items.size(); i += chunk_size) {
        size_t end_idx = std::min(i + chunk_size, roi_items.size());
        
        // Process current chunk
        for (size_t j = i; j < end_idx; ++j) {
            int roi_id = roi_items[j].first;
            const ROI* roi = roi_items[j].second;
            
            if (!roi) continue;  // Safe check
            
            // Get ROI area and perform detection
            cv::Rect roi_rect = roi->bbox;
            roi_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
            
            if (roi_rect.width > 0 && roi_rect.height > 0 && local_inference_) {
                cv::Mat roi_image = frame(roi_rect);
                auto roi_detections_local = local_inference_->detect(roi_image, config_.local_conf_thres);
                
                // Convert coordinates and add to results
                for (auto& det : roi_detections_local) {
                    det.bbox.x += roi_rect.x;
                    det.bbox.y += roi_rect.y;
                    all_tracking_detections.push_back(det);
                    all_roi_detections[roi_id].push_back(det);
                }
                
                if (verbose && !roi_detections_local.empty()) {
                    std::cout << "ROI " << roi_id << " detected " << roi_detections_local.size() 
                              << " targets (position: " << roi_rect.x << "," << roi_rect.y 
                              << " size: " << roi_rect.width << "x" << roi_rect.height << ")" << std::endl;
                }
            }
        }
    }
    
    return std::make_tuple(all_tracking_detections, all_roi_detections);
}

// Single ROI detection
std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>>
TrackingDetectionSystem::singleROIDetection(const cv::Mat& frame, bool verbose) {
    std::vector<Detection> tracking_detections;
    std::unordered_map<int, std::vector<Detection>> roi_detections;
    
    for (const auto& [roi_id, roi] : roi_manager_->getROIs()) {
        auto [x, y, w, h] = validateROIBounds(*roi, frame.cols, frame.rows);
        
        if (w <= 0 || h <= 0) {
            roi_detections[roi_id] = {};
            continue;
        }
        
        cv::Mat roi_frame = frame(cv::Rect(x, y, w, h));
        if (local_inference_) {
            auto results = local_inference_->detect(roi_frame, config_.local_conf_thres);
            
            // Convert coordinates
            for (auto& det : results) {
                det.bbox.x += x;
                det.bbox.y += y;
                tracking_detections.push_back(det);
                roi_detections[roi_id].push_back(det);
            }
        }
    }
    
    return std::make_tuple(tracking_detections, roi_detections);
}

// Dynamic adjust batch size
void TrackingDetectionSystem::adjustBatchSize() {
    if (detection_stats_.single_times.size() >= 3 && detection_stats_.batch_times.size() >= 3) {
        double avg_single = std::accumulate(detection_stats_.single_times.begin(), 
                                          detection_stats_.single_times.end(), 0.0) / 
                          detection_stats_.single_times.size();
        double avg_batch = std::accumulate(detection_stats_.batch_times.begin(), 
                                         detection_stats_.batch_times.end(), 0.0) / 
                         detection_stats_.batch_times.size();
        
        // If batch detection is more efficient, increase batch size
        if (avg_batch < avg_single * 0.8) {
            detection_stats_.adaptive_batch_size = std::min(16, detection_stats_.adaptive_batch_size + 1);
        }
        // If batch detection is less efficient, decrease batch size
        else if (avg_batch > avg_single * 1.2) {
            detection_stats_.adaptive_batch_size = std::max(2, detection_stats_.adaptive_batch_size - 1);
        }
    }
}

// Validate ROI bounds
std::tuple<int, int, int, int> TrackingDetectionSystem::validateROIBounds(const ROI& roi, int frame_width, int frame_height) {
    int x = std::max(0, std::min(roi.x(), frame_width - roi.width()));
    int y = std::max(0, std::min(roi.y(), frame_height - roi.height()));
    int w = std::min(roi.width(), frame_width - x);
    int h = std::min(roi.height(), frame_height - y);
    return std::make_tuple(x, y, w, h);
}

// Update candidates from detections
void TrackingDetectionSystem::updateCandidatesFromDetections(const std::vector<Detection>& detections_outside_roi, int frame_id) {
    // Here we need to call the candidate target update method of the ROI manager
    roi_manager_->updateCandidates(detections_outside_roi, frame_id);
}

// ROI-track association optimization
void TrackingDetectionSystem::optimizeROITrackAssociation(const std::vector<std::unique_ptr<STrack>>& tracks,
                                                        const std::vector<Detection>& detections) {
    if (detections.empty()) return;
    
    // Find the best matching global detection for each track in the ROI
    for (const auto& track : tracks) {
        if (!track || track->roi_id <= 0) continue;
        
        const ROI* roi = roi_manager_->getROI(track->roi_id);
        if (!roi) continue;
        
        cv::Point2f track_center = track->center();
        
        // Find the nearest global detection to the track target
        const Detection* best_detection = nullptr;
        float min_distance = std::numeric_limits<float>::max();
        
        for (const auto& detection : detections) {
            cv::Point2f det_center = detection.center();
            float distance = cv::norm(track_center - det_center);
            
            // Check if it is within a reasonable range
            const float max_movement = 100.0f;
            if (distance < max_movement && distance < min_distance) {
                min_distance = distance;
                best_detection = &detection;
            }
        }
        
        // If a suitable detection is found, optimize ROI position
        if (best_detection) {
            cv::Point2f det_center = best_detection->center();
            
            // Check if it is outside the current ROI but within the extended area
            if (!roi->containsPoint(det_center) && roi->containsPoint(det_center, 150)) {
                // Smoothly adjust ROI position
                cv::Point2f roi_center = roi->center();
                const float adjust_factor = 0.3f;
                float new_center_x = roi_center.x + (det_center.x - roi_center.x) * adjust_factor;
                float new_center_y = roi_center.y + (det_center.y - roi_center.y) * adjust_factor;
                
                // Update ROI position
                ROI* mutable_roi = roi_manager_->getROI(track->roi_id);
                if (mutable_roi) {
                    mutable_roi->updatePosition(static_cast<int>(new_center_x - mutable_roi->width()/2), 
                                              static_cast<int>(new_center_y - mutable_roi->height()/2));
                    
                    std::cout << "Global phase optimization ROI-" << track->roi_id << " position: track " 
                              << track->displayId() << " matched detection, distance " << min_distance << std::endl;
                }
            }
        }
    }
}

// Enhanced end global phase processing
void TrackingDetectionSystem::finalizeGlobalPhaseEnhanced(int frame_width, int frame_height) {
    // When using only global detection mode, skip ROI creation and management
    if (detection_mode_ == 0) {
        // Only global detection mode, skip ROI creation and management
        LOG_INFO("Only global detection mode: skip ROI processing at end of global phase");
        return;
    }
    
    // Create ROI for confirmed candidates
    auto roi_adjustment_start = std::chrono::high_resolution_clock::now();
    
    int created_count = roi_manager_->createROIsForConfirmedCandidates(frame_width, frame_height, current_frame_id_);
    
    // Merge overlapping ROIs
    int merged_count = roi_manager_->mergeOverlappingROIs(frame_width, frame_height);
    
    auto roi_adjustment_end = std::chrono::high_resolution_clock::now();
    double roi_adjustment_time = std::chrono::duration<double, std::milli>(roi_adjustment_end - roi_adjustment_start).count();
    updateROIAdjustmentTime(roi_adjustment_time);
    
    // If it is a forced global phase, reset the flag
    if (force_global_phase_) {
        force_global_phase_ = false;
        std::cout << "Frame " << frame_count_ << ": forced global phase ended, return normal cycle" << std::endl;
    }
    
    // Call the existing cleanup logic
    finalizeGlobalPhase(frame_width, frame_height);
}

// Save tracking results for the current frame
void TrackingDetectionSystem::saveFrameResults(const std::vector<std::unique_ptr<STrack>>& tracks) {
    for (const auto& track : tracks) {
        if (!track || !track->is_activated) continue;
        
        // Only save confirmed tracking targets (reduce noise)
        if (!track->is_confirmed) continue;
        
        // Extract tracking information
        TrackingResult result(
            current_frame_id_,                    // frame_id
            track->displayId(),                   // track_id  
            track->tlwh.x,                       // x
            track->tlwh.y,                       // y
            track->tlwh.width,                   // width
            track->tlwh.height,                  // height
            track->score,                        // confidence
            0,                                   // class_id (假设都是无人机，类别0)
            track->is_confirmed                  // is_confirmed
        );
        
        tracking_results_.push_back(result);
    }
}

// OpenMP parallel detection method implementation
void TrackingDetectionSystem::initializeThreadLocalInferences(int num_threads) {
    if (thread_local_initialized_) return;
    
    thread_local_inferences_.clear();
    thread_local_inferences_.resize(num_threads);
    
    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        try {
            // Use local detection model path to create thread local inference engine
            thread_local_inferences_[i] = createLocalInferenceEngine(local_model_path_);
            std::cout << "✅ Thread " << i << " local inference engine initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Thread " << i << " local inference engine initialization failed: " << e.what() << std::endl;
        }
    }
    
    thread_local_initialized_ = true;
}

void TrackingDetectionSystem::cleanupThreadLocalInferences() {
    if (thread_local_initialized_) {
        thread_local_inferences_.clear();
        thread_local_initialized_ = false;
        // std::cout << "Thread local inference engine resources have been cleaned" << std::endl;
    }
}

std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>>
TrackingDetectionSystem::parallelROIDetection(const cv::Mat& frame, bool verbose) {
    std::vector<Detection> all_detections;
    std::unordered_map<int, std::vector<Detection>> roi_detections;
    
    const auto& rois = roi_manager_->getROIs();
    if (rois.empty()) {
        return std::make_tuple(all_detections, roi_detections);
    }
    
    #ifdef _OPENMP
    // Check and initialize thread local inference engine
    if (!thread_local_initialized_) {
        initializeThreadLocalInferences(num_threads_);
    }
    
    
    // Convert ROI to vector to support parallel access
    std::vector<std::pair<int, const ROI*>> roi_items;
    roi_items.reserve(rois.size());
    for (const auto& [roi_id, roi_ptr] : rois) {
        roi_items.emplace_back(roi_id, roi_ptr.get());
    }
    
    // Prepare result containers for each thread (avoid competition conditions)
    std::vector<std::vector<Detection>> thread_detections(num_threads_);
    std::vector<std::unordered_map<int, std::vector<Detection>>> thread_roi_detections(num_threads_);
    
    auto start_time = std::chrono::steady_clock::now();
    
    // OpenMP parallel processing ROI
    #pragma omp parallel for num_threads(num_threads_) schedule(dynamic)
    for (int i = 0; i < static_cast<int>(roi_items.size()); ++i) {
        int thread_id = omp_get_thread_num();
        int roi_id = roi_items[i].first;
        const ROI* roi = roi_items[i].second;
        
        if (!roi) continue;  // Safe check
        
        // Get ROI area
        cv::Rect roi_rect = roi->bbox;
        roi_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
            if (verbose) {
                #pragma omp critical
                {
                    std::cout << "Warning: ROI " << roi_id << " exceeds image boundary, skip" << std::endl;
                }
            }
            continue;
        }
        
        // Extract ROI image
        cv::Mat roi_image = frame(roi_rect);
        
        // Use thread local inference engine for detection
        if (thread_id < static_cast<int>(thread_local_inferences_.size()) && 
            thread_local_inferences_[thread_id]) {
            
            auto roi_detections_local = thread_local_inferences_[thread_id]->detect(
                roi_image, config_.local_conf_thres);
            
            // Convert detection result coordinates back to full image coordinates
            for (auto& det : roi_detections_local) {
                det.bbox.x += roi_rect.x;
                det.bbox.y += roi_rect.y;
                thread_detections[thread_id].push_back(det);
                thread_roi_detections[thread_id][roi_id].push_back(det);
            }
            
            if (verbose && !roi_detections_local.empty()) {
                #pragma omp critical
                {
                    // std::cout << "Thread " << thread_id << " ROI " << roi_id 
                    //           << " detected " << roi_detections_local.size() 
                    //           << " targets (position: " << roi_rect.x << "," << roi_rect.y 
                    //           << " size: " << roi_rect.width << "x" << roi_rect.height << ")" << std::endl;
                }
            }
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Merge all thread detection results
    for (int t = 0; t < num_threads_; ++t) {
        // Merge detection results
        all_detections.insert(all_detections.end(), 
                             thread_detections[t].begin(), 
                             thread_detections[t].end());
        
        // Merge ROI detection results
        for (const auto& [roi_id, detections] : thread_roi_detections[t]) {
            roi_detections[roi_id].insert(roi_detections[roi_id].end(),
                                        detections.begin(), detections.end());
        }
    }
    
    if (verbose) {
        // std::cout << "Parallel local detection completed: time " << duration.count() << "ms, "
        //           << "Total detected " << all_detections.size() << " targets" << std::endl;
    }
    
    #else
    // Fall back to serial implementation
    // std::cout << "Warning: OpenMP unavailable, cannot perform parallel detection, fall back to batch detection" << std::endl;
    return optimizedBatchDetection(frame, verbose);
    #endif
    
    // Update ROI detection state
    for (const auto& [roi_id, roi_ptr] : rois) {
        if (roi_detections[roi_id].empty()) {
            roi_ptr->no_detection_count++;
        } else {
            roi_ptr->no_detection_count = 0;  // Reset count
        }
    }
    
    return std::make_tuple(all_detections, roi_detections);
}

// ROI boundary pre-calculation optimization
void TrackingDetectionSystem::precomputeROIBounds(const cv::Mat& frame) {
    const auto& rois = roi_manager_->getROIs();
    
    for (const auto& [roi_id, roi] : rois) {
        auto it = roi_bounds_cache_.find(roi_id);
        bool need_update = false;
        
        if (it == roi_bounds_cache_.end()) {
            // New ROI, need to calculate
            need_update = true;
        } else if (current_frame_id_ - it->second.last_update_frame > 5) {
            // Over 5 frames not updated, recalculate
            need_update = true;
        }
        
        if (need_update) {
            cv::Rect effective_rect = roi->bbox & cv::Rect(0, 0, frame.cols, frame.rows);
            
            ROIBounds bounds;
            bounds.effective_rect = effective_rect;
            bounds.is_valid = (effective_rect.area() > 0);
            bounds.last_update_frame = current_frame_id_;
            
            roi_bounds_cache_[roi_id] = bounds;
        }
    }
    
    // Clean expired cache
    for (auto it = roi_bounds_cache_.begin(); it != roi_bounds_cache_.end();) {
        if (rois.find(it->first) == rois.end()) {
            it = roi_bounds_cache_.erase(it);
        } else {
            ++it;
        }
    }
}

// Optimized ROI image extraction
cv::Mat TrackingDetectionSystem::extractROIOptimized(const cv::Mat& frame, const cv::Rect& roi_rect, int roi_id) {
    if (roi_rect.area() <= 0) {
        return cv::Mat();
    }
    
    // Try to use Mat object in cache pool
    cv::Mat roi_image;
    if (enable_roi_cache_ && roi_id < static_cast<int>(roi_image_cache_.size())) {
        roi_image = roi_image_cache_[roi_id];
        // Check if the size matches, if not, reallocate
        if (roi_image.size() != roi_rect.size() || roi_image.type() != frame.type()) {
            roi_image = cv::Mat();
        }
    }
    
    if (roi_image.empty()) {
        // Direct extraction, avoid extra copy
        roi_image = frame(roi_rect);
        
        // Update cache pool
        if (enable_roi_cache_) {
            if (roi_id >= static_cast<int>(roi_image_cache_.size())) {
                roi_image_cache_.resize(roi_id + 1);
            }
            // Note: here we store the reference of ROI, not the copy
        }
    } else {
        // Use cached Mat, copy data
        frame(roi_rect).copyTo(roi_image);
    }
    
    return roi_image;
}

// Enhanced performance metrics update
void TrackingDetectionSystem::updatePerformanceMetrics(double extraction_time, double inference_time, double transform_time) {
    perf_metrics_.samples_count++;
    
    // Sliding average update
    double alpha = 0.1;  // Learning rate
    perf_metrics_.avg_roi_extraction_time = 
        (1 - alpha) * perf_metrics_.avg_roi_extraction_time + alpha * extraction_time;
    perf_metrics_.avg_inference_time = 
        (1 - alpha) * perf_metrics_.avg_inference_time + alpha * inference_time;
    perf_metrics_.avg_coordinate_transform_time = 
        (1 - alpha) * perf_metrics_.avg_coordinate_transform_time + alpha * transform_time;
    
    // Update skip frame statistics
    perf_metrics_.skip_ratio = static_cast<double>(perf_metrics_.total_skipped_rois) / 
                              std::max(1, perf_metrics_.total_processed_rois + perf_metrics_.total_skipped_rois);
    
    // Print performance report every 100 frames
    if (perf_metrics_.samples_count % 100 == 0) {
        printPerformanceReport();
    }
}

// Enhanced performance report
void TrackingDetectionSystem::printPerformanceReport() {
    // std::cout << "\n=== 🚀 ROI detection performance report (sample count: " << perf_metrics_.samples_count << ") ===" << std::endl;
    // std::cout << "Average ROI extraction time: " << std::fixed << std::setprecision(2) 
    //           << perf_metrics_.avg_roi_extraction_time << "ms" << std::endl;
    // std::cout << "Average inference time: " << std::fixed << std::setprecision(2) 
    //           << perf_metrics_.avg_inference_time << "ms" << std::endl;
    // std::cout << "Average coordinate transformation time: " << std::fixed << std::setprecision(2) 
    //           << perf_metrics_.avg_coordinate_transform_time << "ms" << std::endl;
    
    double total_avg = perf_metrics_.avg_roi_extraction_time + 
                      perf_metrics_.avg_inference_time + 
                      perf_metrics_.avg_coordinate_transform_time;
    // std::cout << "Total average time: " << std::fixed << std::setprecision(2) << total_avg << "ms" << std::endl;
    
    // GPU monitoring information
    if (perf_metrics_.gpu_monitoring_available) {
        std::cout << "GPU usage: " << std::fixed << std::setprecision(1) 
                  << perf_metrics_.gpu_utilization << "%" << std::endl;
        std::cout << "GPU memory usage: " << std::fixed << std::setprecision(1) 
                  << perf_metrics_.gpu_memory_used << "MB / " 
                  << perf_metrics_.gpu_memory_total << "MB ("
                  << perf_metrics_.gpu_memory_util << "%)" << std::endl;
    }
    
    
    // ROI detection state overview
    int active_rois = 0, low_priority_rois = 0;
    for (const auto& [roi_id, state] : roi_detection_states_) {
        if (state.is_high_priority) active_rois++;
        else low_priority_rois++;
    }
    // std::cout << "ROI state: " << active_rois << " high priority, " 
    //           << low_priority_rois << " low priority" << std::endl;
    
    // 🔥 Comment out separator line to avoid interrupting frame output
    // std::cout << "================================================\n" << std::endl;
}

// Smart skip frame detection judgment
bool TrackingDetectionSystem::shouldSkipROIDetection(int roi_id, const ROI& roi) {
    auto it = roi_detection_states_.find(roi_id);
    if (it == roi_detection_states_.end()) {
        // New ROI, not skip frame
        roi_detection_states_[roi_id] = ROIDetectionState();
        return false;
    }
    
    ROIDetectionState& state = it->second;
    
    // High priority ROI not skip frame
    if (state.is_high_priority) {
        return false;
    }
    
    // If ROI is very small, prioritize skip frame (small ROI processing fast, but false detection many)
    cv::Rect roi_rect = roi.bbox;
    int roi_area = roi_rect.area();
    if (roi_area < 10000) { // Area less than 100x100 pixels
        state.max_skip_frames = std::min(6, state.max_skip_frames + 1);
    }
    
    // Based on detection time smart skip frame
    if (state.avg_detection_time > 25.0) {  // Detection time over 25ms
        if (state.consecutive_empty_frames >= 5) {
            state.skip_frames++;
            if (state.skip_frames < state.max_skip_frames) {
                return true;  // Skip this frame
            } else {
                state.skip_frames = 0;  // Reset skip frame count
                return false;  // This frame not skip
            }
        }
    } else {
        // For fast detection ROI, more conservative skip frame strategy
        if (state.consecutive_empty_frames >= 10) {
            state.skip_frames++;
            if (state.skip_frames < state.max_skip_frames) {
                return true;  // Skip this frame
            } else {
                state.skip_frames = 0;  // Reset skip frame count
                return false;  // This frame not skip
            }
        }
    }
    
    // Based on system load dynamic skip frame
    const auto& all_rois = roi_manager_->getROIs();
    if (all_rois.size() > 6) {  // More aggressive skip frame when ROI number is large
        if (state.consecutive_empty_frames >= 3 && !state.is_high_priority) {
            state.skip_frames++;
            if (state.skip_frames < std::min(4, state.max_skip_frames)) {
                return true;
            } else {
                state.skip_frames = 0;
                return false;
            }
        }
    }
    
    return false;
}

// Adaptive confidence threshold
float TrackingDetectionSystem::getAdaptiveConfidenceThreshold(int roi_id, const ROI& roi) {
    if (!enable_dynamic_quality_) {
        return config_.local_conf_thres;
    }
    
    auto it = roi_detection_states_.find(roi_id);
    if (it == roi_detection_states_.end()) {
        return config_.local_conf_thres;
    }
    
    const ROIDetectionState& state = it->second;
    
    // If ROI often has no detection result, reduce threshold
    if (state.consecutive_empty_frames > 5) {
        return std::max(0.3f, config_.local_conf_thres - 0.1f);
    }
    
    // If ROI detection time is long, can slightly increase threshold to reduce post-processing
    if (state.avg_detection_time > 50.0) {  // 超过50ms
        return std::min(0.8f, config_.local_conf_thres + 0.05f);
    }
    
    return config_.local_conf_thres;
}

// Update ROI detection state
void TrackingDetectionSystem::updateROIDetectionState(int roi_id, bool has_detection, double detection_time) {
    auto& state = roi_detection_states_[roi_id];
    
    // Update consecutive empty detection frames
    if (has_detection) {
        state.consecutive_empty_frames = 0;
    } else {
        state.consecutive_empty_frames++;
    }
    
    // Update average detection time (sliding average)
    const double alpha = 0.2;
    state.avg_detection_time = (1 - alpha) * state.avg_detection_time + alpha * detection_time;
    
    // Dynamic adjustment priority - more intelligent strategy
    // Based on multiple factors: detection result, ROI position, ROI size
    const auto& current_rois = roi_manager_->getROIs();
    auto roi_it = current_rois.find(roi_id);
    if (roi_it != current_rois.end()) {
        const ROI* roi = roi_it->second.get();
        
        // Based on detection result priority adjustment
        if (has_detection || state.consecutive_empty_frames < 5) {
            state.is_high_priority = true;
        } else if (state.consecutive_empty_frames > 15) {
            // Consider ROI size: large ROI even without detection for a long time also keep a certain priority
            cv::Rect roi_rect = roi->bbox;
            int roi_area = roi_rect.area();
            if (roi_area > 50000) {  // Greater than about 224x224 pixels
                state.is_high_priority = (state.consecutive_empty_frames < 25);
            } else {
                state.is_high_priority = false;
            }
        }
        
        // Based on ROI position priority adjustment (image center region priority higher)
        if (current_frame_width_ > 0 && current_frame_height_ > 0) {
            cv::Point2f roi_center = roi->center();
            cv::Point2f frame_center(current_frame_width_ / 2.0f, current_frame_height_ / 2.0f);
            float distance_to_center = cv::norm(roi_center - frame_center);
            float max_distance = cv::norm(cv::Point2f(current_frame_width_, current_frame_height_)) / 2.0f;
            float center_factor = 1.0f - (distance_to_center / max_distance);
            
            // ROI in center region easier to keep high priority
            if (center_factor > 0.7f && state.consecutive_empty_frames < 20) {
                state.is_high_priority = true;
            }
        }
    }
    
    // Dynamic adjustment maximum skip frame - more精细的策略
    if (state.avg_detection_time > 40.0) {  // Very slow detection
        state.max_skip_frames = std::min(8, state.max_skip_frames + 1);
    } else if (state.avg_detection_time > 25.0) {  // Slower detection
        state.max_skip_frames = std::min(5, state.max_skip_frames + 1);
    } else if (state.avg_detection_time < 8.0) {  // Very fast detection
        state.max_skip_frames = std::max(1, state.max_skip_frames - 1);
    } else if (state.avg_detection_time < 15.0) {  // Faster detection
        state.max_skip_frames = std::max(2, state.max_skip_frames - 1);
    }
}

// Optimize ROI detection strategy
void TrackingDetectionSystem::optimizeROIDetectionStrategy() {
    // Every 50 frames execute strategy optimization
    if (frame_count_ % 50 != 0) {
        return;
    }
    
    int total_rois = roi_detection_states_.size();
    int high_priority_rois = 0;
    int low_activity_rois = 0;
    
    for (const auto& [roi_id, state] : roi_detection_states_) {
        if (state.is_high_priority) {
            high_priority_rois++;
        }
        if (state.consecutive_empty_frames > 10) {
            low_activity_rois++;
        }
    }
    
    // If low active ROI too many, enable more aggressive skip frame strategy
    if (total_rois > 0) {
        float low_activity_ratio = static_cast<float>(low_activity_rois) / total_rois;
        
        if (low_activity_ratio > 0.6) {
            // More than 60% of ROI active low, enable aggressive mode
            for (auto& [roi_id, state] : roi_detection_states_) {
                if (!state.is_high_priority) {
                    state.max_skip_frames = std::min(8, state.max_skip_frames + 2);
                }
            }
            // std::cout << "🚀 Enable aggressive skip frame mode: " << low_activity_ratio * 100 
            //           << "% ROI低活跃度" << std::endl;
        } else if (low_activity_ratio < 0.2) {
            // Less than 20% of ROI active low, reduce skip frame
            for (auto& [roi_id, state] : roi_detection_states_) {
                state.max_skip_frames = std::max(1, state.max_skip_frames - 1);
            }
        }
    }
    
    // Clean up non-existent ROI state
    const auto& current_rois = roi_manager_->getROIs();
    for (auto it = roi_detection_states_.begin(); it != roi_detection_states_.end();) {
        if (current_rois.find(it->first) == current_rois.end()) {
            it = roi_detection_states_.erase(it);
        } else {
            ++it;
        }
    }
}

// GPU monitoring method implementation
void TrackingDetectionSystem::initializeGPUMonitoring() {
    #ifdef USE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result == NVML_SUCCESS) {
        gpu_monitoring_initialized_ = true;
        perf_metrics_.gpu_monitoring_available = true;
        
        // Get GPU device number and memory information
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result == NVML_SUCCESS) {
            nvmlMemory_t memInfo;
            result = nvmlDeviceGetMemoryInfo(device, &memInfo);
            if (result == NVML_SUCCESS) {
                perf_metrics_.gpu_memory_total = static_cast<float>(memInfo.total) / (1024.0f * 1024.0f);
            }
        }
        // std::cout << "GPU monitoring initialization successful" << std::endl;
    } else {
        perf_metrics_.gpu_monitoring_available = false;
        // std::cout << "GPU monitoring initialization failed, NVML error: " << nvmlErrorString(result) << std::endl;
    }
    #else
    perf_metrics_.gpu_monitoring_available = false;
        // std::cout << "GPU monitoring requires NVML library support" << std::endl;
    #endif
}

void TrackingDetectionSystem::updateGPUMetrics() {
    if (!perf_metrics_.gpu_monitoring_available || !gpu_monitoring_initialized_) {
        return;
    }
    
    #ifdef USE_NVML
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result == NVML_SUCCESS) {
        // Get GPU usage rate
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result == NVML_SUCCESS) {
            perf_metrics_.gpu_utilization = static_cast<float>(utilization.gpu);
        }
        
        // Get memory usage
        nvmlMemory_t memInfo;
        result = nvmlDeviceGetMemoryInfo(device, &memInfo);
        if (result == NVML_SUCCESS) {
            perf_metrics_.gpu_memory_used = static_cast<float>(memInfo.used) / (1024.0f * 1024.0f);
            perf_metrics_.gpu_memory_total = static_cast<float>(memInfo.total) / (1024.0f * 1024.0f);
            perf_metrics_.gpu_memory_util = (static_cast<float>(memInfo.used) / static_cast<float>(memInfo.total)) * 100.0f;
        }
    }
    #endif
}

void TrackingDetectionSystem::cleanupGPUMonitoring() {
    if (gpu_monitoring_initialized_) {
        #ifdef USE_NVML
        nvmlShutdown();
        #endif
        gpu_monitoring_initialized_ = false;
        // std::cout << "GPU monitoring resources cleaned" << std::endl;
    }
}

// After adding destructor implementation in constructor
TrackingDetectionSystem::~TrackingDetectionSystem() {
    // Clean up thread local inference engine
    cleanupThreadLocalInferences();
    
    // Clean up GPU monitoring
    cleanupGPUMonitoring();
}



// Activate force global detection mode
void TrackingDetectionSystem::activateForceGlobalPhase() {
    if (!force_global_phase_) {
        LOG_WARNING("⚠️ Frame " << current_frame_id_ << ": Activate force global phase");
        force_global_phase_ = true;
        force_global_start_frame_ = frame_count_;
        
        // Soft switch to global phase: retain current tracker state to maintain ID continuity
        // No longer call resetTrackerState(), avoid ID interruption
        
        // Clean up expired ROI detection state
        for (auto it = roi_detection_states_.begin(); it != roi_detection_states_.end(); ) {
            if (roi_manager_->getROI(it->first) == nullptr) {
                it = roi_detection_states_.erase(it);
            } else {
                // Reset consecutive empty detection count, ensure next detection will not be skipped
                it->second.consecutive_empty_frames = 0;
                it->second.skip_frames = 0;
                ++it;
            }
        }
    }
}

// Handle force global phase completion
void TrackingDetectionSystem::handleForceGlobalPhaseEnd() {
    if (!force_global_phase_) return;
    
    // Check if exit condition is met
    if ((frame_count_ - force_global_start_frame_) >= config_.global_duration) {
        LOG_INFO("Frame " << frame_count_ << ": Force global phase ends, return to normal cycle");
        
        // Safe exit force global phase
        force_global_phase_ = false;
        force_global_start_frame_ = -1;
        
        // Ensure at least one valid ROI exists before exiting global phase
        const auto& rois = roi_manager_->getROIs();
        if (rois.empty()) {
            LOG_WARNING("⚠️ Force global phase ends with no valid ROI, continue with global phase");
            activateForceGlobalPhase();
        }
    }
}

// Provide method to return visualization results
void TrackingDetectionSystem::process(const cv::Mat& frame, cv::Mat& vis_frame,
                                     std::vector<Detection>& detections,
                                     std::vector<std::unique_ptr<STrack>>& tracks) {
    // Process frame and get results
    auto [dets, trks] = process(frame);
    
    // Copy detection results
    detections = dets;
    
    // Transfer tracking result ownership
    tracks.clear();
    for (auto& track : trks) {
        tracks.push_back(std::move(track));
    }
    
    // Visualization results
    vis_frame = visualize(frame, detections, tracks);
}

// Set whether to use global detection
void TrackingDetectionSystem::setUseGlobalDetection(bool use_global) {
    use_global_detection_ = use_global;
    
    // Select correct tracker based on setting
    if (use_global_detection_) {
        // Use global tracker
        tracker_ = std::move(global_tracker_);
        global_tracker_ = std::make_unique<TPTrack>(config_, 30, roi_manager_);
    } else {
        // Use local tracker
        tracker_ = std::move(local_tracker_);
        local_tracker_ = std::make_unique<TPTrack>(config_, 30, roi_manager_);
    }
    
    LOG_INFO((use_global ? "Switch to global detection mode" : "Switch to local detection mode"));
}

// Save global tracker state
void TrackingDetectionSystem::saveGlobalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks) {
    if (use_global_detection_) {
        // Current using global tracker, directly save state from tracker_
        tracker_->saveTrackerState(saved_tracks);
    } else {
        // Current using local tracker, save state from global_tracker_
        global_tracker_->saveTrackerState(saved_tracks);
    }
}

// Save local tracker state
void TrackingDetectionSystem::saveLocalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks) {
    if (!use_global_detection_) {
        // Current using local tracker, directly save state from tracker_
        tracker_->saveTrackerState(saved_tracks);
    } else {
        // Current using global tracker, save state from local_tracker_
        local_tracker_->saveTrackerState(saved_tracks);
    }
}

// Restore global tracker state
void TrackingDetectionSystem::restoreGlobalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks) {
    if (use_global_detection_) {
        // Current using global tracker, directly restore to tracker_
        tracker_->restoreTrackerState(saved_tracks);
    } else {
        // Current using local tracker, restore to global_tracker_
        global_tracker_->restoreTrackerState(saved_tracks);
    }
}

// Restore local tracker state
void TrackingDetectionSystem::restoreLocalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks) {
    if (!use_global_detection_) {
        // Current using local tracker, directly restore to tracker_
        tracker_->restoreTrackerState(saved_tracks);
    } else {
        // Current using global tracker, restore to local_tracker_
        local_tracker_->restoreTrackerState(saved_tracks);
    }
}

// Time statistics method implementation

void TrackingDetectionSystem::updateInferenceTime(double inference_time) {
    perf_metrics_.total_inference_time += inference_time;
    perf_metrics_.inference_count++;
    perf_metrics_.avg_inference_time_per_frame = perf_metrics_.total_inference_time / perf_metrics_.inference_count;
    perf_metrics_.max_inference_time = std::max(perf_metrics_.max_inference_time, inference_time);
    perf_metrics_.min_inference_time = std::min(perf_metrics_.min_inference_time, inference_time);
}

void TrackingDetectionSystem::updateTrackingTime(double tracking_time) {
    perf_metrics_.total_tracking_time += tracking_time;
    perf_metrics_.tracking_count++;
    perf_metrics_.avg_tracking_time_per_frame = perf_metrics_.total_tracking_time / perf_metrics_.tracking_count;
    perf_metrics_.max_tracking_time = std::max(perf_metrics_.max_tracking_time, tracking_time);
    perf_metrics_.min_tracking_time = std::min(perf_metrics_.min_tracking_time, tracking_time);
}

void TrackingDetectionSystem::updateROIAdjustmentTime(double roi_adjustment_time) {
    perf_metrics_.total_roi_adjustment_time += roi_adjustment_time;
    perf_metrics_.roi_adjustment_count++;
    perf_metrics_.avg_roi_adjustment_time_per_frame = perf_metrics_.total_roi_adjustment_time / perf_metrics_.roi_adjustment_count;
    perf_metrics_.max_roi_adjustment_time = std::max(perf_metrics_.max_roi_adjustment_time, roi_adjustment_time);
    perf_metrics_.min_roi_adjustment_time = std::min(perf_metrics_.min_roi_adjustment_time, roi_adjustment_time);
}

void TrackingDetectionSystem::updateProcessingTime(double processing_time) {
    perf_metrics_.total_processing_time += processing_time;
    perf_metrics_.processing_count++;
    perf_metrics_.avg_processing_time_per_frame = perf_metrics_.total_processing_time / perf_metrics_.processing_count;
    perf_metrics_.max_processing_time = std::max(perf_metrics_.max_processing_time, processing_time);
    perf_metrics_.min_processing_time = std::min(perf_metrics_.min_processing_time, processing_time);
}

void TrackingDetectionSystem::updateDataProcessingTime(double data_processing_time) {
    perf_metrics_.total_data_processing_time += data_processing_time;
    perf_metrics_.data_processing_count++;
    perf_metrics_.avg_data_processing_time_per_frame = perf_metrics_.total_data_processing_time / perf_metrics_.data_processing_count;
    perf_metrics_.max_data_processing_time = std::max(perf_metrics_.max_data_processing_time, data_processing_time);
    perf_metrics_.min_data_processing_time = std::min(perf_metrics_.min_data_processing_time, data_processing_time);
}

void TrackingDetectionSystem::printTimeStatistics() {
    LOG_INFO("=== Time statistics report ===");
    LOG_INFO("Inference time statistics:");
    LOG_INFO("  Total inference time: " << std::fixed << std::setprecision(2) << perf_metrics_.total_inference_time << "ms");
    LOG_INFO("  Average inference time: " << std::fixed << std::setprecision(2) << perf_metrics_.avg_inference_time_per_frame << "ms");
    LOG_INFO("  Maximum inference time: " << std::fixed << std::setprecision(2) << perf_metrics_.max_inference_time << "ms");
    LOG_INFO("  Minimum inference time: " << std::fixed << std::setprecision(2) << perf_metrics_.min_inference_time << "ms");
    LOG_INFO("  Inference count: " << perf_metrics_.inference_count);
    
    LOG_INFO("Tracking time statistics:");
    LOG_INFO("  Total tracking time: " << std::fixed << std::setprecision(2) << perf_metrics_.total_tracking_time << "ms");
    LOG_INFO("  Average tracking time: " << std::fixed << std::setprecision(2) << perf_metrics_.avg_tracking_time_per_frame << "ms");
    LOG_INFO("  Max tracking time: " << std::fixed << std::setprecision(2) << perf_metrics_.max_tracking_time << "ms");
    LOG_INFO("  Minimum tracking time: " << std::fixed << std::setprecision(2) << perf_metrics_.min_tracking_time << "ms");
    LOG_INFO("  Tracking count: " << perf_metrics_.tracking_count);
    
    LOG_INFO("ROI adjustment time statistics:");
    LOG_INFO("  Total ROI adjustment time: " << std::fixed << std::setprecision(2) << perf_metrics_.total_roi_adjustment_time << "ms");
    LOG_INFO("  Average ROI adjustment time: " << std::fixed << std::setprecision(2) << perf_metrics_.avg_roi_adjustment_time_per_frame << "ms");
    LOG_INFO("  Maximum ROI adjustment time: " << std::fixed << std::setprecision(2) << perf_metrics_.max_roi_adjustment_time << "ms");
    LOG_INFO("  Minimum ROI adjustment time: " << std::fixed << std::setprecision(2) << perf_metrics_.min_roi_adjustment_time << "ms");
    LOG_INFO("  ROI adjustment count: " << perf_metrics_.roi_adjustment_count);
    
    LOG_INFO("Data processing time statistics:");
    LOG_INFO("  Total data processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.total_data_processing_time << "ms");
    LOG_INFO("  Average data processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.avg_data_processing_time_per_frame << "ms");
    LOG_INFO("  Maximum data processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.max_data_processing_time << "ms");
    LOG_INFO("  Minimum data processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.min_data_processing_time << "ms");
    LOG_INFO("  Data processing count: " << perf_metrics_.data_processing_count);
    
    LOG_INFO("Overall processing time statistics:");
    LOG_INFO("  Total processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.total_processing_time << "ms");
    LOG_INFO("  Average processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.avg_processing_time_per_frame << "ms");
    LOG_INFO("  Maximum processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.max_processing_time << "ms");
    LOG_INFO("  Minimum processing time: " << std::fixed << std::setprecision(2) << perf_metrics_.min_processing_time << "ms");
    LOG_INFO("  Processing count: " << perf_metrics_.processing_count);
    
    // Calculate time ratio of each part
    if (perf_metrics_.total_processing_time > 0) {
        double inference_ratio = (perf_metrics_.total_inference_time / perf_metrics_.total_processing_time) * 100.0;
        double tracking_ratio = (perf_metrics_.total_tracking_time / perf_metrics_.total_processing_time) * 100.0;
        double roi_ratio = (perf_metrics_.total_roi_adjustment_time / perf_metrics_.total_processing_time) * 100.0;
        double data_processing_ratio = (perf_metrics_.total_data_processing_time / perf_metrics_.total_processing_time) * 100.0;
        double other_ratio = 100.0 - inference_ratio - tracking_ratio - roi_ratio - data_processing_ratio;
        
        LOG_INFO("Time ratio analysis:");
        LOG_INFO("  Inference time ratio: " << std::fixed << std::setprecision(1) << inference_ratio << "%");
        LOG_INFO("  Tracking time ratio: " << std::fixed << std::setprecision(1) << tracking_ratio << "%");
        LOG_INFO("  ROI adjustment time ratio: " << std::fixed << std::setprecision(1) << roi_ratio << "%");
        LOG_INFO("  Data processing time ratio: " << std::fixed << std::setprecision(1) << data_processing_ratio << "%");
        LOG_INFO("  Other time ratio: " << std::fixed << std::setprecision(1) << other_ratio << "%");
    }
    
    LOG_INFO("==================");
}

void TrackingDetectionSystem::resetTimeStatistics() {
    // Reset inference time statistics
    perf_metrics_.total_inference_time = 0.0;
    perf_metrics_.avg_inference_time_per_frame = 0.0;
    perf_metrics_.max_inference_time = 0.0;
    perf_metrics_.min_inference_time = std::numeric_limits<double>::max();
    perf_metrics_.inference_count = 0;
    
    // Reset tracking time statistics
    perf_metrics_.total_tracking_time = 0.0;
    perf_metrics_.avg_tracking_time_per_frame = 0.0;
    perf_metrics_.max_tracking_time = 0.0;
    perf_metrics_.min_tracking_time = std::numeric_limits<double>::max();
    perf_metrics_.tracking_count = 0;
    
    // Reset ROI adjustment time statistics
    perf_metrics_.total_roi_adjustment_time = 0.0;
    perf_metrics_.avg_roi_adjustment_time_per_frame = 0.0;
    perf_metrics_.max_roi_adjustment_time = 0.0;
    perf_metrics_.min_roi_adjustment_time = std::numeric_limits<double>::max();
    perf_metrics_.roi_adjustment_count = 0;
    
    // Reset overall processing time statistics
    perf_metrics_.total_processing_time = 0.0;
    perf_metrics_.avg_processing_time_per_frame = 0.0;
    perf_metrics_.max_processing_time = 0.0;
    perf_metrics_.min_processing_time = std::numeric_limits<double>::max();
    perf_metrics_.processing_count = 0;
    
    // Reset data processing time statistics
    perf_metrics_.total_data_processing_time = 0.0;
    perf_metrics_.avg_data_processing_time_per_frame = 0.0;
    perf_metrics_.max_data_processing_time = 0.0;
    perf_metrics_.min_data_processing_time = std::numeric_limits<double>::max();
    perf_metrics_.data_processing_count = 0;
    
    LOG_INFO("Time statistics reset");
}

// Generate different colors for different IDs
cv::Scalar TrackingDetectionSystem::generateColorForID(int track_id) {
    // Use predefined color palette, ensure color contrast is enough
    static const std::vector<cv::Scalar> color_palette = {
        cv::Scalar(255, 0, 0),     // Red
        cv::Scalar(0, 255, 0),     // Green
        cv::Scalar(0, 0, 255),     // Blue
        cv::Scalar(255, 255, 0),   // Cyan
        cv::Scalar(255, 0, 255),   // Magenta
        cv::Scalar(0, 255, 255),   // Yellow
        cv::Scalar(128, 0, 128),   // Purple
        cv::Scalar(255, 165, 0),   // Orange
        cv::Scalar(0, 128, 128),   // Dark cyan
        cv::Scalar(128, 128, 0),   // Olive
        cv::Scalar(255, 192, 203), // Pink
        cv::Scalar(0, 255, 127),   // Spring green
        cv::Scalar(255, 20, 147),  // Deep pink
        cv::Scalar(0, 191, 255),   // Deep sky blue
        cv::Scalar(50, 205, 50),   // Lime green
        cv::Scalar(255, 69, 0),    // Red orange
        cv::Scalar(138, 43, 226),  // Blue violet
        cv::Scalar(0, 100, 0),     // Dark green
        cv::Scalar(255, 215, 0),   // Gold
        cv::Scalar(70, 130, 180)   // Steel blue
    };
    
    // Use modulus operation to ensure ID is within color palette range
    int color_index = track_id % color_palette.size();
    return color_palette[color_index];
}

// Adjust color based on track state
cv::Scalar TrackingDetectionSystem::adjustColorForState(cv::Scalar base_color, bool is_confirmed, bool is_recovered, bool is_lost) {
    if (is_lost) {
        // Lost track uses gray, but keep a certain hue
        return cv::Scalar(128, 128, 128);
    } else if (is_recovered) {
        // Recovered track uses brighter color
        return cv::Scalar(
            std::min(255, static_cast<int>(base_color[0] * 1.2)),
            std::min(255, static_cast<int>(base_color[1] * 1.2)),
            std::min(255, static_cast<int>(base_color[2] * 1.2))
        );
    } else if (is_confirmed) {
        // Confirmed track uses original color
        return base_color;
    } else {
        // Temporary track uses darker color
        return cv::Scalar(
            static_cast<int>(base_color[0] * 0.7),
            static_cast<int>(base_color[1] * 0.7),
            static_cast<int>(base_color[2] * 0.7)
        );
    }
}

} // namespace tracking