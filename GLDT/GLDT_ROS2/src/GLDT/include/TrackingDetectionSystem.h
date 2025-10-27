#ifndef TRACKING_DETECTION_SYSTEM_H
#define TRACKING_DETECTION_SYSTEM_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <limits>

// GPU monitoring support
#include <cuda_runtime.h>
#ifdef USE_NVML
#include <nvml.h>
#endif

// Required headers
#include "common/Config.h"
#include "common/Detection.h"
#include "inference/InferenceInterface.h"

// Forward declarations
namespace tracking {
    class TPTrack;
    class STrack;
    class ROIManager;
    class ROI;
}

namespace tracking {

// Tracking result
struct TrackingResult {
    int frame_id;
    int track_id;
    float x, y, width, height;
    float confidence;
    int class_id;
    bool is_confirmed;
    
    TrackingResult(int f_id, int t_id, float x_, float y_, float w_, float h_, 
                   float conf, int cls_id = 0, bool confirmed = true) 
        : frame_id(f_id), track_id(t_id), x(x_), y(y_), width(w_), height(h_), 
          confidence(conf), class_id(cls_id), is_confirmed(confirmed) {}
};

// Detection statistics
struct DetectionStats {
    std::deque<double> single_times;
    std::deque<double> batch_times;
    int adaptive_batch_size = 4;
    static constexpr size_t MAX_STATS = 10;
};

// Performance metrics
struct PerformanceMetrics {
    double avg_roi_extraction_time = 0.0;
    double avg_inference_time = 0.0;
    double avg_coordinate_transform_time = 0.0;
    int samples_count = 0;
    
    // Skip-frame statistics
    int total_skipped_rois = 0;
    int total_processed_rois = 0;
    double skip_ratio = 0.0;
    
    // Inference time statistics
    double total_inference_time = 0.0;
    double avg_inference_time_per_frame = 0.0;
    double max_inference_time = 0.0;
    double min_inference_time = std::numeric_limits<double>::max();
    int inference_count = 0;
    
    // Tracking time statistics
    double total_tracking_time = 0.0;
    double avg_tracking_time_per_frame = 0.0;
    double max_tracking_time = 0.0;
    double min_tracking_time = std::numeric_limits<double>::max();
    int tracking_count = 0;
    
    // ROI adjustment time statistics
    double total_roi_adjustment_time = 0.0;
    double avg_roi_adjustment_time_per_frame = 0.0;
    double max_roi_adjustment_time = 0.0;
    double min_roi_adjustment_time = std::numeric_limits<double>::max();
    int roi_adjustment_count = 0;
    
    // Overall processing time statistics
    double total_processing_time = 0.0;
    double avg_processing_time_per_frame = 0.0;
    double max_processing_time = 0.0;
    double min_processing_time = std::numeric_limits<double>::max();
    int processing_count = 0;
        
    // Data processing time statistics
    double total_data_processing_time = 0.0;
    double avg_data_processing_time_per_frame = 0.0;
    double max_data_processing_time = 0.0;
    double min_data_processing_time = std::numeric_limits<double>::max();
    int data_processing_count = 0;
    
    // GPU monitoring
    float gpu_utilization = 0.0f;     // GPU utilization (%)
    float gpu_memory_used = 0.0f;     // Used GPU memory (MB)
    float gpu_memory_total = 0.0f;    // Total GPU memory (MB)
    float gpu_memory_util = 0.0f;     // GPU memory utilization (%)
    bool gpu_monitoring_available = false;  // Whether GPU monitoring is available
};

class TrackingDetectionSystem {
public:
    // Constructor takes both global and local model paths
    TrackingDetectionSystem(const std::string& global_model_path,
                           const std::string& local_model_path, 
                           const Config& config);
    
    ~TrackingDetectionSystem();
    
    std::tuple<std::vector<Detection>, std::vector<std::unique_ptr<STrack>>> 
    process(const cv::Mat& frame);
    
    // Overload that also returns a visualization frame
    void process(const cv::Mat& frame, cv::Mat& vis_frame,
                std::vector<Detection>& detections,
                std::vector<std::unique_ptr<STrack>>& tracks);
    
    cv::Mat visualize(const cv::Mat& frame, const std::vector<Detection>& detections,
                     const std::vector<std::unique_ptr<STrack>>& tracks);
    
    void saveResults(const std::string& save_dir);
    // Overload with video name
    void saveResults(const std::string& save_dir, const std::string& video_name);
    void setTotalFrames(int total_frames);
    
    // Toggle global detection
    void setUseGlobalDetection(bool use_global);
    
    // Performance statistics accessor
    const PerformanceMetrics& getPerformanceMetrics() const { return perf_metrics_; }
    
    // Save tracker states
    void saveGlobalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks);
    void saveLocalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks);
    
    // Restore tracker states
    void restoreGlobalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks);
    void restoreLocalTrackerState(std::vector<std::unique_ptr<STrack>>& saved_tracks);
    
private:
    // Configuration and state
    Config config_;
    int frame_count_;
    int current_frame_id_;
    bool is_first_frame_;
    int total_frames_;
    int num_rois_ = 0;  // number of active ROIs
    
    // Detection mode: 0=global only, 1=global+local
    int detection_mode_ = 1;
    
    // Whether to use global detection
    bool use_global_detection_ = true;
    
    // Cached frame size
    int current_frame_width_ = 0;
    int current_frame_height_ = 0;
    
    // Previous frame cache (for motion-based detection)
    cv::Mat prev_frame_;
    
    // Cached model paths (for thread-local initialization)
    std::string global_model_path_;
    std::string local_model_path_;
    
    // Collected tracking results
    std::vector<TrackingResult> tracking_results_;
    
    // Detection statistics
    DetectionStats detection_stats_;
    
    // Core components
    std::shared_ptr<ROIManager> roi_manager_;
    std::unique_ptr<TPTrack> tracker_;
    // Separate global and local trackers
    std::unique_ptr<TPTrack> global_tracker_;
    std::unique_ptr<TPTrack> local_tracker_;
    std::unique_ptr<InferenceInterface> global_inference_;
    std::unique_ptr<InferenceInterface> local_inference_;
    
    // Memory pools and caches
    std::vector<cv::Mat> roi_image_cache_;  // ROI image cache pool
    std::vector<std::vector<Detection>> detection_cache_;  // detection result cache pool
    bool enable_roi_cache_ = true;
    
    // ROI detection cache
    struct ROIDetectionCache {
        std::vector<Detection> last_detections;
        int last_update_frame = -1;
        bool is_valid = false;
    };
    std::unordered_map<int, ROIDetectionCache> roi_detection_cache_;
    
    // Precomputed ROI bounds
    struct ROIBounds {
        cv::Rect effective_rect;
        bool is_valid;
        int last_update_frame;
    };
    std::unordered_map<int, ROIBounds> roi_bounds_cache_;
    
    // Performance monitoring
    PerformanceMetrics perf_metrics_;
    
    // GPU monitoring helpers
    void initializeGPUMonitoring();
    void updateGPUMetrics();
    void cleanupGPUMonitoring();
    bool gpu_monitoring_initialized_ = false;
    
    // Optimization helpers
    void precomputeROIBounds(const cv::Mat& frame);
    cv::Mat extractROIOptimized(const cv::Mat& frame, const cv::Rect& roi_rect, int roi_id);
    void updatePerformanceMetrics(double extraction_time, double inference_time, double transform_time);
    void printPerformanceReport();
    
    // Time statistics update helpers
    void updateInferenceTime(double inference_time);
    void updateTrackingTime(double tracking_time);
    void updateROIAdjustmentTime(double roi_adjustment_time);
    void updateProcessingTime(double processing_time);
    void updateDataProcessingTime(double data_processing_time);
    void printTimeStatistics();
    void resetTimeStatistics();
    
    // Phase checks
    bool isGlobalPhase() const;
    bool isGlobalPhaseEnd() const;
    void activateForceGlobalPhase();  // force entering global phase
    void handleForceGlobalPhaseEnd(); // handle end of forced global phase
    
    // Detection methods
    std::vector<Detection> globalDetection(const cv::Mat& frame);
    std::vector<Detection> globalDetectionWithMotion(const cv::Mat& frame);
    std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>> 
    localDetection(const cv::Mat& frame);
    
    // Batch detection methods
    std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>> 
    optimizedBatchDetection(const cv::Mat& frame, bool verbose = false);
    std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>>
    chunkedBatchDetection(const cv::Mat& frame, int chunk_size, bool verbose = false);
    std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>>
    singleROIDetection(const cv::Mat& frame, bool verbose = false);
    
    // Phase handlers
    void handleGlobalPhase(std::vector<std::unique_ptr<STrack>>& tracks,
                          const std::vector<Detection>& detections,
                          int frame_width, int frame_height);
    void handleLocalPhase(std::vector<std::unique_ptr<STrack>>& tracks,
                         int frame_width, int frame_height);
    
    // Utilities
    void ensureTargetsInSafetyZones(std::vector<std::unique_ptr<STrack>>& tracks,
                                   int frame_width, int frame_height);
    void updateTrackROIAssociations(std::vector<std::unique_ptr<STrack>>& tracks);
    void finalizeGlobalPhase(int frame_width, int frame_height);
    void finalizeGlobalPhaseEnhanced(int frame_width, int frame_height);
    
    // Detection/processing utilities
    void adjustBatchSize();
    std::tuple<int, int, int, int> validateROIBounds(const ROI& roi, int frame_width, int frame_height);
    void updateCandidatesFromDetections(const std::vector<Detection>& detections_outside_roi, int frame_id);
    void optimizeROITrackAssociation(const std::vector<std::unique_ptr<STrack>>& tracks,
                                   const std::vector<Detection>& detections);
    
    // Visualization helpers
    void drawDashedRectangle(cv::Mat& img, cv::Point pt1, cv::Point pt2, 
                           cv::Scalar color, int thickness = 1, int dash_length = 10, int gap_length = 5);
    void drawDashedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2, 
                       cv::Scalar color, int thickness = 1, int dash_length = 10, int gap_length = 5);
    
    // Colors
    cv::Scalar generateColorForID(int track_id);
    cv::Scalar adjustColorForState(cv::Scalar base_color, bool is_confirmed, bool is_recovered, bool is_lost);
    
    // Logging helpers
    void printGlobalDetectionSummary(const std::vector<Detection>& detections, 
                                   double inference_time, float threshold);
    void printDetectionSummary(const std::vector<Detection>& detections, double inference_time);
    void printTrackingInfo(const std::vector<std::unique_ptr<STrack>>& tracks);
    void printKeyTrackingResults(int frame_id, const std::vector<Detection>& detections,
                                const std::vector<std::unique_ptr<STrack>>& tracks);  // 🔥 Key results (bypasses log_level)
    
    // Per-frame result saving
    void saveFrameResults(const std::vector<std::unique_ptr<STrack>>& tracks);
    
    // OpenMP-based parallel ROI detection
    std::tuple<std::vector<Detection>, std::unordered_map<int, std::vector<Detection>>>
    parallelROIDetection(const cv::Mat& frame, bool verbose = false);
    
    void initializeThreadLocalInferences(int num_threads);
    void cleanupThreadLocalInferences();
    
    // Thread-local inference engines
    std::vector<std::unique_ptr<InferenceInterface>> thread_local_inferences_;
    bool thread_local_initialized_ = false;
    int num_threads_ = 0;

    // Intelligent ROI skip strategy
    struct ROIDetectionState {
        int consecutive_empty_frames = 0;  // consecutive empty detection frames
        int skip_frames = 0;               // current skip count
        int max_skip_frames = 3;           // max skip frames
        double avg_detection_time = 0.0;   // average detection time
        bool is_high_priority = true;      // whether ROI is high-priority
    };
    std::unordered_map<int, ROIDetectionState> roi_detection_states_;
    
    // Dynamic quality adjustment
    bool enable_dynamic_quality_ = true;
    float base_conf_threshold_ = 0.5f;
    
    // ROI detection optimization
    bool shouldSkipROIDetection(int roi_id, const ROI& roi);
    float getAdaptiveConfidenceThreshold(int roi_id, const ROI& roi);
    void updateROIDetectionState(int roi_id, bool has_detection, double detection_time);
    void optimizeROIDetectionStrategy();

    // Phase control flags
    bool force_global_phase_;            // force entering global phase
    int force_global_start_frame_;       // starting frame of forced global phase
};

} // namespace tracking

#endif // TRACKING_DETECTION_SYSTEM_H 