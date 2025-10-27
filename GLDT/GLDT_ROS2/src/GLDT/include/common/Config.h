#ifndef CONFIG_H
#define CONFIG_H

#include "common/Flags.h"

namespace tracking {

// Configuration struct wrapping gflags
struct Config {
    // Construct from gflags
    Config() {
        updateFromFlags();
    }
    
    // Refresh from gflags
    void updateFromFlags() {
        // Detection
        global_conf_thres = FLAGS_global_conf_threshold;
        global_nms_thres = FLAGS_global_nms_threshold;
        local_conf_thres = FLAGS_local_conf_threshold;
        local_nms_thres = FLAGS_local_nms_threshold;
        global_duration = FLAGS_global_duration;
        local_duration = FLAGS_local_duration;
        
        // Pre/Postprocess modes
        global_preprocess_mode = FLAGS_global_preprocess_mode;
        global_postprocess_mode = FLAGS_global_postprocess_mode;
        local_preprocess_mode = FLAGS_local_preprocess_mode;
        local_postprocess_mode = FLAGS_local_postprocess_mode;
        
        // ROI config
        roi_size = FLAGS_roi_min_size;  // use min size as initial size
        roi_min_size = FLAGS_roi_min_size;
        roi_max_size = FLAGS_roi_max_size;
        roi_margin = FLAGS_roi_margin;
        roi_overlap_threshold = FLAGS_roi_overlap_threshold;
        roi_max_no_detection = FLAGS_roi_max_no_detection;
        roi_memory_frames = FLAGS_roi_memory_frames;
        
        // Adaptive ROI
        roi_target_scale_factor = FLAGS_roi_target_scale_factor;
        enable_adaptive_roi_size = FLAGS_enable_adaptive_roi_size;
        roi_size_smooth_factor = FLAGS_roi_size_smooth_factor;
        
        // Tracking
        track_high_thresh = FLAGS_track_high_thresh;
        track_low_thresh = FLAGS_track_low_thresh;
        new_track_thresh = FLAGS_new_track_thresh;
        track_buffer = FLAGS_track_buffer;
        match_thresh = FLAGS_match_thresh;
        
        // ROI-constrained tracking
        roi_match_thresh = FLAGS_roi_match_thresh;
        roi_spatial_tolerance = FLAGS_roi_spatial_tolerance;
        roi_reactivate_thresh = FLAGS_roi_reactivate_thresh;
        
        // Candidate targets
        candidate_confirm_frames = FLAGS_candidate_confirm_frames;
        candidate_window = FLAGS_candidate_window;
        
        // Performance
        enable_batch = FLAGS_enable_batch_inference;
        batch_size = FLAGS_batch_size;
        enable_parallel = FLAGS_enable_parallel_processing;
        num_threads = FLAGS_num_threads;
        
        // Appearance feature extraction
        enable_appearance = FLAGS_enable_appearance;
        
        // Algorithm selection
        use_original_bytetrack = FLAGS_use_original_bytetrack;
        
        // Cost matrix fusion weights
        cost_iou_weight = FLAGS_cost_iou_weight;
        cost_roi_weight = FLAGS_cost_roi_weight;
        cost_radius_weight = FLAGS_cost_radius_weight;
        cost_spatial_weight = FLAGS_cost_spatial_weight;
        trajectory_radius_factor = FLAGS_trajectory_radius_factor;
        use_enhanced_matching = FLAGS_use_enhanced_matching;
        
        // Logging
        log_level = FLAGS_log_level;
        verbose_logging = FLAGS_verbose_log;
        log_inference_details = FLAGS_log_inference_details;
        log_binding_info = FLAGS_log_binding_info;
        log_detection_boxes = FLAGS_log_detection_boxes;
    }
    
    // Detection
    float global_conf_thres;    
    float global_nms_thres;
    float local_conf_thres;     
    float local_nms_thres;
    int global_duration;
    int local_duration;
    
    // Pre/Postprocess modes
    int global_preprocess_mode;  // 0=CPU, 1=CPU+GPU hybrid, 2=GPU
    int global_postprocess_mode; // 0=CPU, 1=GPU
    int local_preprocess_mode;   // 0=CPU, 1=CPU+GPU hybrid, 2=GPU
    int local_postprocess_mode;  // 0=CPU, 1=GPU
    
    // ROI configuration
    int roi_size;
    int roi_min_size;
    int roi_max_size;
    int roi_margin;
    float roi_overlap_threshold;
    int roi_max_no_detection;
    int roi_memory_frames;
    
    // Adaptive ROI
    double roi_target_scale_factor;      // ROI size relative to target
    bool enable_adaptive_roi_size;       // enable adaptive ROI sizing
    double roi_size_smooth_factor;       // smoothing factor for ROI size
    
    // Tracking
    float track_high_thresh;     
    float track_low_thresh;       
    float new_track_thresh;      
    int track_buffer;
    float match_thresh;        
    
    // ROI-constrained tracking
    float roi_match_thresh; 
    float roi_spatial_tolerance;
    float roi_reactivate_thresh;   
    
    // Candidate targets
    int candidate_confirm_frames;  
    int candidate_window;
    
    // Performance
    bool enable_batch;
    int batch_size;
    bool enable_parallel;
    int num_threads;
    
    // Appearance
    bool enable_appearance;
    
    // Algorithm selection
    bool use_original_bytetrack;
    
    // Cost matrix fusion weights
    float cost_iou_weight;
    float cost_roi_weight;
    float cost_radius_weight;
    float cost_spatial_weight;
    float trajectory_radius_factor;
    bool use_enhanced_matching;
    
    // Logging
    int log_level;
    bool verbose_logging;
    bool log_inference_details;
    bool log_binding_info;
    bool log_detection_boxes;
};

} // namespace tracking

#endif // CONFIG_H 