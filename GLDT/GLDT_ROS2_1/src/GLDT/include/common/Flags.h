#ifndef FLAGS_H
#define FLAGS_H

#include <gflags/gflags.h>
#include <string>

// Declare all global flags

// Video reader
DECLARE_string(video_reader_mode);
DECLARE_bool(enable_video_downscaling);
DECLARE_bool(enable_video_output);     // enable saving output video
DECLARE_bool(enable_display);           // enable on-screen display window

// Global model
DECLARE_double(global_conf_threshold);
DECLARE_double(global_nms_threshold);
DECLARE_int32(global_preprocess_mode);  // 0=CPU, 1=CPU+GPU hybrid, 2=GPU
DECLARE_int32(global_postprocess_mode); // 0=CPU, 1=GPU
DECLARE_int32(global_duration);         // number of frames to run global detection

// Local model
DECLARE_double(local_conf_threshold);
DECLARE_double(local_nms_threshold);
DECLARE_int32(local_preprocess_mode);   // 0=CPU, 1=CPU+GPU hybrid, 2=GPU
DECLARE_int32(local_postprocess_mode);  // 0=CPU, 1=GPU
DECLARE_int32(local_duration);          // number of frames to run local detection

// ROI configuration
DECLARE_int32(roi_margin);
DECLARE_int32(roi_min_size);
DECLARE_int32(roi_max_size);
DECLARE_double(roi_overlap_threshold);
DECLARE_int32(roi_max_no_detection);
DECLARE_int32(roi_memory_frames);

// Adaptive ROI
DECLARE_double(roi_target_scale_factor);    // ROI size relative to target
DECLARE_bool(enable_adaptive_roi_size);     // enable adaptive ROI sizing
DECLARE_double(roi_size_smooth_factor);     // smoothing factor for ROI size

// Tracking configuration
DECLARE_double(track_high_thresh);
DECLARE_double(track_low_thresh);
DECLARE_double(new_track_thresh);
DECLARE_int32(track_buffer);
DECLARE_double(match_thresh);

// ROI-constrained tracking
DECLARE_double(roi_match_thresh);
DECLARE_double(roi_spatial_tolerance);
DECLARE_double(roi_reactivate_thresh);

// Candidate targets
DECLARE_int32(candidate_confirm_frames);
DECLARE_int32(candidate_window);

// Performance
DECLARE_bool(enable_batch_inference);
DECLARE_int32(batch_size);
DECLARE_bool(enable_parallel_processing);
DECLARE_int32(num_threads);
DECLARE_bool(enable_appearance);

// Runtime mode
DECLARE_bool(single_thread_mode);

// Algorithm selection
DECLARE_bool(use_original_bytetrack);
DECLARE_int32(detection_mode);

// Cost matrix fusion weights
DECLARE_double(cost_iou_weight);
DECLARE_double(cost_roi_weight);
DECLARE_double(cost_radius_weight);
DECLARE_double(cost_spatial_weight);
DECLARE_double(trajectory_radius_factor);
DECLARE_bool(use_enhanced_matching);

// Memory recovery control
DECLARE_bool(enable_gmm_memory_recovery);

// Logging
DECLARE_int32(log_level);
DECLARE_bool(log_to_console);
DECLARE_bool(log_to_file);
DECLARE_string(custom_log_dir); // use custom_log_dir to avoid name clash with glog
DECLARE_bool(verbose_log);
DECLARE_bool(log_inference_details);
DECLARE_bool(log_binding_info);
DECLARE_bool(log_detection_boxes);

// Helper functions
namespace flags {

// Initialize flags from file and command line
void init(int* argc, char*** argv, const std::string& flagfile = "./config/config.flag");

// Print current flag values
void printFlags();

// Video reader mode
enum class VideoReaderMode {
    OPENCV,
    FFMPEG,
    GSTREAMER
};

VideoReaderMode getVideoReaderMode();

// Preprocess mode
enum class PreprocessMode {
    CPU = 0,
    MIXED = 1,
    GPU = 2
};

// Postprocess mode
enum class PostprocessMode {
    CPU = 0,
    GPU = 1
};

// Accessors
PreprocessMode getGlobalPreprocessMode();
PostprocessMode getGlobalPostprocessMode();
PreprocessMode getLocalPreprocessMode();
PostprocessMode getLocalPostprocessMode();

// Validate all flag values
bool validateFlags();

}  // namespace flags

#endif // FLAGS_H 