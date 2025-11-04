#include "common/Flags.h"
#include "common/Logger.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>

// Define all global flags

// Video read mode
DEFINE_string(video_reader_mode, "opencv", "Video read mode: opencv, ffmpeg or gstreamer");
DEFINE_bool(enable_video_downscaling, true, "Whether to enable video downscaling to improve performance");
DEFINE_bool(enable_video_output, false, "Whether to enable video output saving (false can get higher FPS)");
DEFINE_bool(enable_display, true, "Whether to display window (overlay/imshow/waitKey). Close to reduce latency and improve FPS");

// Global model configuration
DEFINE_double(global_conf_threshold, 0.5, "Global detection confidence threshold (0.0-1.0)");
DEFINE_double(global_nms_threshold, 0.45, "Global detection NMS threshold (0.0-1.0)");
DEFINE_int32(global_preprocess_mode, 0, "Global preprocess mode: 0=CPU, 1=CPU and GPU mixed, 2=GPU");
DEFINE_int32(global_postprocess_mode, 0, "Global postprocess mode: 0=CPU, 1=GPU");
DEFINE_int32(global_duration, 30, "Global detection duration frames");

// Local model configuration
DEFINE_double(local_conf_threshold, 0.4, "Local detection confidence threshold (0.0-1.0)");
DEFINE_double(local_nms_threshold, 0.45, "Local detection NMS threshold (0.0-1.0)");
DEFINE_int32(local_preprocess_mode, 0, "Local preprocess mode: 0=CPU, 1=CPU and GPU mixed, 2=GPU");
DEFINE_int32(local_postprocess_mode, 0, "Local postprocess mode: 0=CPU, 1=GPU");
DEFINE_int32(local_duration, 120, "Local detection duration frames");

// ROI configuration
DEFINE_int32(roi_margin, 50, "ROI margin pixels");
DEFINE_int32(roi_min_size, 100, "ROI minimum size");
DEFINE_int32(roi_max_size, 500, "ROI maximum size");
DEFINE_double(roi_overlap_threshold, 0.1, "ROI overlap threshold");
DEFINE_int32(roi_max_no_detection, 30, "ROI maximum no detection frames");
DEFINE_int32(roi_memory_frames, 30, "ROI memory frames");

// New: Adaptive ROI configuration
DEFINE_double(roi_target_scale_factor, 5.0, "ROI target scale factor");
DEFINE_bool(enable_adaptive_roi_size, true, "Whether to enable adaptive ROI size");
DEFINE_double(roi_size_smooth_factor, 0.3, "ROI size smooth factor (0.1-0.5)");

// Tracking configuration
DEFINE_double(track_high_thresh, 0.6, "High confidence tracking threshold");
DEFINE_double(track_low_thresh, 0.3, "Low confidence tracking threshold");
DEFINE_double(new_track_thresh, 0.4, "New track threshold");
DEFINE_int32(track_buffer, 30, "Tracking buffer size");
DEFINE_double(match_thresh, 0.5, "Match threshold");

// ROI constrained tracking configuration
DEFINE_double(roi_match_thresh, 0.7, "ROI match threshold");
DEFINE_double(roi_spatial_tolerance, 80.0, "ROI spatial tolerance");
DEFINE_double(roi_reactivate_thresh, 0.4, "ROI reactivate threshold");

// Candidate target configuration
DEFINE_int32(candidate_confirm_frames, 15, "Candidate target confirm frames");
DEFINE_int32(candidate_window, 30, "Candidate target window size");

// Performance configuration
DEFINE_bool(enable_batch_inference, true, "Enable batch inference");
DEFINE_int32(batch_size, 4, "Batch inference size");
DEFINE_bool(enable_parallel_processing, true, "Enable parallel processing");
DEFINE_int32(num_threads, 4, "Parallel threads");
DEFINE_bool(enable_appearance, false, "Whether to extract appearance features in the detection stage");

// Running mode: single thread/multi-thread
DEFINE_bool(single_thread_mode, true, "Use single thread running ( true ) or multi-thread running ( false )");

// Algorithm selection configuration
DEFINE_bool(use_original_bytetrack, false, "Use original ByteTrack algorithm");
DEFINE_int32(detection_mode, 1, "Detection mode: 0=only global detection, 1=global+local joint detection");

// Fusion cost matrix weight configuration
DEFINE_double(cost_iou_weight, 0.6, "IOU cost weight");
DEFINE_double(cost_roi_weight, 0.25, "ROI cost weight");
DEFINE_double(cost_radius_weight, 0.15, "History trajectory circular cost weight");
DEFINE_double(cost_spatial_weight, 0.2, "Spatial distance cost weight");
DEFINE_double(trajectory_radius_factor, 2.0, "History trajectory radius factor");
DEFINE_bool(use_enhanced_matching, true, "Use enhanced matching");

// Memory recovery control
DEFINE_bool(enable_gmm_memory_recovery, false, "Enable GMM memory recovery");

// Log configuration - Note: Avoid conflicts with glog's own flags
DEFINE_int32(log_level, 0, "Log level: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL");
DEFINE_bool(log_to_console, true, "Log output to console");
DEFINE_bool(log_to_file, true, "Log output to file");
DEFINE_string(custom_log_dir, "logs", "Log directory"); // Modify to custom_log_dir to avoid conflicts with glog's log_dir
DEFINE_bool(verbose_log, false, "Enable verbose log");
DEFINE_bool(log_inference_details, false, "Record inference details");
DEFINE_bool(log_binding_info, false, "Record binding information");
DEFINE_bool(log_detection_boxes, false, "Record detection box details");
DEFINE_bool(log_batch_timing, true, "Log detailed batch inference timing for parallel inference verification");

namespace flags {

// Validator functions
static bool validateVideoReaderMode(const char* flagname, const std::string& value) {
    std::string lower_value = value;
    std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    return (lower_value == "opencv" || lower_value == "ffmpeg" || lower_value == "gstreamer");
}

static bool validateThreshold(const char* flagname, double value) {
    return (value >= 0.0 && value <= 1.0);
}

static bool validatePositiveInt(const char* flagname, int value) {
    return value > 0;
}

static bool validateNonNegativeInt(const char* flagname, int value) {
    return value >= 0;
}

static bool validatePreprocessMode(const char* flagname, int value) {
    return (value >= 0 && value <= 2); // 0=CPU, 1=mixed, 2=GPU
}

static bool validatePostprocessMode(const char* flagname, int value) {
    return (value >= 0 && value <= 1); // 0=CPU, 1=GPU
}

static bool validatePositiveDouble(const char* flagname, double value) {
    return value > 0.0;
}

static bool validateBool(const char* flagname, bool value) {
    return true;  // bool type is always valid
}

// Register validator
static const bool video_reader_validator = 
    gflags::RegisterFlagValidator(&FLAGS_video_reader_mode, &validateVideoReaderMode);

static const bool global_conf_validator = 
    gflags::RegisterFlagValidator(&FLAGS_global_conf_threshold, &validateThreshold);

static const bool global_nms_validator = 
    gflags::RegisterFlagValidator(&FLAGS_global_nms_threshold, &validateThreshold);

static const bool global_preprocess_validator = 
    gflags::RegisterFlagValidator(&FLAGS_global_preprocess_mode, &validatePreprocessMode);

static const bool global_postprocess_validator = 
    gflags::RegisterFlagValidator(&FLAGS_global_postprocess_mode, &validatePostprocessMode);

static const bool global_duration_validator = 
    gflags::RegisterFlagValidator(&FLAGS_global_duration, &validatePositiveInt);

static const bool local_conf_validator = 
    gflags::RegisterFlagValidator(&FLAGS_local_conf_threshold, &validateThreshold);

static const bool local_nms_validator = 
    gflags::RegisterFlagValidator(&FLAGS_local_nms_threshold, &validateThreshold);

static const bool local_preprocess_validator = 
    gflags::RegisterFlagValidator(&FLAGS_local_preprocess_mode, &validatePreprocessMode);

static const bool local_postprocess_validator = 
    gflags::RegisterFlagValidator(&FLAGS_local_postprocess_mode, &validatePostprocessMode);

static const bool local_duration_validator = 
    gflags::RegisterFlagValidator(&FLAGS_local_duration, &validatePositiveInt);

static const bool roi_margin_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_margin, &validateNonNegativeInt);

static const bool roi_min_size_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_min_size, &validatePositiveInt);

static const bool roi_max_size_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_max_size, &validatePositiveInt);

static const bool roi_overlap_threshold_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_overlap_threshold, &validateThreshold);

static const bool roi_max_no_detection_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_max_no_detection, &validatePositiveInt);

static const bool roi_memory_frames_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_memory_frames, &validatePositiveInt);

static const bool roi_target_scale_factor_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_target_scale_factor, &validatePositiveDouble);

// enable_adaptive_roi_size is bool type, no validator

static const bool roi_size_smooth_factor_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_size_smooth_factor, &validateThreshold);

static const bool track_high_thresh_validator = 
    gflags::RegisterFlagValidator(&FLAGS_track_high_thresh, &validateThreshold);

static const bool track_low_thresh_validator = 
    gflags::RegisterFlagValidator(&FLAGS_track_low_thresh, &validateThreshold);

static const bool new_track_thresh_validator = 
    gflags::RegisterFlagValidator(&FLAGS_new_track_thresh, &validateThreshold);

static const bool track_buffer_validator = 
    gflags::RegisterFlagValidator(&FLAGS_track_buffer, &validatePositiveInt);

static const bool match_thresh_validator = 
    gflags::RegisterFlagValidator(&FLAGS_match_thresh, &validateThreshold);

static const bool roi_match_thresh_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_match_thresh, &validateThreshold);

static const bool roi_spatial_tolerance_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_spatial_tolerance, &validatePositiveDouble);

static const bool roi_reactivate_thresh_validator = 
    gflags::RegisterFlagValidator(&FLAGS_roi_reactivate_thresh, &validateThreshold);

static const bool candidate_confirm_frames_validator = 
    gflags::RegisterFlagValidator(&FLAGS_candidate_confirm_frames, &validatePositiveInt);

static const bool candidate_window_validator = 
    gflags::RegisterFlagValidator(&FLAGS_candidate_window, &validatePositiveInt);

static const bool batch_size_validator = 
    gflags::RegisterFlagValidator(&FLAGS_batch_size, &validatePositiveInt);

static const bool num_threads_validator = 
    gflags::RegisterFlagValidator(&FLAGS_num_threads, &validatePositiveInt);

static const bool cost_iou_weight_validator = 
    gflags::RegisterFlagValidator(&FLAGS_cost_iou_weight, &validatePositiveDouble);

static const bool cost_roi_weight_validator = 
    gflags::RegisterFlagValidator(&FLAGS_cost_roi_weight, &validatePositiveDouble);

static const bool cost_radius_weight_validator = 
    gflags::RegisterFlagValidator(&FLAGS_cost_radius_weight, &validatePositiveDouble);

static const bool cost_spatial_weight_validator = 
    gflags::RegisterFlagValidator(&FLAGS_cost_spatial_weight, &validatePositiveDouble);

static const bool trajectory_radius_factor_validator = 
    gflags::RegisterFlagValidator(&FLAGS_trajectory_radius_factor, &validatePositiveDouble);

static const bool log_level_validator = 
    gflags::RegisterFlagValidator(&FLAGS_log_level, &validateNonNegativeInt);

void init(int* argc, char*** argv, const std::string& flagfile) {
    // Simplified parameter processing, using gflags's own flagfile mechanism
    std::ifstream file(flagfile);
    if (file.good()) {
        std::cout << "Loading configuration file: " << flagfile << std::endl;
        // Set flagfile parameter in the first position
        gflags::SetCommandLineOption("flagfile", flagfile.c_str());
    } else {
        std::cout << "Configuration file " << flagfile << " does not exist, using default configuration and command line parameters" << std::endl;
    }
    
    // Parse command line parameters
    gflags::ParseCommandLineFlags(argc, argv, true);
    
    // Print flag values
    printFlags();
}

void printFlags() {
    std::cout << "\n=== Configuration parameters ===" << std::endl;
    
    std::cout << "Video read mode: " << FLAGS_video_reader_mode << std::endl;
    
    std::cout << "\nGlobal model configuration:" << std::endl;
    std::cout << "  Confidence threshold: " << FLAGS_global_conf_threshold << std::endl;
    std::cout << "  NMS threshold: " << FLAGS_global_nms_threshold << std::endl;
    std::cout << "  Preprocess mode: " << FLAGS_global_preprocess_mode;
    switch (FLAGS_global_preprocess_mode) {
        case 0: std::cout << " (CPU)"; break;
        case 1: std::cout << " (CPU and GPU mixed)"; break;
        case 2: std::cout << " (GPU)"; break;
    }
    std::cout << std::endl;
    
    std::cout << "  Postprocess mode: " << FLAGS_global_postprocess_mode;
    switch (FLAGS_global_postprocess_mode) {
        case 0: std::cout << " (CPU)"; break;
        case 1: std::cout << " (GPU)"; break;
    }
    std::cout << std::endl;
    std::cout << "  Duration frames: " << FLAGS_global_duration << std::endl;
    
    std::cout << "\nLocal model configuration:" << std::endl;
    std::cout << "  Confidence threshold: " << FLAGS_local_conf_threshold << std::endl;
    std::cout << "  NMS threshold: " << FLAGS_local_nms_threshold << std::endl;
    std::cout << "  Preprocess mode: " << FLAGS_local_preprocess_mode;
    switch (FLAGS_local_preprocess_mode) {
        case 0: std::cout << " (CPU)"; break;
        case 1: std::cout << " (CPU and GPU mixed)"; break;
        case 2: std::cout << " (GPU)"; break;
    }
    std::cout << std::endl;
    
    std::cout << "  Postprocess mode: " << FLAGS_local_postprocess_mode;
    switch (FLAGS_local_postprocess_mode) {
        case 0: std::cout << " (CPU)"; break;
        case 1: std::cout << " (GPU)"; break;
    }
    std::cout << std::endl;
    std::cout << "  Duration frames: " << FLAGS_local_duration << std::endl;
    
    std::cout << "\nROI configuration:" << std::endl;
    std::cout << "  Margin: " << FLAGS_roi_margin << std::endl;
    std::cout << "  Minimum size: " << FLAGS_roi_min_size << std::endl;
    std::cout << "  Maximum size: " << FLAGS_roi_max_size << std::endl;
    std::cout << "  Overlap threshold: " << FLAGS_roi_overlap_threshold << std::endl;
    std::cout << "  Maximum no detection frames: " << FLAGS_roi_max_no_detection << std::endl;
    std::cout << "  Memory frames: " << FLAGS_roi_memory_frames << std::endl;
    std::cout << "  Adaptive ROI size: " << (FLAGS_enable_adaptive_roi_size ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Adaptive ROI scale factor: " << FLAGS_roi_target_scale_factor << std::endl;
    std::cout << "  Adaptive ROI size smooth factor: " << FLAGS_roi_size_smooth_factor << std::endl;
    
    std::cout << "\nTracking configuration:" << std::endl;
    std::cout << "  High confidence threshold: " << FLAGS_track_high_thresh << std::endl;
    std::cout << "  Low confidence threshold: " << FLAGS_track_low_thresh << std::endl;
    std::cout << "  New track threshold: " << FLAGS_new_track_thresh << std::endl;
    std::cout << "  Tracking buffer: " << FLAGS_track_buffer << std::endl;
    std::cout << "  Match threshold: " << FLAGS_match_thresh << std::endl;
    
    std::cout << "\nROI constrained tracking configuration:" << std::endl;
    std::cout << "  ROI match threshold: " << FLAGS_roi_match_thresh << std::endl;
    std::cout << "  ROI spatial tolerance: " << FLAGS_roi_spatial_tolerance << std::endl;
    std::cout << "  ROI reactivate threshold: " << FLAGS_roi_reactivate_thresh << std::endl;
    
    std::cout << "\nCandidate target configuration:" << std::endl;
    std::cout << "  Confirm frames: " << FLAGS_candidate_confirm_frames << std::endl;
    std::cout << "  Window size: " << FLAGS_candidate_window << std::endl;
    
    std::cout << "\nPerformance configuration:" << std::endl;
    std::cout << "  Batch inference: " << (FLAGS_enable_batch_inference ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Batch size: " << FLAGS_batch_size << std::endl;
    std::cout << "  Parallel processing: " << (FLAGS_enable_parallel_processing ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Threads: " << FLAGS_num_threads << std::endl;
    std::cout << "  Appearance extraction: " << (FLAGS_enable_appearance ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Single thread mode: " << (FLAGS_single_thread_mode ? "Enabled" : "Disabled") << std::endl;
    
    std::cout << "\nAlgorithm selection configuration:" << std::endl;
    std::cout << "  Use original ByteTrack: " << (FLAGS_use_original_bytetrack ? "Enabled" : "Disabled") << std::endl;
    
    std::cout << "\nFusion cost matrix weight configuration:" << std::endl;
    std::cout << "  IOU cost weight: " << FLAGS_cost_iou_weight << std::endl;
    std::cout << "  ROI cost weight: " << FLAGS_cost_roi_weight << std::endl;
    std::cout << "  History trajectory circular cost weight: " << FLAGS_cost_radius_weight << std::endl;
    std::cout << "  Spatial distance cost weight: " << FLAGS_cost_spatial_weight << std::endl;
    std::cout << "  History trajectory radius factor: " << FLAGS_trajectory_radius_factor << std::endl;
    std::cout << "  Use enhanced matching: " << (FLAGS_use_enhanced_matching ? "Enabled" : "Disabled") << std::endl;
    
    std::cout << "\nLog configuration:" << std::endl;
    std::cout << "  Log level: " << FLAGS_log_level << std::endl;
    std::cout << "  Console output: " << (FLAGS_log_to_console ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  File output: " << (FLAGS_log_to_file ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Log directory: " << FLAGS_custom_log_dir << std::endl;
    std::cout << "  Detailed log: " << (FLAGS_verbose_log ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Inference details: " << (FLAGS_log_inference_details ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Binding information: " << (FLAGS_log_binding_info ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Detection box details: " << (FLAGS_log_detection_boxes ? "Enabled" : "Disabled") << std::endl;
    
    std::cout << "\nYou can modify the config.flag file or use command line parameters to change these settings\n" << std::endl;
}

VideoReaderMode getVideoReaderMode() {
    std::string mode = FLAGS_video_reader_mode;
    std::transform(mode.begin(), mode.end(), mode.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    if (mode == "ffmpeg") {
        return VideoReaderMode::FFMPEG;
    } else if (mode == "gstreamer") {
        return VideoReaderMode::GSTREAMER;
    } else {
        // Default or unknown value using OpenCV
        return VideoReaderMode::OPENCV;
    }
}

PreprocessMode getGlobalPreprocessMode() {
    switch (FLAGS_global_preprocess_mode) {
        case 0: return PreprocessMode::CPU;
        case 1: return PreprocessMode::MIXED;
        case 2: return PreprocessMode::GPU;
        default: return PreprocessMode::CPU; // Default using CPU mode
    }
}

PostprocessMode getGlobalPostprocessMode() {
    return (FLAGS_global_postprocess_mode == 1) ? PostprocessMode::GPU : PostprocessMode::CPU;
}

PreprocessMode getLocalPreprocessMode() {
    switch (FLAGS_local_preprocess_mode) {
        case 0: return PreprocessMode::CPU;
        case 1: return PreprocessMode::MIXED;
        case 2: return PreprocessMode::GPU;
        default: return PreprocessMode::CPU; // Default using CPU mode
    }
}

PostprocessMode getLocalPostprocessMode() {
    return (FLAGS_local_postprocess_mode == 1) ? PostprocessMode::GPU : PostprocessMode::CPU;
}

bool validateFlags() {
    bool valid = true;
    
    // Check ROI minimum size <= maximum size
    if (FLAGS_roi_min_size > FLAGS_roi_max_size) {
        LOG_ERROR("Invalid ROI size configuration: minimum size(" << FLAGS_roi_min_size 
                 << ") greater than maximum size(" << FLAGS_roi_max_size << ")");
        valid = false;
    }
    

    return valid;
}

}  // namespace flags 