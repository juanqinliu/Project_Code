// Multithreaded implementation
#include "MultiThreadTrackingApp.h"
#include "video/VideoReaderFactory.h"
#include <filesystem>
#include <chrono>
#include <iomanip>  
#include <ctime>
#include "common/Flags.h"
#include "common/Flags.h"
#include <atomic>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace fs = std::filesystem;
namespace tracking {

// Static members initialization
const int READING_BUFFER_SIZE = 3;   
const int PROCESSING_BUFFER_SIZE = 2; 
const int VIDEO_SAVE_BUFFER_SIZE = 100;  

// Inter-thread queues
std::queue<FrameItem> reading_queue;
std::queue<ProcessedItem> processing_queue;
std::queue<VideoSaveItem> video_save_queue; 

// Mutexes
std::mutex reading_mutex;
std::mutex processing_mutex;
std::mutex video_save_mutex;

// Condition variables
std::condition_variable reading_not_full;
std::condition_variable reading_not_empty;
std::condition_variable processing_not_full;
std::condition_variable processing_not_empty;
std::condition_variable video_save_not_full;
std::condition_variable video_save_not_empty;

// Thread control flags
std::atomic<bool> video_finished(false);
std::atomic<bool> processing_finished(false);
std::atomic<bool> display_finished(false);
std::atomic<bool> should_exit(false);
std::atomic<bool> is_paused(false);

// Frame order validation
std::atomic<int> last_processed_frame_number(-1);
std::atomic<int> last_displayed_frame_number(-1);

// Mode switch flags
std::atomic<bool> mode_switching(false);
std::atomic<TrackingMode> requested_mode(TrackingMode::GLOBAL);

// Performance statistics
std::atomic<int> actual_processed_frames(0);
std::atomic<double> video_fps(0.0);
std::atomic<long long> total_delay_ms(0);
std::atomic<long long> total_process_time_ms(0);
std::atomic<long long> total_read_time_ms(0);
std::atomic<long long> total_display_time_ms(0);

// FPS counters (short-window for runtime stats)
std::atomic<double> real_processing_fps(0.0);
std::atomic<double> real_display_fps(0.0);
std::atomic<int> processed_frames_count(0);
std::atomic<int> displayed_frames_count(0);
std::atomic<std::chrono::steady_clock::time_point> last_process_time{std::chrono::steady_clock::now()};
std::atomic<std::chrono::steady_clock::time_point> last_display_time{std::chrono::steady_clock::now()};

// Latency budget and throttle factor
static constexpr int TARGET_LATENCY_BUDGET_MS = 80;
static constexpr double THROTTLE_SAFETY = 1.05;

// Processing mode: deterministic processing, no frame dropping
static constexpr int MAX_ACCEPTABLE_DELAY_MS = 200;

// Fine-grained read/display timings (cross-thread accumulation)
namespace {
std::atomic<long long> g_read_call_time_ms{0};
std::atomic<long long> g_read_call_count{0};
// Count of source frame intervals (successful read intervals)
std::atomic<long long> g_read_interval_count{0};

std::atomic<long long> g_disp_overlay_time_ms{0};
std::atomic<long long> g_disp_imshow_time_ms{0};
std::atomic<long long> g_disp_write_time_ms{0};
std::atomic<long long> g_disp_waitkey_time_ms{0};
std::atomic<long long> g_disp_frames{0};

// Lock and queue wait statistics
std::atomic<long long> g_wait_lock_reading_mutex_ms{0};
std::atomic<long long> g_wait_lock_reading_mutex_cnt{0};
std::atomic<long long> g_wait_lock_processing_mutex_ms{0};
std::atomic<long long> g_wait_lock_processing_mutex_cnt{0};

std::atomic<long long> g_wait_reading_not_full_ms{0};
std::atomic<long long> g_wait_reading_not_full_cnt{0};
std::atomic<long long> g_wait_reading_not_empty_ms{0};
std::atomic<long long> g_wait_reading_not_empty_cnt{0};
std::atomic<long long> g_wait_processing_not_full_ms{0};
std::atomic<long long> g_wait_processing_not_full_cnt{0};
std::atomic<long long> g_wait_processing_not_empty_ms{0};
std::atomic<long long> g_wait_processing_not_empty_cnt{0};
}

// Overall FPS measurement: from first successful frame read to last frame display/output
std::atomic<bool> g_overall_first_seen{false};
std::atomic<std::chrono::steady_clock::time_point> g_overall_first_ts{std::chrono::steady_clock::now()};
std::atomic<std::chrono::steady_clock::time_point> g_overall_last_ts{std::chrono::steady_clock::now()};

MultiThreadTrackingApp::MultiThreadTrackingApp(
    const std::string& global_model_path, 
    const std::string& local_model_path,
    const std::string& video_path,
    const std::string& output_dir)
    : global_model_path_(global_model_path),
      local_model_path_(local_model_path),
      video_path_(video_path),
      output_dir_(output_dir),
      video_reader_(nullptr),
      total_frames_(0),
      fps_(0),
      frame_width_(0),
      frame_height_(0),
      enable_ros_publishing_(false) {
    
    // Build output file base name from input path; for live sources use timestamp
    fs::path video_file_path(video_path);
    auto isDigitString = [](const std::string &s) {
        return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
    };
    auto startsWith = [](const std::string &s, const char *prefix) {
        return s.rfind(prefix, 0) == 0;
    };
    auto makeTimestampName = []() {
        auto now = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        std::ostringstream oss;
        oss << "capture_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return oss.str();
    };

    std::string candidate = video_file_path.stem().string();
    bool path_exists = fs::exists(video_file_path);
    bool no_ext = video_file_path.has_extension() == false;
    bool no_parent = video_file_path.parent_path().empty();
    bool is_live_source = false;
    
    if (path_exists && video_file_path.has_extension()) {
        // If the path exists and has an extension, it is probably a video file
        is_live_source = false;
        video_basename_ = candidate;
    } else if (isDigitString(video_path)) {
        // Pure number: camera index
        is_live_source = true;
        video_basename_ = makeTimestampName();
    } else if (startsWith(video_path, "rtsp://") || startsWith(video_path, "rtsps://") ||
               startsWith(video_path, "http://") || startsWith(video_path, "https://") ||
               startsWith(video_path, "v4l2://") || startsWith(video_path, "camera://")) {
        // Clear stream protocol
        is_live_source = true;
        video_basename_ = makeTimestampName();
    } else if (startsWith(video_path, "/dev/video")) {
        // Linux device file
        is_live_source = true;
        video_basename_ = makeTimestampName();
    } else if (startsWith(video_path, "v4l2src") || startsWith(video_path, "nvarguscamerasrc") || startsWith(video_path, "rtspsrc")) {
        // GStreamer pipeline
        is_live_source = true;
        video_basename_ = makeTimestampName();
    } else if (!path_exists && no_ext && no_parent && 
               (startsWith(video_path, "video") || startsWith(video_path, "cam"))) {
        // Only consider video/cam prefix if the path does not exist, has no extension, and has no parent directory
        is_live_source = true;
        video_basename_ = makeTimestampName();
    } else {
        // Default case: video file
        is_live_source = false;
        if (candidate.empty()) {
            video_basename_ = "unknown_video";
        } else {
            video_basename_ = candidate;
        }
    }
    
    // Processing mode: always deterministic processing, no frame dropping
    LOG_INFO("Deterministic processing mode - all frames will be processed in order (source type: " 
             << (is_live_source ? "live" : "file") << ")");
    
    // Ensure output directory exists
    fs::create_directories(output_dir);
    
    // Output video path
    output_video_path_ = (fs::path(output_dir) / (video_basename_ + ".mp4")).string();
    
    // Check if ROS2 publishing is enabled
    const char* enable_ros_env = std::getenv("ENABLE_ROS_PUBLISHING");
    if (enable_ros_env && std::string(enable_ros_env) == "true") {
        enable_ros_publishing_ = true;
        try {
            // Initialize ROS2 node
            if (!rclcpp::ok()) {
                rclcpp::init(0, nullptr);
            }
            ros2_node_ = std::make_shared<GLDTRos2Node>();
            LOG_INFO("ROS2 node initialized, topic publishing enabled");
        } catch (const std::exception& e) {
            LOG_ERROR("ROS2 initialization failed: " << e.what());
            enable_ros_publishing_ = false;
            ros2_node_ = nullptr;
        }
    } else {
        enable_ros_publishing_ = false;
        ros2_node_ = nullptr;
        LOG_INFO("ROS2 publishing disabled");
    }
    
    // Initialize config and tracking system
    config_ = std::make_unique<tracking::Config>();
    system_ = std::make_unique<tracking::TrackingDetectionSystem>(
        global_model_path_, local_model_path_, *config_);
}

MultiThreadTrackingApp::~MultiThreadTrackingApp() {
    cleanup();
}

void MultiThreadTrackingApp::run() {
    // Print startup banner (bypasses log_level)
    std::cout << "\n========================================" << std::endl;
    std::cout << "  GLDT Tracking System Starting" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Video Source: " << video_path_ << std::endl;
    std::cout << "Output Directory: " << output_dir_ << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    LOG_INFO("Initializing video...");
    if (!initializeVideo()) {
        LOG_ERROR("Failed to initialize video");
        std::cerr << "ERROR: Failed to initialize video!" << std::endl;
        return;
    }
    
    // Print video info (bypasses log_level)
    std::cout << "Video Info: " << frame_width_ << "x" << frame_height_ 
              << " @ " << fps_ << " FPS" << std::endl;
    if (total_frames_ > 0) {
        std::cout << "Total Frames: " << total_frames_ << std::endl;
    }
    std::cout << std::endl;
    
    if (FLAGS_enable_video_output) {
        LOG_INFO("Initializing output video...");
        initializeOutputVideo();
        std::cout << "Output Video: " << output_video_path_ << std::endl;
    } else {
        LOG_INFO("Video output disabled for higher FPS");
        std::cout << "Video output: Disabled" << std::endl;
    }
    std::cout << std::endl;
    
    LOG_INFO("Starting multithread processing...");
    LOG_INFO("Architecture: Reader -> Processor -> Display (with synchronous video save)");
    
    std::cout << "Starting processing...\n" << std::endl;
    
    // Record start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Reset flags and counters (preserve should_exit from signal handler)
    video_finished = false;
    processing_finished = false;
    display_finished = false;
    is_paused = false;
    actual_processed_frames = 0;
    total_delay_ms = 0;
    total_process_time_ms = 0;
    total_read_time_ms = 0;
    total_display_time_ms = 0;
    
    // Reset short-window FPS variables
    real_processing_fps = 0.0;
    real_display_fps = 0.0;
    processed_frames_count = 0;
    displayed_frames_count = 0;
    last_process_time = std::chrono::steady_clock::now();
    last_display_time = std::chrono::steady_clock::now();
    g_overall_first_seen = false;
    g_overall_first_ts = std::chrono::steady_clock::now();
    g_overall_last_ts = g_overall_first_ts.load();
    
    // Reset frame order validation
    last_processed_frame_number = -1;
    last_displayed_frame_number = -1;
    
    // Clear queues
    while (!reading_queue.empty()) reading_queue.pop();
    while (!processing_queue.empty()) processing_queue.pop();
    while (!video_save_queue.empty()) video_save_queue.pop();
    
    // Start threads (video saver first, then processor/display, reader last)
    std::thread video_save_thread;
    if (FLAGS_enable_video_output) {
        video_save_thread = std::thread(&MultiThreadTrackingApp::asyncVideoSaveThread, this);
        LOG_INFO("Async video save thread started");
    }
    std::thread processor_thread(&MultiThreadTrackingApp::frameProcessorThread, this);
    std::thread display_thread(&MultiThreadTrackingApp::resultDisplayThread, this);
    std::thread reader_thread(&MultiThreadTrackingApp::videoReaderThread, this);
    
    // Join threads - ensure all threads complete normally
    LOG_INFO("Waiting for threads to finish...");
    
    // Reader thread usually completes quickly
    if (reader_thread.joinable()) {
        reader_thread.join();
        LOG_INFO("Reader thread joined successfully");
    }
    
    // Processor and Display threads need to wait for the queue to be processed
    // Use a longer timeout, or completely block until completion
    if (processor_thread.joinable()) {
        processor_thread.join();
        LOG_INFO("Processor thread joined successfully");
    }
    
    if (display_thread.joinable()) {
        display_thread.join();
        LOG_INFO("Display thread joined successfully");
    }
    
    // Join video save thread last
    if (FLAGS_enable_video_output && video_save_thread.joinable()) {
        video_save_thread.join();
        LOG_INFO("Video save thread joined successfully");
    }
    
    LOG_INFO("Multithread processing finished");
    
    // Compute total time
    auto end_time = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double total_time_s = total_time_ms / 1000.0;
    
    // Performance summary
    LOG_INFO("Total processed frames: " << actual_processed_frames);
    if (total_delay_ms > 0 && actual_processed_frames > 0) {
        double avg_delay = static_cast<double>(total_delay_ms) / actual_processed_frames;
        LOG_INFO("Average latency: " << avg_delay << " ms");
    }
    
    // Average processing time per frame
    if (total_process_time_ms > 0 && actual_processed_frames > 0) {
        double avg_process_time = static_cast<double>(total_process_time_ms) / actual_processed_frames;
        LOG_INFO("Average processing time: " << avg_process_time << " ms/frame");
    }
    
    // Average source interval and Source FPS (input cadence)
    double avg_source_interval = 0.0;
    double source_fps_metric = 0.0;
    if (total_read_time_ms > 0 && g_read_interval_count > 0) {
        avg_source_interval = static_cast<double>(total_read_time_ms) / g_read_interval_count.load();
        if (avg_source_interval > 0.0) source_fps_metric = 1000.0 / avg_source_interval;
        LOG_INFO("Average source interval: " << std::fixed << std::setprecision(1) << avg_source_interval << " ms  (Source FPS: "
                 << std::fixed << std::setprecision(1) << source_fps_metric << ")");
    }
    
    // Average display time per frame
    if (total_display_time_ms > 0 && actual_processed_frames > 0) {
        double avg_display_time = static_cast<double>(total_display_time_ms) / actual_processed_frames;
        LOG_INFO("Average display time: " << avg_display_time << " ms/frame");
    }
    
    // Validate latency breakdown
    if (total_delay_ms > 0 && actual_processed_frames > 0) {
        double avg_delay = static_cast<double>(total_delay_ms) / actual_processed_frames;
        double avg_process = static_cast<double>(total_process_time_ms) / actual_processed_frames;
        double avg_display = static_cast<double>(total_display_time_ms) / actual_processed_frames;
        
        LOG_INFO("Latency analysis - average end-to-end: " << std::fixed << std::setprecision(1) << avg_delay << " ms");
        LOG_INFO("Latency breakdown - processing: " << std::fixed << std::setprecision(1) << avg_process << " ms");
        LOG_INFO("Latency breakdown - display: " << std::fixed << std::setprecision(1) << avg_display << " ms");
        
        // Queue wait = total latency - processing - display
        double queue_wait_time = avg_delay - (avg_process + avg_display);
        LOG_INFO("Queue wait time: " << std::fixed << std::setprecision(1) << queue_wait_time << " ms");
        
        if (queue_wait_time > 100) {
            LOG_INFO("Warning: Queue wait is too long; consider increasing queue size or optimizing processing");
        }
    }
    
    // Only output standardized FPS at the summary below

    // Unified FPS Summary: from first successful frame read to last frame display
    double pipeline_fps = 0.0;
    if (g_overall_first_seen.load() && actual_processed_frames > 1) {
        auto t0 = g_overall_first_ts.load();
        auto t1 = g_overall_last_ts.load();
        auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (dur_ms > 0) pipeline_fps = (actual_processed_frames.load() * 1000.0) / dur_ms;
    } else if (total_time_s > 0.0 && actual_processed_frames > 0) {
        // Fallback: use start_time from run() (may include thread startup/shutdown overhead)
        pipeline_fps = (actual_processed_frames.load() / total_time_s);
    }
    double avg_proc_ms_final = (actual_processed_frames > 0)
                                   ? (static_cast<double>(total_process_time_ms.load()) / actual_processed_frames.load())
                                   : 0.0;
    double processing_fps = (avg_proc_ms_final > 0.0) ? (1000.0 / avg_proc_ms_final) : 0.0;
    // Source FPS based on read intervals
    double source_fps_summary = 0.0;
    if (total_read_time_ms > 0 && g_read_interval_count > 0) {
        double avg_src_interval_ms = static_cast<double>(total_read_time_ms.load()) / std::max(1LL, g_read_interval_count.load());
        if (avg_src_interval_ms > 0.0) source_fps_summary = 1000.0 / avg_src_interval_ms;
    }
    // Display FPS is a short-window stat; ignore if display disabled
    double display_actual_fps = real_display_fps.load();

    // Use std::cout for performance summary (bypasses log_level filtering, always visible)
    std::cout << "\n================ Performance Summary ===============" << std::endl;
    std::cout << "Total Frames Processed: " << actual_processed_frames.load() << std::endl;
    std::cout << "Overall Average FPS: " << std::fixed << std::setprecision(1) << pipeline_fps << std::endl;
    std::cout << "Source FPS: " << std::fixed << std::setprecision(1) << source_fps_summary << std::endl;
    std::cout << "Processing FPS: " << std::fixed << std::setprecision(1) << processing_fps << std::endl;
    std::cout << "Display FPS: " << std::fixed << std::setprecision(1) << display_actual_fps << std::endl;
    std::cout << "Average Latency: " << std::fixed << std::setprecision(1) 
              << (actual_processed_frames > 0 ? static_cast<double>(total_delay_ms) / actual_processed_frames : 0.0) << " ms" << std::endl;

    // Detailed timing breakdown (optional, only for verbose analysis)
    if (g_read_call_count > 0) {
        double avg_read_call = static_cast<double>(g_read_call_time_ms.load()) / g_read_call_count.load();
        std::cout << "[Read Thread] Average Read Time: " << std::fixed << std::setprecision(2) << avg_read_call << " ms" << std::endl;
    }
    if (g_disp_frames > 0) {
        double avg_overlay = static_cast<double>(g_disp_overlay_time_ms.load()) / g_disp_frames.load();
        double avg_imshow = static_cast<double>(g_disp_imshow_time_ms.load()) / g_disp_frames.load();
        double avg_write = static_cast<double>(g_disp_write_time_ms.load()) / g_disp_frames.load();
        double avg_waitkey = static_cast<double>(g_disp_waitkey_time_ms.load()) / g_disp_frames.load();
        std::cout << "[Display Thread] Overlay: " << std::fixed << std::setprecision(2) << avg_overlay
                  << "ms, Show: " << avg_imshow
                  << "ms, Write: " << avg_write
                  << "ms, WaitKey: " << avg_waitkey << "ms" << std::endl;
    }

    auto safe_avg = [](long long sum, long long cnt) { return cnt > 0 ? (static_cast<double>(sum) / cnt) : 0.0; };
    std::cout << "[Queue Waiting] Read: " << std::fixed << std::setprecision(2)
              << safe_avg(g_wait_reading_not_empty_ms, g_wait_reading_not_empty_cnt) / 1000.0
              << "ms, Process: " << safe_avg(g_wait_processing_not_empty_ms, g_wait_processing_not_empty_cnt) / 1000.0 << "ms" << std::endl;
    
    std::cout << "====================================================\n" << std::endl;


    if (system_) {
        system_->saveResults(output_dir_, video_basename_);
    }
    
    // Output completion banner (bypasses log_level)
    std::cout << "\n=============== Processing Completed ===============" << std::endl;
    // Output save path info
    std::string results_file_path = (fs::path(output_dir_) / (video_basename_ + ".txt")).string();
    std::cout << "Results Saved: " << results_file_path << std::endl;
    
    if (FLAGS_enable_video_output) {
        std::cout << "Video Saved: " << output_video_path_ << std::endl;
    } else {
        std::cout << "Video Output: Disabled" << std::endl;
    }
    std::cout << "====================================================\n" << std::endl;
}

bool MultiThreadTrackingApp::initializeVideo() {
    LOG_INFO("Open Video: " << video_path_);
    if (!initializeVideoReader()) {
        LOG_ERROR("Video Reader Initialization Failed");
        return false;
    }
    LOG_INFO("Video Reader Initialization Success");
    return true;
}

void MultiThreadTrackingApp::initializeOutputVideo() {
    LOG_INFO("Create Output Video: " << output_video_path_);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    // Always use original resolution for video saving (high quality output)
    // Display downscaling only affects on-screen rendering, not saved video
    int out_w = frame_width_;
    int out_h = frame_height_;
    
    LOG_INFO("Output video resolution: " << out_w << "x" << out_h << " (original resolution)");
    if (video_reader_ && video_reader_->isDownscalingEnabled()) {
        auto target = video_reader_->getTargetResolution();
        LOG_INFO("Display resolution: " << target.first << "x" << target.second << " (downscaled for performance)");
    }
    
    video_writer_.open(output_video_path_, fourcc, fps_, cv::Size(out_w, out_h));
    
    if (!video_writer_.isOpened()) {
        LOG_ERROR("Cannot Create Output Video: " << output_video_path_);
    }
}

void MultiThreadTrackingApp::cleanup() {
    LOG_INFO("Cleanup Resources...");
    should_exit = true;
    display_finished = true;
    
    // Wake up all waiting threads
    reading_not_full.notify_all();
    reading_not_empty.notify_all();
    processing_not_full.notify_all();
    processing_not_empty.notify_all();
    video_save_not_full.notify_all();
    video_save_not_empty.notify_all();
    
    if (video_reader_) {
        video_reader_->close();
        video_reader_.reset();
    }
    
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
}

bool MultiThreadTrackingApp::initializeVideoReader() {
    if (video_reader_) {
        LOG_INFO("Video Reader Already Initialized");
        return true;
    }
    
    // Factory registration and type selection logic
    auto modeToReaderType = [](flags::VideoReaderMode mode) {
        switch (mode) {
            case flags::VideoReaderMode::FFMPEG:
                return video::VideoReaderType::FFMPEG;
            case flags::VideoReaderMode::GSTREAMER:
                return video::VideoReaderType::GSTREAMER;
            default:
                return video::VideoReaderType::OPENCV;
        }
    };
    video::VideoReaderType reader_type = modeToReaderType(flags::getVideoReaderMode());
    auto registered_types = video::VideoReaderFactory::getRegisteredReaderTypes();
    bool type_registered = false;
    for (const auto& type : registered_types) {
        if (type == reader_type) {
            type_registered = true;
            break;
        }
    }
    if (!type_registered) {
        if (!registered_types.empty()) {
            reader_type = registered_types[0];
            LOG_WARNING("The Selected Video Reader Type is Not Registered, Using: " << video::videoReaderTypeToString(reader_type));
        } else {
            LOG_ERROR("No Available Video Reader Type");
            return false;
        }
    }
    video::VideoReaderFactory::setDefaultReaderType(reader_type);
    LOG_INFO("Using Video Reader: " << video::videoReaderTypeToString(reader_type));
    
    std::vector<video::VideoReaderType> try_order;
    auto push_unique = [&](video::VideoReaderType t){
        if (std::find(try_order.begin(), try_order.end(), t) == try_order.end()) try_order.push_back(t);
    };
    push_unique(reader_type);
    push_unique(video::VideoReaderType::GSTREAMER);
    push_unique(video::VideoReaderType::FFMPEG);
    push_unique(video::VideoReaderType::OPENCV);
    for (const auto& t : registered_types) {
        push_unique(t);
    }
    bool opened = false;
    for (const auto& t : try_order) {
        video::VideoReaderFactory::setDefaultReaderType(t);
        LOG_INFO("Try Video Reader: " << video::videoReaderTypeToString(t));
        video_reader_ = video::VideoReaderFactory::createVideoReader();
        if (video_reader_ && video_reader_->open(video_path_)) {
            LOG_INFO("Using Video Reader: " << video::videoReaderTypeToString(t));
            opened = true;
            break;
        }
    }
    if (!opened) {
        LOG_ERROR("Cannot Open Video: " << video_path_);
        return false;
    }
    
    frame_width_ = video_reader_->getWidth();
    frame_height_ = video_reader_->getHeight();
    total_frames_ = video_reader_->getTotalFrames();
    fps_ = video_reader_->getFPS();
    
    LOG_INFO("Video info (for inference): " << frame_width_ << "x" << frame_height_
             << ", FPS=" << fps_ << ", Total frames=" << total_frames_);
    if (video_reader_->isDownscalingEnabled()) {
        auto target_res = video_reader_->getTargetResolution();
        LOG_INFO("Display downscaling: " << frame_width_ << "x" << frame_height_
                 << " -> " << target_res.first << "x" << target_res.second);
    } else {
        LOG_INFO("Display keeps original resolution (no downscaling)");
    }
    
    if (frame_width_ <= 0 || frame_height_ <= 0) {
        LOG_ERROR("Invalid video size");
        return false;
    }
    return true;
}

void MultiThreadTrackingApp::videoReaderThread() {
    LOG_INFO("Video reader thread started");
    int frame_number = 0;  // Local frame counter to ensure continuity
    cv::Mat frame;
    const int READ_LOG_INTERVAL = 30;
    
    auto last_read_tick = std::chrono::steady_clock::now();
    
    // Check video reader initialization
    if (!isVideoReaderInitialized()) {
        LOG_ERROR("Video reader not initialized, exiting read thread");
        should_exit = true;
        reading_not_empty.notify_all();
        return;
    }
    
    // Check if we should exit immediately
    if (should_exit) {
        LOG_INFO("Video reader thread exiting due to should_exit flag");
        reading_not_empty.notify_all();
        return;
    }

    while (!should_exit) {
        if (is_paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        {
            // Wait for space in read queue
            auto t_lock_start = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(reading_mutex);
            auto t_lock_end = std::chrono::steady_clock::now();
            g_wait_lock_reading_mutex_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_lock_end - t_lock_start).count();
            g_wait_lock_reading_mutex_cnt++;
            auto t_wait_start = std::chrono::steady_clock::now();
            // Deterministic backpressure: simple blocking wait when queue is full
            while (reading_queue.size() >= READING_BUFFER_SIZE && !should_exit) {
                reading_not_full.wait(lock, [&] {
                    return reading_queue.size() < READING_BUFFER_SIZE || should_exit;
                });
            }
            // Also backpressure when processing queue near capacity
            while (!should_exit) {
                size_t proc_q_size = 0;
                {
                    std::lock_guard<std::mutex> lk(processing_mutex);
                    proc_q_size = processing_queue.size();
                }
                if (proc_q_size <= PROCESSING_BUFFER_SIZE * 0.8) break;
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                lock.lock();
            }
            auto t_wait_end = std::chrono::steady_clock::now();
            g_wait_reading_not_full_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
            g_wait_reading_not_full_cnt++;
            
            if (should_exit) break;
            
            // If still full, briefly spin (no drop)
            if (reading_queue.size() >= READING_BUFFER_SIZE) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }

            // Read frame (IO outside lock)
            lock.unlock();
        }

        auto read_call_start = std::chrono::steady_clock::now();
        int reader_frame_number = 0;
        bool read_ok = video_reader_->readNextFrame(frame, reader_frame_number);
        auto read_call_end = std::chrono::steady_clock::now();
        g_read_call_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(read_call_end - read_call_start).count();
        g_read_call_count++;

        {
            std::unique_lock<std::mutex> lock(reading_mutex);
            if (read_ok) {
                auto now = std::chrono::steady_clock::now();
                
                // Compute read interval since last successful read
                static auto last_read_time = now;
                auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_read_time).count();
                if (frame_number > 0) {
                    total_read_time_ms += read_time;
                    g_read_interval_count++;
                }
                last_read_time = now;
                last_read_tick = now;
                
                // Use locally maintained continuous frame number for determinism
                tracking::FrameItem qitem;
                qitem.frame = frame.clone();
                qitem.frame_number = frame_number;
                qitem.timestamp = now;
                if (video_reader_) {
                    if (video_reader_->supportsTimestamps()) {
                        qitem.source_pts_ms = video_reader_->getLastFramePtsMs();
                        qitem.upstream_latency_ms = video_reader_->getLastUpstreamLatencyMs();
                    }
                }
                // Mark overall start time: first successful frame read
                if (!g_overall_first_seen.load()) {
                    g_overall_first_ts = now;
                    g_overall_first_seen = true;
                }
                reading_queue.push(std::move(qitem));

                if (READ_LOG_INTERVAL > 0 && (frame_number % READ_LOG_INTERVAL == 0)) {
                    LOG_INFO("Read thread: got frame " << frame_number 
                             << " (reader frame: " << reader_frame_number << ")");
                }
                
                frame_number++;
                
                // Notify processor there is a new frame
                lock.unlock();
                reading_not_empty.notify_one();
            } else {
                // Reading finished
                video_finished = true;
                LOG_INFO("Reading finished, total frames read: " << frame_number);
                
                lock.unlock();
                reading_not_empty.notify_all();
                break;
            }
        }
    }
    
    LOG_INFO("Video reader thread exited");
}

void MultiThreadTrackingApp::frameProcessorThread() {
    LOG_INFO("Frame processor thread started");
    
    while (!should_exit) {
        if (is_paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        
        // Handle mode switch if requested
        if (mode_switching) {
            handleModeSwitch(requested_mode);
            mode_switching = false;
        }
        
        // Pop a frame from read queue
        FrameItem item;
        bool got_frame = false;
        
        {
            auto t_lock_start = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(reading_mutex);
            auto t_lock_end = std::chrono::steady_clock::now();
            g_wait_lock_reading_mutex_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_lock_end - t_lock_start).count();
            g_wait_lock_reading_mutex_cnt++;
            auto t_wait_start = std::chrono::steady_clock::now();
            // Deterministic waiting for processor thread
            if (reading_queue.empty() && !video_finished && !should_exit) {
                reading_not_empty.wait(lock, [&] { 
                    return !reading_queue.empty() || video_finished || should_exit; 
                });
            }
            auto t_wait_end = std::chrono::steady_clock::now();
            g_wait_reading_not_empty_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
            g_wait_reading_not_empty_cnt++;
            
            if (should_exit) break;
            
            if (!reading_queue.empty()) {
                item = reading_queue.front();
                reading_queue.pop();
                got_frame = true;
                reading_not_full.notify_one();
            } else if (video_finished) {
                break; // no more frames
            } else {
                // Queue empty but video not finished, short wait then retry
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10)); // reduce waiting time
                continue;
            }
        }
        
        // Process frame
        if (got_frame) {
            // Frame order validation (for debugging)
            int expected_next = last_processed_frame_number.load() + 1;
            if (item.frame_number != expected_next && expected_next > 0) {
                LOG_WARNING("Frame order issue: expected " << expected_next 
                           << ", got " << item.frame_number 
                           << " (diff: " << (item.frame_number - expected_next) << ")");
            }
            last_processed_frame_number = item.frame_number;
            
            auto process_start_time = std::chrono::steady_clock::now();
            
            // Run tracking system
            cv::Mat vis_frame;
            std::vector<tracking::Detection> detections;
            std::vector<std::unique_ptr<tracking::STrack>> tracks;
            
            system_->process(item.frame, vis_frame, detections, tracks);
            
            auto process_end_time = std::chrono::steady_clock::now();
            
            // Accumulate processing time
            auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(process_end_time - process_start_time).count();
            total_process_time_ms += process_time;
            
            // Push result to processing queue
            ProcessedItem processed;
            processed.frame = item.frame;
            processed.vis_frame = std::move(vis_frame);
            processed.detections = std::move(detections);
            processed.tracks = std::move(tracks);
            processed.frame_number = item.frame_number;
            processed.process_start_time = process_start_time;
            processed.process_end_time = process_end_time;
            processed.read_timestamp = item.timestamp;
            processed.source_pts_ms = item.source_pts_ms;
            processed.upstream_latency_ms = item.upstream_latency_ms;
            
            {
                auto t_lock_start = std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lock(processing_mutex);
                auto t_lock_end = std::chrono::steady_clock::now();
                g_wait_lock_processing_mutex_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_lock_end - t_lock_start).count();
                g_wait_lock_processing_mutex_cnt++;
                auto t_wait_start = std::chrono::steady_clock::now();
                // Deterministic backpressure: simple blocking wait when queue is full
                while (processing_queue.size() >= PROCESSING_BUFFER_SIZE && !should_exit) {
                    processing_not_full.wait(lock, [&] {
                        return processing_queue.size() < PROCESSING_BUFFER_SIZE || should_exit;
                    });
                }
                auto t_wait_end = std::chrono::steady_clock::now();
                g_wait_processing_not_full_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
                g_wait_processing_not_full_cnt++;
                
                if (should_exit) break;
                
                processing_queue.push(std::move(processed));
                processing_not_empty.notify_one();
            }
            
            // Update counters
            actual_processed_frames++;
            
            // Update short-window processing FPS
            processed_frames_count++;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_process_time.load()).count();
            
            if (processed_frames_count % 10 == 0 && elapsed > 0) {
                real_processing_fps = processed_frames_count * 1000.0 / elapsed;
                last_process_time = now;
                processed_frames_count = 0;
            }
        }
    }
    
    processing_finished = true;
    processing_not_empty.notify_all();
    
    LOG_INFO("Frame processor thread exited");
}


void MultiThreadTrackingApp::handleModeSwitch(TrackingMode new_mode) {
    std::lock_guard<std::mutex> lock(mode_switch_mutex_);
    LOG_INFO("Handle mode switch to: " + std::string(new_mode == TrackingMode::GLOBAL ? "GLOBAL" : "LOCAL"));
    
    if (new_mode == current_mode_) {
        LOG_INFO("Already in requested mode, skip switching");
        return;
    }
    
    // Save current tracking state
    saveTrackingState();
    
    // Update mode
    previous_mode_ = current_mode_;
    current_mode_ = new_mode;
    
    // Notify system to switch mode
    bool use_global = (new_mode == TrackingMode::GLOBAL);
    system_->setUseGlobalDetection(use_global);
    
    // Restore previous saved tracking state
    restoreTrackingState();
    
    LOG_INFO("Mode switch completed");
}


void MultiThreadTrackingApp::saveTrackingState() {
    saved_tracks_.clear();

    if (current_mode_ == TrackingMode::GLOBAL) {
        system_->saveGlobalTrackerState(saved_tracks_);
    } else {
        system_->saveLocalTrackerState(saved_tracks_);
    }
    
    LOG_INFO("Saved tracking states: " + std::to_string(saved_tracks_.size()));
}


void MultiThreadTrackingApp::restoreTrackingState() {
    // Restore saved states to the corresponding tracker
    if (current_mode_ == TrackingMode::GLOBAL) {
        system_->restoreGlobalTrackerState(saved_tracks_);
    } else {
        system_->restoreLocalTrackerState(saved_tracks_);
    }
    
    LOG_INFO("Restored tracking states: " + std::to_string(saved_tracks_.size()));
}

void MultiThreadTrackingApp::resultDisplayThread() {
    LOG_INFO("Result display thread started");
    
    int display_count = 0;
    auto display_start_time = std::chrono::steady_clock::now();
    
    while (!should_exit) {
        if (is_paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        ProcessedItem item;
        bool has_result = false;
        
        {
            // Wait for result to display
            auto t_lock_start = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(processing_mutex);
            auto t_lock_end = std::chrono::steady_clock::now();
            g_wait_lock_processing_mutex_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_lock_end - t_lock_start).count();
            g_wait_lock_processing_mutex_cnt++;
            auto t_wait_start = std::chrono::steady_clock::now();
            // Deterministic waiting for display thread
            if (processing_queue.empty() && !(processing_finished && video_finished) && !should_exit) {
                processing_not_empty.wait(lock, [&] {
                    return !processing_queue.empty() || (processing_finished && video_finished) || should_exit;
                });
            }
            auto t_wait_end = std::chrono::steady_clock::now();
            g_wait_processing_not_empty_ms += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
            g_wait_processing_not_empty_cnt++;
            
            if (should_exit) break;
            
            if (!processing_queue.empty()) {
                item = std::move(processing_queue.front());
                processing_queue.pop();
                has_result = true;
                
                if (item.frame_number % 50 == 0) {
                    LOG_INFO("Display Thread: Process Frame " << item.frame_number 
                            << ", Queue Remaining: " << processing_queue.size() 
                            << "/" << PROCESSING_BUFFER_SIZE);
                }
                
                // Notify processing thread queue has space
                lock.unlock();
                processing_not_full.notify_one();
            } else if (processing_finished && video_finished) {
                // Processing and video finished and queue empty
                LOG_INFO("Display finished, no more results");
                break;
            } else {
                // Queue empty but processing not finished, short spin wait
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }
        }
        
                    if (has_result) {
            // Frame order validation (for debugging)
            int expected_display = last_displayed_frame_number.load() + 1;
            if (item.frame_number != expected_display && expected_display > 0) {
                LOG_WARNING("Display order issue: expected " << expected_display 
                           << ", got " << item.frame_number 
                           << " (diff: " << (item.frame_number - expected_display) << ")");
            }
            last_displayed_frame_number = item.frame_number;
            

            // End-to-end latency (read to display)
            auto now = std::chrono::steady_clock::now();
            g_overall_last_ts = now;  // Mark overall end time: last display/output
            auto delay = std::chrono::duration_cast<std::chrono::milliseconds>(now - item.read_timestamp).count();
            total_delay_ms += delay;
            
            // Detailed delay analysis (source PTS and upstream buffer)
            auto read_to_process = std::chrono::duration_cast<std::chrono::milliseconds>(
                item.process_start_time - item.read_timestamp).count();
            auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                item.process_end_time - item.process_start_time).count();
            auto process_to_display = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - item.process_end_time).count();

            long long end_to_end_from_pts = -1;
            if (item.source_pts_ms >= 0) {
                static const auto steady_zero = std::chrono::steady_clock::now();
                long long display_ms_since_zero = std::chrono::duration_cast<std::chrono::milliseconds>(now - steady_zero).count();

                end_to_end_from_pts = display_ms_since_zero - item.source_pts_ms;
            }
            
            if (item.frame_number % 50 == 0) {
                LOG_INFO("Delay analysis frame " << item.frame_number << ": total=" << delay 
                         << "ms (read->process=" << read_to_process 
                         << "ms, process=" << process_time 
                         << "ms, process->display=" << process_to_display << "ms)" 
                         << (item.upstream_latency_ms >= 0 ? (std::string(", upstream buffer est=") + std::to_string(item.upstream_latency_ms) + "ms") : std::string())
                         << (end_to_end_from_pts >= 0 ? (std::string(", PTS e2e est=") + std::to_string(end_to_end_from_pts) + "ms") : std::string())
                );
                
                if (delay > 500) {
                    LOG_WARNING("High latency warning: frame " << item.frame_number << " delay " << delay << "ms");
                }
            }
            
            // Display time since processing finished
            auto frame_display_start_time = std::chrono::steady_clock::now();
            
            // Update short-window display FPS
            displayed_frames_count++;
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_display = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_display_time.load()).count();
            
            if (displayed_frames_count % 10 == 0 && elapsed_display > 0) {
                real_display_fps = displayed_frames_count * 1000.0 / elapsed_display;
                last_display_time = current_time;
                displayed_frames_count = 0;
            }
            
            // Maintain video_fps for compatibility
            display_count++;
            auto elapsed_display_old = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - display_start_time).count();
            
            if (display_count % 10 == 0 && elapsed_display_old > 0) {
                video_fps = display_count * 1000.0 / elapsed_display_old;
                display_start_time = current_time;
                display_count = 0;
            }
            
            // Add overlay info to original frame (for video saving)
            auto t_overlay_start = std::chrono::steady_clock::now();
            std::stringstream ss;
            
            // Debug output: print top-K detection confidences if verbose logging enabled
            if (config_ && config_->verbose_logging) {
                if (!item.detections.empty()) {
                    std::vector<float> confs;
                    confs.reserve(item.detections.size());
                    for (const auto& d : item.detections) confs.push_back(d.confidence);
                    std::sort(confs.begin(), confs.end(), std::greater<float>());
                    size_t topk = std::min<size_t>(5, confs.size());
                    std::ostringstream topk_ss;
                    topk_ss << "[";
                    for (size_t i = 0; i < topk; ++i) {
                        if (i) topk_ss << ", ";
                        topk_ss << std::fixed << std::setprecision(3) << confs[i];
                    }
                    topk_ss << "]";
                    LOG_INFO("[ConfStats][Display] frame=" << item.frame_number
                             << ", dets=" << item.detections.size()
                             << ", top" << topk << "=" << topk_ss.str());
                } else {
                    LOG_INFO("[ConfStats][Display] frame=" << item.frame_number << ": no detections");
                }
            }
            
            // Calculate overall average FPS (pipeline throughput)
            // Use global metrics instead of windowed display_count to avoid FPS=0 every 10 frames
            double overall_avg_fps = 0.0;
            if (g_overall_first_seen.load() && actual_processed_frames > 1) {
                auto t0 = g_overall_first_ts.load();
                auto t1 = std::chrono::steady_clock::now();
                auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                if (dur_ms > 0) {
                    overall_avg_fps = (actual_processed_frames.load() * 1000.0) / dur_ms;
                }
            }
            
            ss << "Average FPS: " << std::fixed << std::setprecision(1) << overall_avg_fps;
            ss << " | Detections: " << item.detections.size();
            ss << " | Tracks: " << item.tracks.size();
            
            // Add overlay to original frame (for saving with full resolution)
            cv::putText(item.vis_frame, ss.str(), cv::Point(10, 50),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
            auto t_overlay_end = std::chrono::steady_clock::now();
            g_disp_overlay_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_overlay_end - t_overlay_start).count();
            
            // Create downscaled version for display (if enabled)
            cv::Mat vis_for_display = item.vis_frame;
            if (video_reader_ && video_reader_->isDownscalingEnabled()) {
                auto target = video_reader_->getTargetResolution();
                if (target.first > 0 && target.second > 0) {
                    cv::resize(item.vis_frame, vis_for_display, cv::Size(target.first, target.second), 0, 0, cv::INTER_AREA);
                }
            }
            
            // Publish ROS2 messages (use downscaled version for bandwidth efficiency)
            if (enable_ros_publishing_ && ros2_node_) {
                try {
                    double processing_fps_for_ros = real_processing_fps.load();
                    
                    ros2_node_->publishTrackingResult(
                        item.frame,  
                        item.detections, 
                        item.tracks, 
                        item.frame_number, 
                        processing_fps_for_ros
                    );
                    
                    ros2_node_->publishVisualizationImage(
                        vis_for_display, 
                        item.frame_number
                    );
                } catch (const std::exception& e) {
                    LOG_ERROR("ROS2 publish failed: " << e.what());
                }
            }
            
            if (FLAGS_enable_display) {
                auto t_imshow_start = std::chrono::steady_clock::now();
                cv::imshow("Tracking", vis_for_display);
                auto t_imshow_end = std::chrono::steady_clock::now();
                g_disp_imshow_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_imshow_end - t_imshow_start).count();

                auto display_end_time = std::chrono::steady_clock::now();
                auto display_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    display_end_time - frame_display_start_time).count();
                total_display_time_ms += display_time;

                auto t_waitkey_start = std::chrono::steady_clock::now();
                int key = cv::waitKey(1);
                auto t_waitkey_end = std::chrono::steady_clock::now();
                g_disp_waitkey_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_waitkey_end - t_waitkey_start).count();
                g_disp_frames++;
                if (key == 27) {
                    should_exit = true;
                } else if (key == 32) {
                    is_paused = !is_paused;
                    if (is_paused) {
                        LOG_INFO("Paused");
                    } else {
                        LOG_INFO("Resumed");
                    }
                }
                
                // Async video saving: push original resolution to queue (not downscaled)
                if (FLAGS_enable_video_output) {
                    auto t_write_start = std::chrono::steady_clock::now();
                    
                    std::unique_lock<std::mutex> save_lock(video_save_mutex);
                    if (video_save_queue.size() >= VIDEO_SAVE_BUFFER_SIZE) {
                        auto timeout = std::chrono::milliseconds(5);
                        video_save_not_full.wait_for(save_lock, timeout, [&] {
                            return video_save_queue.size() < VIDEO_SAVE_BUFFER_SIZE || should_exit;
                        });
                    }
                    
                    if (video_save_queue.size() < VIDEO_SAVE_BUFFER_SIZE) {
                        VideoSaveItem save_item;
                        save_item.vis_frame = item.vis_frame.clone();
                        save_item.frame_number = item.frame_number;
                        save_item.timestamp = std::chrono::steady_clock::now();
                        video_save_queue.push(std::move(save_item));
                        save_lock.unlock();
                        video_save_not_empty.notify_one();
                    } else {
                        LOG_WARNING("Video save queue full, skipping frame " << item.frame_number);
                    }
                    
                    auto t_write_end = std::chrono::steady_clock::now();
                    g_disp_write_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_write_end - t_write_start).count();
                }
            } else {
                // When display disabled, also use async video saving (original resolution)
                if (FLAGS_enable_video_output) {
                    std::unique_lock<std::mutex> save_lock(video_save_mutex);
                    if (video_save_queue.size() >= VIDEO_SAVE_BUFFER_SIZE) {
                        auto timeout = std::chrono::milliseconds(5);
                        video_save_not_full.wait_for(save_lock, timeout, [&] {
                            return video_save_queue.size() < VIDEO_SAVE_BUFFER_SIZE || should_exit;
                        });
                    }
                    
                    if (video_save_queue.size() < VIDEO_SAVE_BUFFER_SIZE) {
                        VideoSaveItem save_item;
                        save_item.vis_frame = item.vis_frame.clone();
                        save_item.frame_number = item.frame_number;
                        save_item.timestamp = std::chrono::steady_clock::now();
                        video_save_queue.push(std::move(save_item));
                        save_lock.unlock();
                        video_save_not_empty.notify_one();
                    }
                }
                auto display_end_time = std::chrono::steady_clock::now();
                auto display_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    display_end_time - frame_display_start_time).count();
                total_display_time_ms += display_time;
            }
        }
    }
    
    display_finished = true;
    video_save_not_empty.notify_all();
    
    LOG_INFO("Result display thread exited");
    if (FLAGS_enable_display) {
        cv::destroyAllWindows();
    }
}

// Async video save thread: writes frames to disk without blocking display
void MultiThreadTrackingApp::asyncVideoSaveThread() {
    LOG_INFO("Async video save thread started");
    
    int saved_frame_count = 0;
    auto save_start_time = std::chrono::steady_clock::now();
    
    while (!should_exit) {
        VideoSaveItem item;
        bool has_frame = false;
        
        {
            std::unique_lock<std::mutex> lock(video_save_mutex);
            
            // Wait for frames to save or finish signal
            if (video_save_queue.empty() && !display_finished && !should_exit) {
                video_save_not_empty.wait(lock, [&] {
                    return !video_save_queue.empty() || display_finished || should_exit;
                });
            }
            
            if (should_exit) break;
            
            if (!video_save_queue.empty()) {
                item = std::move(video_save_queue.front());
                video_save_queue.pop();
                has_frame = true;
                
                // Notify display thread there's space
                lock.unlock();
                video_save_not_full.notify_one();
            } else if (display_finished) {
                // No more frames and display finished
                break;
            } else {
                // Queue empty but display not finished, continue waiting
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }
        
        // Write frame (outside lock to avoid blocking)
        if (has_frame && video_writer_.isOpened()) {
            auto write_start = std::chrono::steady_clock::now();
            video_writer_.write(item.vis_frame);
            auto write_end = std::chrono::steady_clock::now();
            auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start).count();
            
            saved_frame_count++;
            
            if (saved_frame_count % 100 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - save_start_time).count();
                double save_fps = (saved_frame_count * 1000.0) / elapsed;
                LOG_INFO("Video save: " << saved_frame_count << " frames saved, "
                         << "FPS: " << std::fixed << std::setprecision(1) << save_fps
                         << ", last write: " << write_time << "ms");
            }
        }
    }
    
    LOG_INFO("Async video save thread exited, total saved: " << saved_frame_count << " frames");
}

} // namespace tracking 