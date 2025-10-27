#include "SingleThreadTrackingApp.h"
#include "video/VideoReaderFactory.h"
#include "common/Flags.h"
#include <filesystem>
#include <atomic>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <ctime>

namespace fs = std::filesystem;

namespace tracking {

// exit
extern std::atomic<bool> should_exit;

SingleThreadTrackingApp::SingleThreadTrackingApp(const std::string& global_model_path,
                                                 const std::string& local_model_path,
                                                 const std::string& video_path,
                                                 const std::string& output_dir)
    : global_model_path_(global_model_path),
      local_model_path_(local_model_path),
      video_path_(video_path),
      output_dir_(output_dir) {}

SingleThreadTrackingApp::~SingleThreadTrackingApp() {
    cleanup();
}

bool SingleThreadTrackingApp::initialize() {
    // Read configuration
    config_ = std::make_unique<Config>();
    config_->updateFromFlags();

    // Initialize video reader (select specific reader according to flags)
    video::VideoReaderType vr_type = video::VideoReaderType::OPENCV;
    switch (flags::getVideoReaderMode()) {
        case flags::VideoReaderMode::OPENCV: vr_type = video::VideoReaderType::OPENCV; break;
        case flags::VideoReaderMode::FFMPEG: vr_type = video::VideoReaderType::FFMPEG; break;
        case flags::VideoReaderMode::GSTREAMER: vr_type = video::VideoReaderType::GSTREAMER; break;
    }
    video_reader_ = video::VideoReaderFactory::createVideoReader(vr_type);
    if (!video_reader_ || !video_reader_->open(video_path_)) {
        LOG_ERROR("Failed to open video: " << video_path_);
        return false;
    }

    frame_width_ = video_reader_->getWidth();
    frame_height_ = video_reader_->getHeight();
    fps_ = video_reader_->getFPS();
    total_frames_ = video_reader_->getTotalFrames();

    // Initialize downsampling as needed
    video_reader_->initializeDownscaling();

    // Initialize system
    system_ = std::make_unique<TrackingDetectionSystem>(global_model_path_, local_model_path_, *config_);
    system_->setTotalFrames(total_frames_);

    // Output path
    if (!output_dir_.empty()) {
        fs::create_directories(output_dir_);
        // Unified naming: use timestamp for camera/stream or unrecognized paths
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

        fs::path p(video_path_);
        std::string candidate = p.stem().string();
        bool path_exists = fs::exists(p);
        bool no_ext = p.has_extension() == false;
        bool no_parent = p.parent_path().empty();
        if (candidate.empty() || isDigitString(video_path_) ||
            startsWith(video_path_, "rtsp://") || startsWith(video_path_, "rtsps://") ||
            startsWith(video_path_, "http://") || startsWith(video_path_, "https://") ||
            startsWith(video_path_, "v4l2://") || startsWith(video_path_, "camera://") ||
            startsWith(video_path_, "/dev/video") ||
            startsWith(video_path_, "video") || startsWith(video_path_, "cam") ||
            (!path_exists && no_ext && no_parent)) {
            video_basename_ = makeTimestampName();
        } else {
            video_basename_ = candidate;
        }

        output_video_path_ = (fs::path(output_dir_) / (video_basename_ + ".mp4")).string();
        initializeOutputVideo();
    }

    // ROS2
    // Single thread default does not publish ROS2 topics, avoid extra dependencies
    enable_ros_publishing_ = false;

    return true;
}

void SingleThreadTrackingApp::initializeOutputVideo() {
    if (output_video_path_.empty()) return;
    // Same as multi-thread: use mp4v encoding; resolution consistent with display (if downsampling is enabled)
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double out_fps = fps_ > 0 ? fps_ : 25.0;
    int out_w = frame_width_;
    int out_h = frame_height_;
    if (video_reader_ && video_reader_->isDownscalingEnabled()) {
        auto target = video_reader_->getTargetResolution();
        if (target.first > 0 && target.second > 0) {
            out_w = target.first;
            out_h = target.second;
        }
    }
    video_writer_.open(output_video_path_, fourcc, out_fps, cv::Size(out_w, out_h));
    if (!video_writer_.isOpened()) {
        LOG_WARNING("Failed to open output video: " << output_video_path_);
    }
}

void SingleThreadTrackingApp::cleanup() {
    if (video_writer_.isOpened()) video_writer_.release();
    if (video_reader_) video_reader_->close();
}

void SingleThreadTrackingApp::run() {
    if (!initialize()) return;

    cv::Mat frame;
    int frame_id = 0;
    // Same as multi-thread FPS statistics
    int display_count = 0;
    auto display_start_time = std::chrono::steady_clock::now();
    double video_fps = 0.0;
    // Statistics total time to output average FPS
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        if (tracking::should_exit) break;
        auto read_start = std::chrono::steady_clock::now();
        int frame_number = 0;
        if (!video_reader_->readNextFrame(frame, frame_number) || frame.empty()) {
            break;
        }
        auto read_end = std::chrono::steady_clock::now();

        auto proc_start = std::chrono::steady_clock::now();

        std::vector<Detection> detections;
        std::vector<std::unique_ptr<STrack>> tracks;
        cv::Mat vis;
        system_->process(frame, vis, detections, tracks);

        auto proc_end = std::chrono::steady_clock::now();

        // Output/display (align with multi-thread display information)
        if (video_writer_.isOpened()) {
            video_writer_.write(vis.empty() ? frame : vis);
        }

        if (FLAGS_enable_display) {
            // Same as multi-thread: display downsampling
            cv::Mat show = vis.empty() ? frame : vis;
            if (video_reader_ && video_reader_->isDownscalingEnabled()) {
                auto target = video_reader_->getTargetResolution();
                if (target.first > 0 && target.second > 0) {
                    cv::resize(show, show, cv::Size(target.first, target.second), 0, 0, cv::INTER_AREA);
                }
            }

            // Calculate display FPS (update every 10 frames), same as multi-thread
            display_count++;
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_display = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - display_start_time).count();
            if (display_count % 10 == 0 && elapsed_display > 0) {
                video_fps = display_count * 1000.0 / elapsed_display;
                display_start_time = current_time;
                display_count = 0;
            }

            // Overlay information: same as multi-thread
            std::stringstream ss;
            ss << "FPS: " << std::fixed << std::setprecision(1) << video_fps;
            ss << " | Detections: " << detections.size();
            ss << " | Tracks: " << tracks.size();
            cv::putText(show, ss.str(), cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

            // Window name same as multi-thread
            cv::imshow("Tracking", show);
            int key = cv::waitKey(1);
            if (key == 27) { // ESC
                break;
            }
        }

        // ROS2 publish
        // if (enable_ros_publishing_ && ros2_node_) {
        //     ros2_node_->publishTrackingResult(tracks);
        // }

        // Simple log
        double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
        double proc_ms = std::chrono::duration<double, std::milli>(proc_end - proc_start).count();
        if ((++frame_id) % 20 == 0) {
            LOG_INFO("[SingleThread] frame=" << frame_id << " read=" << read_ms << "ms proc=" << proc_ms << "ms");
        }
    }

    // End statistics and output average FPS (same as multi-thread style)
    auto end_time = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double total_time_s = total_time_ms / 1000.0;
    LOG_INFO("Single thread processing completed");
    LOG_INFO("Total processed frames: " << frame_id);
    if (total_time_s > 0 && frame_id > 0) {
        double avg_fps = frame_id / total_time_s;
        LOG_INFO("Average processing speed: " << avg_fps << " FPS");
    }

    // Save tracking results to file (same as multi-thread)
    if (system_) {
        system_->saveResults(output_dir_, video_basename_);
    }
    std::string results_file_path = (fs::path(output_dir_) / (video_basename_ + ".txt")).string();
    if (FLAGS_enable_video_output) {
        std::cout << "Video save path: " << output_video_path_ << std::endl;
    } else {
        std::cout << "Video output disabled (enable_video_output=false)" << std::endl;
    }
    std::cout << "Result save path: " << results_file_path << std::endl;

    cleanup();
}

} // namespace tracking


