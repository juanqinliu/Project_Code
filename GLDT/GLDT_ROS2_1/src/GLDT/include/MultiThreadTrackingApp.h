#ifndef MULTI_THREAD_TRACKING_APP_H
#define MULTI_THREAD_TRACKING_APP_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "TrackingDetectionSystem.h"
#include "tracking/STrack.h"
#include "common/Config.h"
#include "common/Detection.h"
#include "common/Logger.h"
#include "video/VideoReader.h"
#include "ros2/GLDTRos2Node.h"

namespace tracking {
    // Multithreaded data structures
    // 保持深拷贝确保确定性和线程安全
    struct FrameItem {
        cv::Mat frame;              // original frame (深拷贝保证线程安全)
        int frame_number;           // frame index
        std::chrono::steady_clock::time_point timestamp; // capture timestamp

        long long source_pts_ms = -1;

        long long upstream_latency_ms = -1;
    };

    struct ProcessedItem {
        cv::Mat frame;              // original frame (深拷贝)
        cv::Mat vis_frame;          // visualization frame (深拷贝)
        std::vector<tracking::Detection> detections; // detection results
        std::vector<std::unique_ptr<tracking::STrack>> tracks; // tracking results
        int frame_number;           // frame index
        std::chrono::steady_clock::time_point process_start_time; // processing start time
        std::chrono::steady_clock::time_point process_end_time;   // processing end time
        std::chrono::steady_clock::time_point read_timestamp;     // read timestamp
        long long source_pts_ms = -1;                              // frame PTS (ms)
        long long upstream_latency_ms = -1;                        // upstream latency (ms)
    };

    // Video saving item for async video writing
    struct VideoSaveItem {
        cv::Mat vis_frame;          // frame to save (深拷贝)
        int frame_number;           // frame index
        std::chrono::steady_clock::time_point timestamp; // save timestamp
    };

    // Tracking mode
    enum class TrackingMode {
        GLOBAL,  // global tracking mode
        LOCAL    // local tracking mode
    };

    // Multithreaded tracking application
    class MultiThreadTrackingApp {
    public:
        MultiThreadTrackingApp(const std::string& global_model_path, 
                            const std::string& local_model_path,
                            const std::string& video_path,
                            const std::string& output_dir);
        
        ~MultiThreadTrackingApp();
        
        // Run the multithreaded pipeline
        void run();
        
        // Check whether the video reader is initialized
        bool isVideoReaderInitialized() const { return video_reader_ != nullptr; }

    private:
        // Thread 1: video reading
        void videoReaderThread();

        // Thread 2: frame processing
        void frameProcessorThread();

        // Thread 3: result display
        void resultDisplayThread();
        
        // Thread 4: async video saving
        void asyncVideoSaveThread();
        
        // Initialize video metadata/resources
        bool initializeVideo();
        
        // Initialize output video writer
        void initializeOutputVideo();
        
        // Cleanup resources
        void cleanup();

        // Initialize the video reader
        bool initializeVideoReader();
        
        // Handle state transition when switching modes
        void handleModeSwitch(TrackingMode new_mode);
        
        // Save current tracking state
        void saveTrackingState();
        
        // Restore tracking state
        void restoreTrackingState();

        // Members
        std::string global_model_path_;
        std::string local_model_path_;
        std::string video_path_;
        std::string output_dir_;
        std::string video_basename_;
        std::string output_video_path_;
        
        std::unique_ptr<tracking::Config> config_;
        std::unique_ptr<tracking::TrackingDetectionSystem> system_;
        
        std::shared_ptr<video::VideoReader> video_reader_;
        cv::VideoWriter video_writer_;
        
        int total_frames_;
        double fps_;
        int frame_width_;
        int frame_height_;
        
        // Tracking state
        TrackingMode current_mode_ = TrackingMode::GLOBAL;
        TrackingMode previous_mode_ = TrackingMode::GLOBAL;
        std::vector<std::unique_ptr<STrack>> saved_tracks_;  // saved tracking state
        std::mutex mode_switch_mutex_;  // mode switch mutex
        
        // ROS2 node
        std::shared_ptr<GLDTRos2Node> ros2_node_;
        bool enable_ros_publishing_;
    };

    // Buffer size configuration
    extern const int BUFFER_SIZE;

    // Inter-thread queues
    extern std::queue<FrameItem> reading_queue;           // reader -> processor
    extern std::queue<ProcessedItem> processing_queue;    // processor -> displayer
    extern std::queue<VideoSaveItem> video_save_queue;    // displayer -> video saver

    // Mutexes
    extern std::mutex reading_mutex;
    extern std::mutex processing_mutex;
    extern std::mutex video_save_mutex;

    // Condition variables
    extern std::condition_variable reading_not_full;
    extern std::condition_variable reading_not_empty;
    extern std::condition_variable processing_not_full;
    extern std::condition_variable processing_not_empty;
    extern std::condition_variable video_save_not_full;
    extern std::condition_variable video_save_not_empty;

    // Thread control flags
    extern std::atomic<bool> video_finished;       // video reading done
    extern std::atomic<bool> processing_finished;  // processing done
    extern std::atomic<bool> video_save_finished;  // video saving done
    extern std::atomic<bool> should_exit;          // global exit flag
    extern std::atomic<bool> is_paused;            // pause flag
    
    // Mode switching flags
    extern std::atomic<bool> mode_switching;       // switching in progress
    extern std::atomic<TrackingMode> requested_mode; // requested mode

    // Performance counters
    extern std::atomic<int> actual_processed_frames;   // processed frame count
    extern std::atomic<double> video_fps;              // input video fps
    extern std::atomic<long long> total_delay_ms;      // total delay (ms)
    extern std::atomic<long long> total_process_time_ms;  // total processing time (ms)
    extern std::atomic<long long> total_read_time_ms;     // total reading time (ms)
    extern std::atomic<long long> total_display_time_ms;  // total display time (ms)
}

#endif // MULTI_THREAD_TRACKING_APP_H 