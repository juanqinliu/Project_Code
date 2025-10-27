#ifndef SINGLE_THREAD_TRACKING_APP_H
#define SINGLE_THREAD_TRACKING_APP_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include "TrackingDetectionSystem.h"
#include "video/VideoReader.h"
#include "common/Config.h"
#include "common/Detection.h"
#include "tracking/STrack.h"
#include "common/Logger.h"
#include "ros2/GLDTRos2Node.h"

namespace tracking {

class SingleThreadTrackingApp {
public:
    SingleThreadTrackingApp(const std::string& global_model_path,
                            const std::string& local_model_path,
                            const std::string& video_path,
                            const std::string& output_dir);

    ~SingleThreadTrackingApp();

    // Run the single-threaded pipeline (read->process->display/save)
    void run();

    bool isVideoReaderInitialized() const { return video_reader_ != nullptr; }

private:
    bool initialize();
    void initializeOutputVideo();
    void cleanup();

private:
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

    int total_frames_ = 0;
    double fps_ = 0.0;
    int frame_width_ = 0;
    int frame_height_ = 0;

    // ROS2
    std::shared_ptr<GLDTRos2Node> ros2_node_;
    bool enable_ros_publishing_ = false;
};

} // namespace tracking

#endif // SINGLE_THREAD_TRACKING_APP_H


