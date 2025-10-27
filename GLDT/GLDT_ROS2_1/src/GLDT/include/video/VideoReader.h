#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "common/Flags.h"
namespace video {

// Base interface for video readers
class VideoReader {
public:
    virtual ~VideoReader() = default;
    
    // Open a video file
    virtual bool open(const std::string& filename) = 0;
    
    // Read next frame
    virtual bool readNextFrame(cv::Mat& img, int& frame_number) = 0;
    
    // Video width
    virtual int getWidth() const = 0;
    
    // Video height
    virtual int getHeight() const = 0;
    
    // Total number of frames
    virtual int getTotalFrames() const = 0;
    
    // Frames per second
    virtual double getFPS() const = 0;
    
    // Close video
    virtual void close() = 0;
    
    // Downscaling feature
    // Initialize downscaling parameters (call after open)
    virtual void initializeDownscaling() {
        if (!FLAGS_enable_video_downscaling) {
            enable_downscaling_ = false;
            return;
        }
        
        int original_width = getWidth();
        int original_height = getHeight();
        
        if (original_width <= 0 || original_height <= 0) {
            enable_downscaling_ = false;
            return;
        }
        
        // Heuristic downscaling: choose target resolution by source resolution
        if (original_width >= 1920 && original_height >= 1080) {
            // 1080p+ -> 720p
            target_width_ = 1280;
            target_height_ = 720;
        } else if (original_width >= 1280 && original_height >= 720) {
            // 720p -> 480p
            target_width_ = 854;
            target_height_ = 480;
        } else if (original_width >= 854 && original_height >= 480) {
            // 480p -> 360p
            target_width_ = 640;
            target_height_ = 360;
        } else {
            // keep original resolution
            enable_downscaling_ = false;
            return;
        }
        
        enable_downscaling_ = true;
    }
    
    // Manually set target resolution
    virtual void setTargetResolution(int width, int height) {
        target_width_ = width;
        target_height_ = height;
        enable_downscaling_ = FLAGS_enable_video_downscaling && (width > 0 && height > 0);
    }
    
    // Get target resolution
    virtual std::pair<int, int> getTargetResolution() const {
        return {target_width_, target_height_};
    }
    
    // Enable/disable downscaling
    virtual void enableDownscaling(bool enable) {
        enable_downscaling_ = enable && FLAGS_enable_video_downscaling && (target_width_ > 0 && target_height_ > 0);
    }
    
    // Whether downscaling is enabled
    virtual bool isDownscalingEnabled() const {
        return enable_downscaling_ && FLAGS_enable_video_downscaling;
    }
    
    // Get original resolution
    virtual std::pair<int, int> getOriginalResolution() const {
        return {getWidth(), getHeight()};
    }
    
    // Get current output resolution
    virtual std::pair<int, int> getCurrentResolution() const {
        if (isDownscalingEnabled()) {
            return {target_width_, target_height_};
        } else {
            return {getWidth(), getHeight()};
        }
    }

    // Optional timing support (live sources)
    // Whether this reader can provide source PTS and arrival timestamps
    virtual bool supportsTimestamps() const { return false; }

    // Last decoded frame's best-effort PTS in milliseconds (monotonic, starts near 0)
    virtual long long getLastFramePtsMs() const { return -1; }

    // Last frame's arrival time in steady_clock
    virtual std::chrono::steady_clock::time_point getLastArrivalSteadyTime() const {
        return std::chrono::steady_clock::now();
    }

    // Estimated upstream latency (driver/device/internal buffering) in ms; -1 if unknown
    virtual long long getLastUpstreamLatencyMs() const { return -1; }

protected:
    // Downscaling configuration
    int target_width_ = -1;
    int target_height_ = -1;
    bool enable_downscaling_ = false;
    
    // Downscale frame in-place
    virtual void downscaleFrame(cv::Mat& frame) {
        if (!isDownscalingEnabled()) {
            return;
        }
        
        // INTER_AREA is suitable for image shrinking
        cv::resize(frame, frame, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_AREA);
    }
    
    // Downscale while maintaining aspect ratio
    virtual void downscaleFrameMaintainAspectRatio(cv::Mat& frame) {
        if (!isDownscalingEnabled()) {
            return;
        }
        
        int original_width = frame.cols;
        int original_height = frame.rows;
        
        // Compute scale with aspect ratio preserved
        double scale_x = static_cast<double>(target_width_) / original_width;
        double scale_y = static_cast<double>(target_height_) / original_height;
        double scale = std::min(scale_x, scale_y);
        
        // Compute new size
        int new_width = static_cast<int>(original_width * scale);
        int new_height = static_cast<int>(original_height * scale);
        
        cv::resize(frame, frame, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
    }
};

} // namespace video

#endif // VIDEO_READER_H 