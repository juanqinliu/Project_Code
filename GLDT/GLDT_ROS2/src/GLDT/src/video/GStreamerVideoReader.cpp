#include "video/GStreamerVideoReader.h"
#include "video/VideoReaderFactory.h"
#include "common/Logger.h"

namespace video {

GStreamerVideoReader::GStreamerVideoReader()
    : width(0), height(0), total_frames(0), frame_count(0), fps(0.0) {
    LOG_INFO("Create GStreamer video reader");
}

GStreamerVideoReader::~GStreamerVideoReader() {
    close();
}

bool GStreamerVideoReader::open(const std::string& filename) {
    LOG_INFO("GStreamer open video: " << filename);
    
    // Build GStreamer pipeline: distinguish v4l2 device / number index / file&network
    auto isDigits = [](const std::string& s){ return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit); };
    bool is_v4l2 = (filename.rfind("/dev/video", 0) == 0) || isDigits(filename);

    if (is_v4l2) {
        std::string dev = filename;
        if (isDigits(filename)) dev = "/dev/video" + filename;
        // Low latency and no drop: appsink sync=false max-buffers=2 drop=false; ensure output BGR, avoid extra SWS
        pipeline =
            "v4l2src device=" + dev +
            " io-mode=2 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false";
    } else if (filename.rfind("rtsp://", 0) == 0 || filename.rfind("rtsps://", 0) == 0) {
        pipeline =
            "rtspsrc location=\"" + filename +
            "\" latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false";
    } else {
        // Local file (mp4/mov etc.): try multiple pipelines in order to improve compatibility
        std::vector<std::string> candidates;
        // 1) uridecodebin
        candidates.push_back(
            std::string("uridecodebin uri=\"file://") + filename +
            "\" ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false");
        // 2) filesrc+decodebin
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false");
        // 3) filesrc+qtdemux -> mpeg4videoparse -> avdec_mpeg4 (mp4v/MPEG-4 SP/ASP)
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! mpeg4videoparse ! avdec_mpeg4 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false");
        // 4) filesrc+qtdemux -> decodebin (explicit demux)
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false");
        // 5) H.264 explicit decode chain
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false");
        // 6) H.265/HEVC explicit decode chain
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=false");

        bool opened = false;
        for (const auto& p : candidates) {
            LOG_INFO("GStreamer pipeline: " << p);
            cap.open(p, cv::CAP_GSTREAMER);
            if (cap.isOpened()) {
                pipeline = p;
                opened = true;
                break;
            }
        }
        if (!opened) {
            LOG_ERROR("Cannot use GStreamer open video: " << filename);
            return false;
        }
        // Get subsequent information and initialize downscaling in the same path
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        fps = cap.get(cv::CAP_PROP_FPS);
        frame_count = 0;
        
        LOG_INFO("GStreamer video information: width=" << width 
                 << ", height=" << height 
                 << ", total frames=" << total_frames 
                 << ", FPS=" << fps);
        
        // Initialize downscaling parameters for display (only for display, not for inference)
        initializeDownscaling();
        if (isDownscalingEnabled()) {
            auto target_res = getTargetResolution();
            LOG_INFO("Display will use downsampling: " << width << "x" << height
                     << " -> " << target_res.first << "x" << target_res.second);
        } else {
            LOG_INFO("Display will not use downsampling, keep original resolution: " << width << "x" << height);
        }
        return true;
    }

    LOG_INFO("GStreamer pipeline: " << pipeline);
    cap.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        LOG_ERROR("Cannot use GStreamer open video: " << filename);
        return false;
    }
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    fps = cap.get(cv::CAP_PROP_FPS);
    frame_count = -1;
    LOG_INFO("GStreamer video information: width=" << width 
             << ", height=" << height 
             << ", total frames=" << total_frames 
             << ", FPS=" << fps);
    initializeDownscaling();
    if (isDownscalingEnabled()) {
        auto target_res = getTargetResolution();
        LOG_INFO("Display will use downsampling: " << width << "x" << height
                 << " -> " << target_res.first << "x" << target_res.second);
    } else {
        LOG_INFO("Display will not use downsampling, keep original resolution: " << width << "x" << height);
    }
    return true;
}

bool GStreamerVideoReader::readNextFrame(cv::Mat& img, int& frame_number) {
    if (!cap.isOpened()) {
        return false;
    }
    
    bool ret = cap.read(img);
    if (ret) {
        frame_number = frame_count++;
    }
    return ret;
}

int GStreamerVideoReader::getWidth() const {
    return width;
}

int GStreamerVideoReader::getHeight() const {
    return height;
}

int GStreamerVideoReader::getTotalFrames() const {
    return total_frames;
}

double GStreamerVideoReader::getFPS() const {
    return fps;
}

void GStreamerVideoReader::close() {
    if (cap.isOpened()) {
        cap.release();
        LOG_INFO("close GStreamer video reader");
    }
}

} // namespace video 

// Use macro to automatically register GStreamer video reader
REGISTER_VIDEO_READER(GSTREAMER, video::GStreamerVideoReader) 