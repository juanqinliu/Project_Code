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
        // 多候选管线：优先 dma-buf，再回退到 mmap；引入 queue 并增大 appsink 缓冲，严格 drop=false（不丢帧）
        std::vector<std::string> candidates;
        candidates.push_back(
            std::string("v4l2src device=") + dev +
            " io-mode=4 ! queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 ! decodebin ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        candidates.push_back(
            std::string("v4l2src device=") + dev +
            " io-mode=2 ! queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 ! decodebin ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");

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
            LOG_ERROR("Cannot use GStreamer open v4l2 source: " << dev);
            return false;
        }
    } else if (filename.rfind("rtsp://", 0) == 0 || filename.rfind("rtsps://", 0) == 0) {
        // RTSP：优先硬解（NVDEC/VAAPI），TCP 传输，较低 latency；appsink 不丢帧
        std::vector<std::string> candidates;
        // NVDEC
        candidates.push_back(
            std::string("rtspsrc location=\"") + filename +
            "\" protocols=tcp latency=50 drop-on-latency=false do-timestamp=true ! rtph264depay ! h264parse ! nvh264dec ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // VAAPI
        candidates.push_back(
            std::string("rtspsrc location=\"") + filename +
            "\" protocols=tcp latency=50 drop-on-latency=false do-timestamp=true ! rtph264depay ! h264parse ! vaapih264dec ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // CPU 解码
        candidates.push_back(
            std::string("rtspsrc location=\"") + filename +
            "\" protocols=tcp latency=50 drop-on-latency=false do-timestamp=true ! rtph264depay ! h264parse ! avdec_h264 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");

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
            LOG_ERROR("Cannot use GStreamer open RTSP: " << filename);
            return false;
        }
    } else {
        // Local file (mp4/mov etc.): try multiple pipelines in order to improve compatibility
        std::vector<std::string> candidates;
        // 1) uridecodebin（自动解码）
        candidates.push_back(
            std::string("uridecodebin uri=\"file://") + filename +
            "\" ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // 2) filesrc+decodebin（显式 filesrc）
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! decodebin ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // 3) filesrc+qtdemux -> mpeg4videoparse -> avdec_mpeg4 (mp4v/MPEG-4 SP/ASP)
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! mpeg4videoparse ! avdec_mpeg4 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // 4) filesrc+qtdemux -> decodebin (explicit demux)
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! decodebin ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // 5) H.264 explicit decode chain（优先 NVDEC/VAAPI，再回退 CPU）
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! nvh264dec ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! vaapih264dec ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! avdec_h264 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        // 6) H.265/HEVC explicit decode chain（优先 NVDEC/VAAPI，再回退 CPU）
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h265parse ! nvh265dec ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h265parse ! vaapih265dec ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");
        candidates.push_back(
            std::string("filesrc location=\"") + filename +
            "\" ! qtdemux name=demux demux.video_0 ! queue ! h265parse ! avdec_h265 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=16 drop=false enable-last-sample=false");

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