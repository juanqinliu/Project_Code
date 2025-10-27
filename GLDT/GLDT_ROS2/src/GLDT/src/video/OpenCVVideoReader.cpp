#include "video/OpenCVVideoReader.h"
#include "video/VideoReaderFactory.h"
#include "common/Logger.h"

namespace video {

OpenCVVideoReader::OpenCVVideoReader()
    : width(0), height(0), total_frames(0), frame_count(0), fps(0.0) {
    LOG_INFO("Create OpenCV video reader");
}

OpenCVVideoReader::~OpenCVVideoReader() {
    close();
}

bool OpenCVVideoReader::open(const std::string& filename) {
    LOG_INFO("OpenCV open video: " << filename);
    cap.open(filename);
    
    if (!cap.isOpened()) {
        LOG_ERROR("Failed to open video with OpenCV: " << filename);
        return false;
    }
    
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    fps = cap.get(cv::CAP_PROP_FPS);
    frame_count = 0;
    
    LOG_INFO("OpenCV video information: Width=" << width 
             << ", Height=" << height 
             << ", Total frames=" << total_frames 
             << ", FPS=" << fps);
    
    // Initialize the downsampling parameters for display (only used for display, not for inference)
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

bool OpenCVVideoReader::readNextFrame(cv::Mat& img, int& frame_number) {
    if (!cap.isOpened()) {
        return false;
    }
    
    bool ret = cap.read(img);
    if (ret) {
        // Only return the original resolution frame, downsampling is done in the display stage
        frame_number = frame_count++;
    }
    return ret;
}

int OpenCVVideoReader::getWidth() const {
    return width;
}

int OpenCVVideoReader::getHeight() const {
    return height;
}

int OpenCVVideoReader::getTotalFrames() const {
    return total_frames;
}

double OpenCVVideoReader::getFPS() const {
    return fps;
}

void OpenCVVideoReader::close() {
    if (cap.isOpened()) {
        cap.release();
        LOG_INFO("Close OpenCV video reader");
    }
}

} // namespace video 

// Use macro to automatically register OpenCV video reader
REGISTER_VIDEO_READER(OPENCV, video::OpenCVVideoReader) 