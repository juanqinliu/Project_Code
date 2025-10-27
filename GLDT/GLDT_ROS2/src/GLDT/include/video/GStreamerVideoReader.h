#ifndef GSTREAMER_VIDEO_READER_H
#define GSTREAMER_VIDEO_READER_H

#include <opencv2/opencv.hpp>
#include <string>
#include "video/VideoReader.h"

namespace video {

// GStreamer-based video reader
class GStreamerVideoReader : public VideoReader {
public:
    GStreamerVideoReader();
    ~GStreamerVideoReader() override;

    bool open(const std::string& filename) override;
    bool readNextFrame(cv::Mat& img, int& frame_number) override;
    int getWidth() const override;
    int getHeight() const override;
    int getTotalFrames() const override;
    double getFPS() const override;
    void close() override;

private:
    cv::VideoCapture cap;
    int width, height, total_frames, frame_count;
    double fps;
    std::string pipeline;
};

} // namespace video

#endif // GSTREAMER_VIDEO_READER_H 