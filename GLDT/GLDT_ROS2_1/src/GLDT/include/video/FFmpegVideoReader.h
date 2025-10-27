#ifndef FFMPEG_VIDEO_READER_H
#define FFMPEG_VIDEO_READER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <atomic>
#include <chrono>
#include <string>
#include "video/VideoReader.h"

// FFmpeg headers
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

namespace video {

// FFmpeg-based video reader
class FFmpegVideoReader : public VideoReader {
public:
    FFmpegVideoReader();
    ~FFmpegVideoReader() override;

    bool open(const std::string& filename) override;
    bool readNextFrame(cv::Mat& img, int& frame_number) override;
    int getWidth() const override;
    int getHeight() const override;
    int getTotalFrames() const override;
    double getFPS() const override;
    void close() override;

    // Timing support
    bool supportsTimestamps() const override { return true; }
    long long getLastFramePtsMs() const override { return last_pts_ms; }
    std::chrono::steady_clock::time_point getLastArrivalSteadyTime() const override { return last_arrival_steady; }
    long long getLastUpstreamLatencyMs() const override { return last_upstream_latency_ms; }

private:
    AVFormatContext* fmt_ctx;
    AVCodecContext* codec_ctx;
    AVPacket* pkt;
    AVFrame* frame;
    AVFrame* rgb_frame;
    SwsContext* sws_ctx;
    int video_stream_index;
    int width, height, frame_count, total_frames;
    double fps;
    std::vector<uint8_t> buffer;

    // Timing state
    AVRational stream_time_base{};
    std::atomic<long long> last_pts_ms{-1};
    std::atomic<long long> last_upstream_latency_ms{-1};
    std::atomic<long long> start_time_realtime_us{0};
    std::chrono::steady_clock::time_point last_arrival_steady;
    // PTS to steady_clock offset (ms), used to estimate when start_time_realtime is not available
    std::atomic<long long> pts_to_steady_offset_ms{LLONG_MIN};
};

} // namespace video

#endif // FFMPEG_VIDEO_READER_H 