#pragma once

#include <iostream>
#include <thread>         
#include <chrono> 


#include <fmt/core.h>
// #include <glog/logging.h>

#include <queue>
#include <mutex>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
// #include <libavutil/error.h>
#include <libswresample/swresample.h>
}

#include <opencv2/core/core.hpp>

#include "ffmpeg/bs_common.h"


class BsVideoSaver
{
public:
    BsVideoSaver();
    ~BsVideoSaver();

    // Initialize video push stream, only called once
    bool setup(std::string name, int width, int height, int fps, std::string encoder, int bitrate);
    // Push stream one frame image, called in loop
    void write(cv::Mat& image);


    // Connect to streaming server
    bool init(std::string name, int width, int height, int fps, std::string encoder, int bitrate);
    void start();
    void stop();

    // Encode video frame and push stream
    static void encodeVideoAndSaveThread(void* arg); 

    bool videoFrameQisEmpty();

    int writePkt(AVPacket *pkt);


    // Context
    AVFormatContext *mFmtCtx = nullptr;
    // Video frame
    AVCodecContext *mVideoCodecCtx = NULL;
    AVStream *mVideoStream = NULL;


    int mVideoIndex = -1;


private:

    // Get RGB frame from mRGB_VideoFrameQ
    bool getVideoFrame(VideoFrame *&frame, int &frameQSize); 


    // bgr24 to yuv420p
    unsigned char clipValue(unsigned char x, unsigned char min_val, unsigned char max_val);
    bool bgr24ToYuv420p(unsigned char *bgrBuf, int w, int h, unsigned char *yuvBuf);

    int width = -1;
    int height = -1;


    bool push_running = false;
    bool nd_push_frame = false;

    // Video frame
    std::queue<VideoFrame *> mRGB_VideoFrameQ;
    std::mutex mRGB_VideoFrameQ_mtx;


    // Push stream lock
    std::mutex mWritePkt_mtx;
    std::thread* mThread = nullptr;


};