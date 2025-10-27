#include "video/FFmpegVideoReader.h"
#include "video/VideoReaderFactory.h"
#include "common/Logger.h"
#include <atomic>
#include <chrono>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
}

namespace video {

FFmpegVideoReader::FFmpegVideoReader()
    : fmt_ctx(nullptr), codec_ctx(nullptr), pkt(nullptr),
      frame(nullptr), rgb_frame(nullptr), sws_ctx(nullptr),
      video_stream_index(-1), width(0), height(0), frame_count(0), total_frames(0), fps(0.0) {
    LOG_INFO("Create FFmpeg Video Reader");
    
    // Initialize FFmpeg
    #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
    #endif
    // Register input output devices(e.g. v4l2)
    avdevice_register_all();
}

FFmpegVideoReader::~FFmpegVideoReader() {
    close();
}

// Read stage timing accumulation (process level accumulation)
namespace {
std::atomic<long long> g_ff_packet_time_ms{0};
std::atomic<long long> g_ff_decode_time_ms{0};
std::atomic<long long> g_ff_sws_time_ms{0};
std::atomic<long long> g_ff_clone_time_ms{0};
std::atomic<long long> g_ff_downscale_time_ms{0};
std::atomic<long long> g_ff_total_time_ms{0};
std::atomic<long long> g_ff_read_frames{0};
}

bool FFmpegVideoReader::open(const std::string& filename) {
    LOG_INFO("FFmpeg Open Video: " << filename);
    
    fmt_ctx = avformat_alloc_context();
    if (!fmt_ctx) {
        LOG_ERROR("Cannot Allocate FFmpeg Format Context");
        return false;
    }
    
    AVInputFormat* input_fmt = nullptr;
    AVDictionary* options = nullptr;

    bool is_camera_index = !filename.empty() && std::all_of(filename.begin(), filename.end(), ::isdigit);
    bool is_v4l2_device = filename.rfind("/dev/video", 0) == 0;

    if (is_camera_index || is_v4l2_device) {
        input_fmt = av_find_input_format("video4linux2");
        av_dict_set(&options, "framerate", "30", 0);
        av_dict_set(&options, "buffercount", "2", 0); 
        av_dict_set(&options, "buffersize", "0", 0);
        av_dict_set(&options, "use_wallclock_as_timestamps", "1", 0);
        av_dict_set(&options, "rtbufsize", "64M", 0);
    }

    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "probesize", "32768", 0);
    av_dict_set(&options, "analyzeduration", "0", 0);
    av_dict_set(&options, "thread_queue_size", "64", 0);
    av_dict_set(&options, "reorder_queue_size", "0", 0);

    if (avformat_open_input(&fmt_ctx, filename.c_str(), input_fmt, &options) != 0) {
        LOG_ERROR("Cannot Open Video File: " << filename);
        return false;
    }
    
    // Read stream information
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        LOG_ERROR("Cannot Read Stream Information");
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Find video stream
    video_stream_index = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    
    if (video_stream_index == -1) {
        LOG_ERROR("Cannot Find Video Stream");
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Get codec parameters
    AVCodecParameters* codec_params = fmt_ctx->streams[video_stream_index]->codecpar;
    
    // Get width and height
    width = codec_params->width;
    height = codec_params->height;
    
    // Get frame rate
    AVRational frame_rate = av_guess_frame_rate(fmt_ctx, fmt_ctx->streams[video_stream_index], nullptr);
    fps = frame_rate.num > 0 && frame_rate.den > 0 ? 
          static_cast<double>(frame_rate.num) / frame_rate.den : 0.0;
    stream_time_base = fmt_ctx->streams[video_stream_index]->time_base;
    
    // Get total frames
    total_frames = fmt_ctx->streams[video_stream_index]->nb_frames;
    if (total_frames == 0) {
        if (fps > 0 && fmt_ctx->duration > 0) {
            total_frames = static_cast<int>((fmt_ctx->duration * fps) / AV_TIME_BASE);
        } else {
            total_frames = 0;
        }
    }
    
    // Get decoder
    AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        LOG_ERROR("Cannot Find Decoder");
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Create codec context
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        LOG_ERROR("Cannot Allocate Codec Context");
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Copy codec parameters
    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
        LOG_ERROR("Cannot Copy Codec Parameters");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Enable multi-thread decoding
    codec_ctx->thread_count = 4;
    codec_ctx->thread_type = FF_THREAD_FRAME; 
    codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;
    
    codec_ctx->skip_frame = AVDISCARD_DEFAULT;
    codec_ctx->skip_idct = AVDISCARD_DEFAULT;
    codec_ctx->skip_loop_filter = AVDISCARD_DEFAULT;
    
    // Open codec
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        LOG_ERROR("Cannot Open Codec");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Allocate packet
    pkt = av_packet_alloc();
    if (!pkt) {
        LOG_ERROR("Cannot Allocate Packet");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    // Allocate frame
    frame = av_frame_alloc();
    rgb_frame = av_frame_alloc();
    if (!frame || !rgb_frame) {
        LOG_ERROR("Cannot Allocate Frame");
        if (frame) av_frame_free(&frame);
        if (rgb_frame) av_frame_free(&rgb_frame);
        av_packet_free(&pkt);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    buffer.resize(av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1));
    av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer.data(), 
                        AV_PIX_FMT_BGR24, width, height, 1);
    
    sws_ctx = sws_getContext(
        width, height, codec_ctx->pix_fmt,
        width, height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
    );
    
    if (!sws_ctx) {
        LOG_ERROR("Cannot Create Conversion Context");
        av_frame_free(&frame);
        av_frame_free(&rgb_frame);
        av_packet_free(&pkt);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }
    
    frame_count = 0;
    // Initialize start realtime time (for estimating capture buffer/driver delay)
    if (start_time_realtime_us.load() == 0 && fmt_ctx->start_time_realtime != AV_NOPTS_VALUE) {
        start_time_realtime_us = fmt_ctx->start_time_realtime; 
    }
    
    LOG_INFO("FFmpeg Video Information: Width=" << width 
             << ", Height=" << height 
             << ", Total Frames=" << total_frames 
             << ", FPS=" << fps);
    
    // Initialize display downscaling parameters (only for display,不影响推理）
    initializeDownscaling();
    if (isDownscalingEnabled()) {
        auto target_res = getTargetResolution();
        LOG_INFO("Display will use downscaling: " << width << "x" << height
                 << " -> " << target_res.first << "x" << target_res.second);
    } else {
        LOG_INFO("Display will not downscale, keep original resolution: " << width << "x" << height);
    }
    
    return true;
}

bool FFmpegVideoReader::readNextFrame(cv::Mat& img, int& frame_number) {
    if (!fmt_ctx || !codec_ctx) {
        return false;
    }
    
    int ret = 0;
    

    long long packet_ms_sum = 0;
    long long decode_ms_sum = 0;
    long long sws_ms_sum = 0;
    long long clone_ms_sum = 0;
    long long downscale_ms_sum = 0;
    auto total_start = std::chrono::steady_clock::now();

    while (true) {
        auto t_pkt_start = std::chrono::steady_clock::now();
        ret = av_read_frame(fmt_ctx, pkt);
        auto t_pkt_end = std::chrono::steady_clock::now();
        packet_ms_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t_pkt_end - t_pkt_start).count();
        
        if (ret < 0) {
            return false;
        }
        
        if (pkt->stream_index != video_stream_index) {
            av_packet_unref(pkt);
            continue;
        }
        
        auto t_dec_start = std::chrono::steady_clock::now();
        ret = avcodec_send_packet(codec_ctx, pkt);
        if (ret < 0) {
            LOG_ERROR("Send packet to decoder failed");
            av_packet_unref(pkt);
            return false;
        }
        
        ret = avcodec_receive_frame(codec_ctx, frame);
        auto t_dec_end = std::chrono::steady_clock::now();
        decode_ms_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t_dec_end - t_dec_start).count();
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_unref(pkt);
            continue;
        } else if (ret < 0) {
            LOG_ERROR("Receive frame from decoder failed");
            av_packet_unref(pkt);
            return false;
        }
        
        last_arrival_steady = std::chrono::steady_clock::now();
        long long best_effort_pts = (frame->best_effort_timestamp == AV_NOPTS_VALUE) ? 0 : frame->best_effort_timestamp;
        long long pts_ms = (stream_time_base.num > 0 && stream_time_base.den > 0)
            ? (best_effort_pts * 1000LL * stream_time_base.num) / stream_time_base.den
            : -1;
        last_pts_ms = pts_ms;

        // Estimate upstream delay
        long long arrival_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(last_arrival_steady.time_since_epoch()).count();
        if (pts_ms >= 0) {
            if (start_time_realtime_us.load() > 0) {
                long long start_ms = start_time_realtime_us.load() / 1000LL;
                last_upstream_latency_ms = arrival_now_ms - (start_ms + pts_ms);
            } else {
                long long expected_ms = LLONG_MIN;
                long long offset = pts_to_steady_offset_ms.load();
                if (offset == LLONG_MIN) {
                    pts_to_steady_offset_ms = arrival_now_ms - pts_ms;
                    expected_ms = arrival_now_ms; 
                } else {
                    expected_ms = pts_ms + offset;
                }
                last_upstream_latency_ms = arrival_now_ms - expected_ms;
            }
        } else {
            last_upstream_latency_ms = -1;
        }


        auto t_sws_start = std::chrono::steady_clock::now();
        img.create(height, width, CV_8UC3);

        uint8_t* dst_data[4] = {img.data, nullptr, nullptr, nullptr};
        int dst_linesize[4] = {static_cast<int>(img.step[0]), 0, 0, 0};
        
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, height,
                 dst_data, dst_linesize);
        auto t_sws_end = std::chrono::steady_clock::now();
        sws_ms_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t_sws_end - t_sws_start).count();
        
        auto t_clone_start = std::chrono::steady_clock::now();
        auto t_clone_end = t_clone_start;
        clone_ms_sum += 0;
        
        auto t_down_start = std::chrono::steady_clock::now();
        auto t_down_end = std::chrono::steady_clock::now();
        downscale_ms_sum += 0;
        
        frame_number = frame_count++;
        
        av_packet_unref(pkt);
        auto total_end = std::chrono::steady_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        g_ff_packet_time_ms += packet_ms_sum;
        g_ff_decode_time_ms += decode_ms_sum;
        g_ff_sws_time_ms += sws_ms_sum;
        g_ff_clone_time_ms += clone_ms_sum;
        g_ff_downscale_time_ms += downscale_ms_sum;
        g_ff_total_time_ms += total_ms;
        g_ff_read_frames++;
        return true;
    }
}

int FFmpegVideoReader::getWidth() const {
    return width;
}

int FFmpegVideoReader::getHeight() const {
    return height;
}

int FFmpegVideoReader::getTotalFrames() const {
    return total_frames;
}

double FFmpegVideoReader::getFPS() const {
    return fps;
}

void FFmpegVideoReader::close() {
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        sws_ctx = nullptr;
    }
    
    if (rgb_frame) {
        av_frame_free(&rgb_frame);
    }
    
    if (frame) {
        av_frame_free(&frame);
    }
    
    if (pkt) {
        av_packet_free(&pkt);
    }
    
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
    }
    
    if (fmt_ctx) {
        avformat_close_input(&fmt_ctx);
    }
    
    frame_count = 0;
    LOG_INFO("Close FFmpeg Video Reader");

    // Print read stage average time statistics
    long long frames = g_ff_read_frames.load();
    if (frames > 0) {
        double avg_packet = static_cast<double>(g_ff_packet_time_ms.load()) / frames;
        double avg_decode = static_cast<double>(g_ff_decode_time_ms.load()) / frames;
        double avg_sws = static_cast<double>(g_ff_sws_time_ms.load()) / frames;
        double avg_clone = static_cast<double>(g_ff_clone_time_ms.load()) / frames;
        double avg_downscale = static_cast<double>(g_ff_downscale_time_ms.load()) / frames;
        double avg_total = static_cast<double>(g_ff_total_time_ms.load()) / frames;
        LOG_INFO("[Read Thread-FFmpeg] Average Packet: " << avg_packet << " ms, Average Decode: " << avg_decode
                 << " ms, Average SWS: " << avg_sws << " ms, Average Clone: " << avg_clone
                 << " ms, Average Downscale: " << avg_downscale << " ms, Average Total: " << avg_total << " ms");
    }
}

} // namespace video 

// Use macro to automatically register FFmpeg video reader
REGISTER_VIDEO_READER(FFMPEG, video::FFmpegVideoReader) 