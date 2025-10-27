// 系统和标准库头文件
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <csignal>
#include <unistd.h>
#include <atomic>
#include <iomanip> // Added for std::fixed and std::setprecision

// Project header files
#include "common/Logger.h"
#include "SingleThreadTrackingApp.h"
#include "MultiThreadTrackingApp.h"
#include "video/VideoReaderFactory.h"
#include "common/Flags.h"

// Global exit flag from multi-thread implementation (for graceful exit)
namespace tracking { extern std::atomic<bool> should_exit; }

// Signal handling: support Ctrl+C (SIGINT) / Terminate (SIGTERM)随时安全退出
namespace {
    std::atomic<int> signal_count{0};

    void handle_signal(int signum) {
        int count = signal_count.fetch_add(1) + 1;
        
        if (count == 1) {
        // First signal: set graceful exit flag
        tracking::should_exit = true;
        const char msg[] = "Caught interrupt signal, graceful exit...\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
    } else if (count == 2) {
        // Second signal: force exit
        const char msg[] = "Caught interrupt signal again, force exit...\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
        std::_Exit(1);  // Exit immediately, do not call destructor
    } else {
        // Third or more signals: exit immediately
        std::_Exit(2);
    }
    }
}

int main(int argc, char* argv[]) {
    try {
        // Register signal handler, ensure Ctrl+C can be terminated随时终止
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        // Initialize gflags, load configuration file (use absolute path, avoid working directory differences causing not loaded)
        flags::init(&argc, &argv, "src/GLDT/config/config.flag");
        
        // Validate flag values
        if (!flags::validateFlags()) {
            std::cerr << "Configuration parameter validation failed, program exit" << std::endl;
            return 1;
        }
        
        // Ensure log directory exists
        int mkdir_result = system(("mkdir -p " + FLAGS_custom_log_dir).c_str());
        if (mkdir_result != 0) {
            std::cerr << "Warning: Failed to create log directory: " << FLAGS_custom_log_dir << std::endl;
        }
        
        // Initialize log system - use the same absolute path configuration file
        Logger::init(argv[0], "src/GLDT/config/config.flag");
        
        // Check parameters
        if (argc < 3) {
            LOG_ERROR("Usage: " << argv[0] << " <global detection model path> <local detection model path> [<video path> <output directory>]");
            return 1;
        }
        
        std::string global_model_path = argv[1];
        std::string local_model_path = argv[2];
        
        // Video path and output directory can be obtained from command line parameters or configuration file
        std::string video_path = (argc > 3) ? argv[3] : "input.mp4";
        std::string output_dir = (argc > 4) ? argv[4] : "output";
               
        std::cout << "Start single thread tracking system" << std::endl;
        std::cout << "Global detection model: " << global_model_path << std::endl;
        std::cout << "Local detection model: " << local_model_path << std::endl;
        std::cout << "Video path: " << video_path << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        // std::cout << "Video reader: " << video::videoReaderTypeToString(reader_type) << std::endl;
        
        // Apply configuration values to the system (here it is assumed that the Config class has been updated to accept these parameters)
        tracking::Config config;
        config.global_conf_thres = FLAGS_global_conf_threshold;
        config.global_nms_thres = FLAGS_global_nms_threshold;
        config.local_conf_thres = FLAGS_local_conf_threshold;
        config.local_nms_thres = FLAGS_local_nms_threshold;
        config.roi_margin = FLAGS_roi_margin;
        config.roi_min_size = FLAGS_roi_min_size;
        config.roi_max_size = FLAGS_roi_max_size;
        config.log_level = FLAGS_log_level;
        config.batch_size = FLAGS_batch_size;
        config.enable_batch = FLAGS_enable_batch_inference;
        config.enable_parallel = FLAGS_enable_parallel_processing;
        config.num_threads = FLAGS_num_threads;

        // Print critical thresholds and log switches at startup,便于诊断“低置信度被阈值过滤”的问题
        LOG_INFO("Startup thresholds: global_conf_thres=" << std::fixed << std::setprecision(3)
                 << config.global_conf_thres << ", local_conf_thres=" << config.local_conf_thres
                 << ", global_nms_thres=" << config.global_nms_thres
                 << ", local_nms_thres=" << config.local_nms_thres);
        LOG_INFO("Verbose logging: " << (config.verbose_logging ? "ON" : "OFF"));
        
        // Check if the exit signal has been received
        if (tracking::should_exit) {
            std::cout << "Program received exit signal before startup, exiting..." << std::endl;
            return 0;
        }
        
        if (FLAGS_single_thread_mode) {
            tracking::SingleThreadTrackingApp app(global_model_path, local_model_path, video_path, output_dir);
            app.run();
        } else {
            // Multi-thread mode
            tracking::MultiThreadTrackingApp app(global_model_path, local_model_path, video_path, output_dir);
            app.run();
        }
        
        std::cout << "Program exited normally" << std::endl;
        Logger::shutdown();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        Logger::shutdown();
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        Logger::shutdown();
        return 1;
    }
}