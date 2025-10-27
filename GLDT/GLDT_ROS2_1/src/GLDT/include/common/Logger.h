#pragma once

#include <glog/logging.h>
#include <string>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <map>

/**
 * @brief Logger wrapper based on Google glog
 */
class Logger {
private:
    // Global settings
    static bool initialized_;
    static int currentLogLevel_;
    static std::string logDir_;
    static bool toConsole_;
    static bool toFile_;

public:
    /**
     * @brief Parse config.flag and extract parameters
     * @param path Path to the config file
     * @return Map of parameter names to values
     */
    static std::map<std::string, std::string> parseConfigFile(const std::string& path) {
        std::map<std::string, std::string> config;
        std::ifstream infile(path);
        
        if (!infile.is_open()) {
            std::cerr << "Warning: cannot open config file: " << path << std::endl;
            return config;
        }
        
        std::string line;
        while (std::getline(infile, line)) {
            // Skip empty and comment lines
            if (line.empty() || line[0] == '#') continue;
            
            // Find '=' position
            auto pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            // Extract key-value
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Remove leading "--" and trim spaces
            if (key.rfind("--", 0) == 0) key = key.substr(2);
            
            // Trim spaces
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            config[key] = value;
        }
        
        return config;
    }

    /**
     * @brief Initialize logging system
     * @param argv0 Program name
     * @param configPath Path to config file
     */
    static void init(const char* argv0, const std::string& configPath = "./config/config.flag") {
        // Parse config file
        auto config = parseConfigFile(configPath);
        
        // Read logging config
        currentLogLevel_ = 0;  // default
        toConsole_ = true;     // default
        toFile_ = true;        // default
        logDir_ = "logs";      // default
        
        // Read values from config
        try {
            if (config.find("log_level") != config.end()) {
                currentLogLevel_ = std::stoi(config["log_level"]);
            }
            
            // Console output
            if (config.find("log_to_console") != config.end()) {
                toConsole_ = (config["log_to_console"] == "true");
            }
            
            // File output
            if (config.find("log_to_file") != config.end()) {
                toFile_ = (config["log_to_file"] == "true");
            }
            
            // Log directory
            if (config.find("custom_log_dir") != config.end()) {
                logDir_ = config["custom_log_dir"];
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: parse config file parameters failed: " << e.what() << std::endl;
            std::cerr << "Use default log configuration continue" << std::endl;
        }
        
        // Ensure log directory exists
        if (toFile_ && !logDir_.empty()) {
            createDirectoryIfNotExists(logDir_);
        }
        
        // Initialize glog
        google::InitGoogleLogging(argv0);
        
        // Set glog flags
        FLAGS_minloglevel = currentLogLevel_;
        FLAGS_stderrthreshold = currentLogLevel_;
        
        // Control output targets
        if (toConsole_ && !toFile_) {
            // Console only
            FLAGS_logtostderr = 1;
            FLAGS_alsologtostderr = 0;
        } else if (!toConsole_ && toFile_) {
            // File only
            FLAGS_logtostderr = 0;
            FLAGS_alsologtostderr = 0;
            FLAGS_log_dir = logDir_;
        } else if (toConsole_ && toFile_) {
            // Console and file
            FLAGS_logtostderr = 0;
            FLAGS_alsologtostderr = 1;
            FLAGS_log_dir = logDir_;
        } else {
            // Fallback: console
            FLAGS_logtostderr = 1;
            FLAGS_alsologtostderr = 0;
        }
        
        // Other settings
        FLAGS_colorlogtostderr = true;      // colorized logs
        
        // Optional settings
        try {
            if (config.find("color_log") != config.end()) {
                FLAGS_colorlogtostderr = (config["color_log"] == "true");
            }
            
            if (config.find("max_log_size") != config.end()) {
                FLAGS_max_log_size = std::stoi(config["max_log_size"]);
            } else {
                FLAGS_max_log_size = 10;    // rotate file every 10MB by default
            }
            
            if (config.find("verbose_log") != config.end()) {
                FLAGS_v = (config["verbose_log"] == "true") ? 10 : 0;
            } else {
                FLAGS_v = 0;                // verbose disabled by default
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: parse config file optional parameters failed: " << e.what() << std::endl;
            std::cerr << "Use default value continue" << std::endl;
            FLAGS_max_log_size = 10;
            FLAGS_v = 0;
        }
        
        FLAGS_logbuflevel = -1;             // -1 means no buffering (immediate flush)
        FLAGS_stop_logging_if_full_disk = true; // stop logging when disk is full
        
        // Print initialization info
        std::cout << "=== Logging system initialized (using config.flag) ===" << std::endl;
        std::cout << "Program name: " << argv0 << std::endl;
        std::cout << "Config file: " << configPath << std::endl;
        std::cout << "Log level: " << currentLogLevel_ << " (" << logLevelToString(currentLogLevel_) << ")" << std::endl;
        std::cout << "Output to console: " << (toConsole_ ? "yes" : "no") << std::endl;
        std::cout << "Output to file: " << (toFile_ ? "yes" : "no") << std::endl;
        if (toFile_) {
            std::cout << "Log directory: " << logDir_ << std::endl;
            std::cout << "Log directory exists: " << (directoryExists(logDir_) ? "yes" : "no") << std::endl;
        }
        
        // Show glog internal settings
        std::cout << "glog internal settings:" << std::endl;
        std::cout << " - FLAGS_logtostderr: " << FLAGS_logtostderr << std::endl;
        std::cout << " - FLAGS_alsologtostderr: " << FLAGS_alsologtostderr << std::endl;
        std::cout << " - FLAGS_minloglevel: " << FLAGS_minloglevel << std::endl;
        std::cout << " - FLAGS_stderrthreshold: " << FLAGS_stderrthreshold << std::endl;
        std::cout << " - FLAGS_v: " << FLAGS_v << std::endl;
        std::cout << " - FLAGS_max_log_size: " << FLAGS_max_log_size << std::endl;
        std::cout << " - FLAGS_colorlogtostderr: " << FLAGS_colorlogtostderr << std::endl;
        std::cout << "===================" << std::endl;
        
        initialized_ = true;
        
        // Log initialization info via glog INFO
        google::LogMessage(__FILE__, __LINE__, google::GLOG_INFO).stream() 
            << "Logging system initialized, path: " << (logDir_.empty() ? "standard output" : logDir_)
            << ", level: " << logLevelToString(currentLogLevel_);
    }
    
    /**
     * @brief Set log output level
     * @param level 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
     */
    static void setLogLevel(int level) {
        if (level < 0 || level > 3) {
            std::cerr << "Warning: invalid log level " << level << ", use range [0-3]" << std::endl;
            level = std::max(0, std::min(level, 3));  // clamp to [0,3]
        }
        
        currentLogLevel_ = level;
        
        // Update glog settings
        FLAGS_minloglevel = level;
        FLAGS_stderrthreshold = level;

        // Log level change via glog
        if (level <= 0) { // only output when INFO allowed
            google::LogMessage(__FILE__, __LINE__, google::GLOG_INFO).stream() 
                << "Log level set to: " << level << " (" << logLevelToString(level) << ")";
        }
    }
    
    /**
     * @brief Enable/disable console output
     * @param toConsole true to enable console output
     */
    static void setConsoleOutput(bool toConsole) {
        toConsole_ = toConsole;
        
        // Update glog settings
        if (toConsole_ && !toFile_) {
            FLAGS_logtostderr = 1;
            FLAGS_alsologtostderr = 0;
        } else if (!toConsole_ && toFile_) {
            FLAGS_logtostderr = 0;
            FLAGS_alsologtostderr = 0;
        } else if (toConsole_ && toFile_) {
            FLAGS_logtostderr = 0;
            FLAGS_alsologtostderr = 1;
        }
    }
    
    /**
     * @brief Enable/disable file output
     * @param toFile true to enable file output
     */
    static void setFileOutput(bool toFile) {
        toFile_ = toFile;
        
        // Update glog settings
        if (toConsole_ && !toFile_) {
            FLAGS_logtostderr = 1;
            FLAGS_alsologtostderr = 0;
        } else if (!toConsole_ && toFile_) {
            FLAGS_logtostderr = 0;
            FLAGS_alsologtostderr = 0;
            FLAGS_log_dir = logDir_;
        } else if (toConsole_ && toFile_) {
            FLAGS_logtostderr = 0;
            FLAGS_alsologtostderr = 1;
            FLAGS_log_dir = logDir_;
        }
    }
    
    /**
     * @brief Enable/disable verbose logs
     * @param verbose true to enable
     */
    static void setVerbose(bool verbose) {
        FLAGS_v = verbose ? 10 : 0;
        
        // Log verbosity change via glog
        if (currentLogLevel_ <= 0) { // only output when INFO allowed
            google::LogMessage(__FILE__, __LINE__, google::GLOG_INFO).stream() 
                << "Verbose logs " << (verbose ? "enabled" : "disabled") << " (level: " << FLAGS_v << ")";
        }
    }
    
    /**
     * @brief Check if a VLOG level is enabled
     * @param level verbosity level
     * @return true if enabled
     */
    static bool isVLogEnabled(int level) {
        return FLAGS_v >= level;
    }
    
    /**
     * @brief Shutdown logging system
     */
    static void shutdown() {
        google::ShutdownGoogleLogging();
        initialized_ = false;
    }

    /**
     * @brief Get string representation of a log level
     */
    static inline const char* logLevelToString(int level) {
        switch(level) {
            case 0: return "INFO";
            case 1: return "WARNING";
            case 2: return "ERROR";
            case 3: return "FATAL";
            default: return "UNKNOWN";
        }
    }

private:
    /**
     * @brief Check if a directory exists
     * @param path directory path
     * @return true if exists
     */
    static bool directoryExists(const std::string& path) {
        struct stat info;
        return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
    }

    /**
     * @brief Create a directory if missing
     * @param path directory path
     */
    static void createDirectoryIfNotExists(const std::string& path) {
        if (!directoryExists(path)) {
            // Not exists, try to create
            #ifdef _WIN32
                int result = mkdir(path.c_str());
            #else
                int result = mkdir(path.c_str(), 0755);
            #endif
            
            if (result != 0) {
                std::cerr << "Warning: cannot create log directory: " << path << std::endl;
                perror("mkdir failed reason");
            } else {
                std::cout << "Log directory created: " << path << std::endl;
            }
        }
    }
};

// Simplified logging macros - use glog macros with custom tags
#define LOG_DEBUG(msg) LOG(INFO) << "[DEBUG] " << msg
#define LOG_INFO(msg) LOG(INFO) << msg
#define LOG_WARNING(msg) LOG(WARNING) << msg
#define LOG_ERROR(msg) LOG(ERROR) << msg
#define LOG_FATAL(msg) LOG(FATAL) << msg

// Conditional logging macros
#define LOG_INFO_IF(condition, msg) LOG_IF(INFO, condition) << msg
#define LOG_WARNING_IF(condition, msg) LOG_IF(WARNING, condition) << msg
#define LOG_ERROR_IF(condition, msg) LOG_IF(ERROR, condition) << msg

// Verbose logging (controlled by --v=N)
#define LOG_VERBOSE(level, msg) VLOG(level) << msg

// Log every N times (use different name to avoid glog conflict)
#define LOG_EVERY_N_TIMES(times, msg) LOG_EVERY_N(INFO, times) << msg 