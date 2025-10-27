#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <iostream>
#include <string>

// Simple TensorRT logger implementation
class Logger : public nvinfer1::ILogger
{
public:
    // Define log severity enum, corresponding to TensorRT levels
    enum class Severity
    {
        kINTERNAL_ERROR = 0,  // Corresponds to nvinfer1::ILogger::Severity::kINTERNAL_ERROR
        kERROR = 1,           // Corresponds to nvinfer1::ILogger::Severity::kERROR
        kWARNING = 2,         // Corresponds to nvinfer1::ILogger::Severity::kWARNING
        kINFO = 3,            // Corresponds to nvinfer1::ILogger::Severity::kINFO
        kVERBOSE = 4          // Corresponds to nvinfer1::ILogger::Severity::kVERBOSE
    };

    Logger(Severity severity = Severity::kINFO) : mSeverity(severity) {}

    void log(Severity severity, const char* msg) noexcept
    {
        if (severity <= mSeverity)
        {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[F] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[E] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "[W] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[I] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[V] " << msg << std::endl;
                break;
            default:
                std::cout << "[?] " << msg << std::endl;
                break;
        }
        }
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        log(static_cast<Severity>(static_cast<int>(severity)), msg);
    }

    Severity getReportableSeverity() const { return mSeverity; }

    void setReportableSeverity(Severity severity) { mSeverity = severity; }

    nvinfer1::ILogger::Severity getTRTSeverity() const 
    { 
        return static_cast<nvinfer1::ILogger::Severity>(static_cast<int>(mSeverity)); 
    }

    // Get ILogger interface
    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }

private:
    Severity mSeverity{Severity::kINFO};
};

namespace sample
{
    // Global logger instance
    static Logger gLogger{Logger::Severity::kINFO};
    
    #define gLogError   (gLogger.log(Logger::Severity::kERROR, ""))
    #define gLogWarning (gLogger.log(Logger::Severity::kWARNING, ""))
    #define gLogInfo    (gLogger.log(Logger::Severity::kINFO, ""))
    #define gLogVerbose (gLogger.log(Logger::Severity::kVERBOSE, ""))
}

// Simple logging macro definitions
#define LOG(severity) sample::gLogger.log(sample::Logger::Severity::severity, "")

#endif // LOGGER_H 