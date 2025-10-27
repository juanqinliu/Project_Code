#ifndef VIDEO_READER_FACTORY_H
#define VIDEO_READER_FACTORY_H

#include <memory>
#include <string>
#include <map>
#include <functional>
#include <vector> // for getRegisteredReaderTypes
#include "video/VideoReader.h"

namespace video {

// Supported reader types
enum class VideoReaderType {
    OPENCV,
    FFMPEG,
    GSTREAMER
    // New reader types can be added without changing the factory API
};

// Convert enum to string
std::string videoReaderTypeToString(VideoReaderType type);

// Video reader factory
class VideoReaderFactory {
public:
    using CreatorFunc = std::function<std::shared_ptr<VideoReader>()>;
    
    // Register a creator function
    static bool registerReader(VideoReaderType type, CreatorFunc creator);
    
    // Get registered reader types
    static std::vector<VideoReaderType> getRegisteredReaderTypes();
    
    // Set default reader type
    static void setDefaultReaderType(VideoReaderType type);
    
    // Get default reader type
    static VideoReaderType getDefaultReaderType();
    
    // Create a video reader instance
    static std::shared_ptr<VideoReader> createVideoReader(
        VideoReaderType type = VideoReaderFactory::getDefaultReaderType());
    
private:
    // Mapping from type to creator
    static std::map<VideoReaderType, CreatorFunc>& getCreatorMap();
    
    // Default reader type
    static VideoReaderType defaultReaderType;
};

// Auto-registration macro
#define REGISTER_VIDEO_READER(readerType, readerClass) \
    namespace { \
        static bool registered = video::VideoReaderFactory::registerReader( \
            video::VideoReaderType::readerType, \
            []() -> std::shared_ptr<video::VideoReader> { \
                return std::make_shared<readerClass>(); \
            } \
        ); \
    }

} // namespace video

#endif // VIDEO_READER_FACTORY_H 