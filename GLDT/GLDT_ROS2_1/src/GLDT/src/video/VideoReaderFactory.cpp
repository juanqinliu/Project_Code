#include "video/VideoReaderFactory.h"
#include "common/Logger.h"

// Initialize static member variable, default using OpenCV reader
video::VideoReaderType video::VideoReaderFactory::defaultReaderType = video::VideoReaderType::OPENCV;

// Convert enum to string
std::string video::videoReaderTypeToString(VideoReaderType type) {
    switch (type) {
        case VideoReaderType::OPENCV:
            return "OpenCV";
        case VideoReaderType::FFMPEG:
            return "FFmpeg";
        case VideoReaderType::GSTREAMER:
            return "GStreamer";
        default:
            return "Unknown";
    }
}

// Get creator mapping table
std::map<video::VideoReaderType, video::VideoReaderFactory::CreatorFunc>& 
video::VideoReaderFactory::getCreatorMap() {
    static std::map<VideoReaderType, CreatorFunc> creators;
    return creators;
}

// Register creator function
bool video::VideoReaderFactory::registerReader(VideoReaderType type, CreatorFunc creator) {
    auto& creators = getCreatorMap();
    
    // Check if it has been registered
    if (creators.find(type) != creators.end()) {
        LOG_WARNING("Video reader type " << videoReaderTypeToString(type) << " has been registered, will be overridden");
    }
    
    // Register creator function
    creators[type] = creator;
    LOG_INFO("Video reader type " << videoReaderTypeToString(type) << " has been registered to the factory");
    return true;
}

// Get the list of registered reader types
std::vector<video::VideoReaderType> video::VideoReaderFactory::getRegisteredReaderTypes() {
    std::vector<VideoReaderType> types;
    const auto& creators = getCreatorMap();
    
    for (const auto& pair : creators) {
        types.push_back(pair.first);
    }
    
    return types;
}

void video::VideoReaderFactory::setDefaultReaderType(VideoReaderType type) {
    const auto& creators = getCreatorMap();
    
    // Check if the type has been registered
    if (creators.find(type) == creators.end()) {
        LOG_ERROR("Failed to set default video reader type: " << videoReaderTypeToString(type) << " not registered");
        return;
    }
    
    defaultReaderType = type;
    LOG_INFO("Set default video reader type to: " << videoReaderTypeToString(type));
}

video::VideoReaderType video::VideoReaderFactory::getDefaultReaderType() {
    return defaultReaderType;
}

std::shared_ptr<video::VideoReader> video::VideoReaderFactory::createVideoReader(VideoReaderType type) {
    const auto& creators = getCreatorMap();
    auto it = creators.find(type);
    
    if (it == creators.end()) {
        LOG_ERROR("Unknown video reader type: " << videoReaderTypeToString(type));
        return nullptr;
    }
    
    LOG_INFO("Create video reader through factory: " << videoReaderTypeToString(type));
    return it->second();
} 