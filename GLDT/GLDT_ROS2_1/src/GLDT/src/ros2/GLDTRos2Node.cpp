#include "ros2/GLDTRos2Node.h"
#include <opencv2/opencv.hpp>
#include <chrono>

namespace tracking {

GLDTRos2Node::GLDTRos2Node() 
    : Node("gldt_tracking_node"), enable_publishing_(true) {
    
    // Declare parameters
    this->declare_parameter("enable_ros_publishing", true);
    this->declare_parameter("publish_rate", 30.0);
    
    // Get parameters
    enable_publishing_ = this->get_parameter("enable_ros_publishing").as_bool();
    
    if (enable_publishing_) {
        // Create publishers
        tracking_result_pub_ = this->create_publisher<gldt_msgs::msg::TrackingResult>(
            "/gldt/tracking_result", 10);
        tracking_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/gldt/tracking_image", 10);
        
        RCLCPP_INFO(this->get_logger(), "GLDT ROS2 node has started, publishing topics:");
        RCLCPP_INFO(this->get_logger(), "  /gldt/tracking_result - Tracking detection results");
        RCLCPP_INFO(this->get_logger(), "  /gldt/tracking_image - Visualization image");
    } else {
        RCLCPP_INFO(this->get_logger(), "ROS2 publishing is disabled");
    }
}

GLDTRos2Node::~GLDTRos2Node() {
    // Clean up resources
}

void GLDTRos2Node::publishTrackingResult(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::vector<std::unique_ptr<STrack>>& tracks,
    int frame_number,
    double fps) {
    
    if (!enable_publishing_ || !tracking_result_pub_) {
        return;
    }
    
    try {
        auto msg = convertToTrackingResult(frame, detections, tracks, frame_number, fps);
        tracking_result_pub_->publish(msg);
        
        // Optional: publish frequency control
        static auto last_publish_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_publish_time);
        
        if (elapsed.count() >= 33) { // Approximate 30fps
            last_publish_time = now;
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error publishing tracking results: %s", e.what());
    }
}

void GLDTRos2Node::publishVisualizationImage(
    const cv::Mat& vis_frame,
    int frame_number) {
    
    if (!enable_publishing_ || !tracking_image_pub_) {
        return;
    }
    
    try {
        auto img_msg = convertToImageMsg(vis_frame);
        img_msg->header.frame_id = "camera_frame";
        img_msg->header.stamp = this->now();
        tracking_image_pub_->publish(*img_msg);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Publish Visualization Image Error: %s", e.what());
    }
}

gldt_msgs::msg::TrackingResult GLDTRos2Node::convertToTrackingResult(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::vector<std::unique_ptr<STrack>>& tracks,
    int frame_number,
    double fps) {
    
    gldt_msgs::msg::TrackingResult msg;
    
    // Set message header
    msg.header.stamp = this->now();
    msg.header.frame_id = "camera_frame";
    
    // Set frame information
    msg.frame_number = frame_number;
    msg.frame_width = frame.cols;
    msg.frame_height = frame.rows;
    msg.fps = fps;  // Processing FPS (system processing capability)
    
    // Convert detection results
    msg.detections.clear();
    for (const auto& detection : detections) {
        msg.detections.push_back(convertToInference(detection));
    }
    
    // Convert tracking results
    msg.tracks.clear();
    for (const auto& track : tracks) {
        if (track) {
            msg.tracks.push_back(convertToTrack(*track));
        }
    }
    
    return msg;
}

gldt_msgs::msg::Track GLDTRos2Node::convertToTrack(const STrack& track) {
    gldt_msgs::msg::Track msg;
    
    msg.track_id = track.displayId();
    msg.class_id = track.class_id;
    msg.class_name = "drone"; // Set to drone category
    msg.confidence = track.score;
    
    // Convert bounding box
    msg.bbox = convertToBoundingBox(track.tlwh);
    
    return msg;
}

gldt_msgs::msg::Inference GLDTRos2Node::convertToInference(const Detection& detection) {
    gldt_msgs::msg::Inference msg;
    
    msg.class_id = detection.class_id;
    msg.class_name = "drone"; 
    msg.score = detection.confidence;
    
    // Convert bounding box
    msg.bbox = convertToBoundingBox(detection.bbox);
    
    return msg;
}

gldt_msgs::msg::BoundingBox GLDTRos2Node::convertToBoundingBox(const cv::Rect2f& bbox) {
    gldt_msgs::msg::BoundingBox msg;
    
    // Center point
    msg.center.x = bbox.x + bbox.width / 2.0;
    msg.center.y = bbox.y + bbox.height / 2.0;
    
    // Size
    msg.size.x = bbox.width;
    msg.size.y = bbox.height;
    
    return msg;
}

gldt_msgs::msg::Point2D GLDTRos2Node::convertToPoint2D(const cv::Point2f& point) {
    gldt_msgs::msg::Point2D msg;
    msg.x = point.x;
    msg.y = point.y;
    return msg;
}


sensor_msgs::msg::Image::SharedPtr GLDTRos2Node::convertToImageMsg(const cv::Mat& image) {
    try {
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = this->now();
        cv_image.header.frame_id = "camera_frame";
        
        if (image.channels() == 3) {
            cv_image.encoding = "bgr8";
        } else if (image.channels() == 1) {
            cv_image.encoding = "mono8";
        } else {
            RCLCPP_WARN(this->get_logger(), "Unsupported image channels: %d", image.channels());
            cv_image.encoding = "bgr8";
        }
        
        cv_image.image = image.clone();
        return cv_image.toImageMsg();
        
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge Error: %s", e.what());
        return nullptr;
    }
}

} // namespace tracking
