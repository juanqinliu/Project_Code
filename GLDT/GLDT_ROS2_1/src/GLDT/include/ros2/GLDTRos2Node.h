#ifndef GLDT_ROS2_NODE_H
#define GLDT_ROS2_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>

// GLDT message types
#include "gldt_msgs/msg/tracking_result.hpp"
#include "gldt_msgs/msg/track.hpp"
#include "gldt_msgs/msg/inference.hpp"
#include "gldt_msgs/msg/bounding_box.hpp"
#include "gldt_msgs/msg/point2_d.hpp"
#include "gldt_msgs/msg/mask.hpp"

#include "common/Detection.h"
#include "tracking/STrack.h"
#include "common/Logger.h"

namespace tracking {

/**
 * ROS2 node class, for publishing GLDT tracking detection results
 */
class GLDTRos2Node : public rclcpp::Node {
public:
    GLDTRos2Node();
    ~GLDTRos2Node();

    /**
     * Publish tracking detection results
     * @param frame Original frame image
     * @param detections Detection results
     * @param tracks Tracking results
     * @param frame_number Frame number
     * @param fps Frame rate
     */
    void publishTrackingResult(
        const cv::Mat& frame,
        const std::vector<Detection>& detections,
        const std::vector<std::unique_ptr<STrack>>& tracks,
        int frame_number,
        double fps
    );

    /**
     * Publish visualization image
     * @param vis_frame Visualization image
     * @param frame_number Frame number
     */
    void publishVisualizationImage(
        const cv::Mat& vis_frame,
        int frame_number
    );

    /**
     * Check if ROS2 publishing is enabled
     * @return true if enabled, false otherwise
     */
    bool isPublishingEnabled() const { return enable_publishing_; }

private:
    // ROS2 publishers
    rclcpp::Publisher<gldt_msgs::msg::TrackingResult>::SharedPtr tracking_result_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracking_image_pub_;
    
    // Publishing control
    bool enable_publishing_;
    
    // Message conversion helper functions
    gldt_msgs::msg::TrackingResult convertToTrackingResult(
        const cv::Mat& frame,
        const std::vector<Detection>& detections,
        const std::vector<std::unique_ptr<STrack>>& tracks,
        int frame_number,
        double fps
    );
    
    gldt_msgs::msg::Track convertToTrack(const STrack& track);
    gldt_msgs::msg::Inference convertToInference(const Detection& detection);
    gldt_msgs::msg::BoundingBox convertToBoundingBox(const cv::Rect2f& bbox);
    gldt_msgs::msg::Point2D convertToPoint2D(const cv::Point2f& point);
    
    // Image conversion helper functions
    sensor_msgs::msg::Image::SharedPtr convertToImageMsg(const cv::Mat& image);
};

} // namespace tracking

#endif // GLDT_ROS2_NODE_H
