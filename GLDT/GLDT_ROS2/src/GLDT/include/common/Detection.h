#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

namespace tracking {

// Detection result
struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
    bool is_from_local_model = false;  // whether produced by local model
    bool is_from_global_model = false; // whether produced by global model
    cv::Mat appearance;                // appearance features
    cv::Point2f center() const {
        return cv::Point2f(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
    }
};

} // namespace tracking

#endif // DETECTION_H 