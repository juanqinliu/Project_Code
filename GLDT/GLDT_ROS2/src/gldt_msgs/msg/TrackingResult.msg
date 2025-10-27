# Tracking detection result message

# Message header
std_msgs/Header header

# Frame information
int32 frame_number
int32 frame_width
int32 frame_height

# Detection results
Inference[] detections

# Tracking results
Track[] tracks

# Performance statistics
float64 fps

