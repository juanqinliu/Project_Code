#!/bin/bash

# ROS2 GLDT tracking detection system startup script

# Set log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
echo "Created log directory: $LOG_DIR"

GLOBAL_MODEL="Weights/global_int8.engine"
LOCAL_MODEL="Weights/local_fp16.engine"

# GLOBAL_MODEL="Weights/weights/global_fixed_best_train63_int8.engine"
# LOCAL_MODEL="Weights/weights/local_fixedwings_train62.engine"

VIDEO_PATH="Videos"   # Video path
OUTPUT_DIR="Results"  # Output directory

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXECUTABLE_PATH="${SCRIPT_DIR}/install/gldt/lib/gldt/gldt_node"

# Supported video formats
VIDEO_EXTENSIONS=("mp4" "avi" "mov" "mkv" "flv" "wmv" "m4v")

# Show help information
show_help() {
    echo "ROS2 GLDT tracking detection system startup script"
    echo "Usage: $0 [options]"
    echo ""
    echo "选项:"
    echo "  -g, --global PATH    Global model path (default: $GLOBAL_MODEL)"
    echo "  -l, --local PATH     Local model path (default: $LOCAL_MODEL)"
    echo "  -v, --video PATH     Video source path/index/pipeline (default: $VIDEO_PATH)"
    echo "                       - Folder: process all supported video files"
    echo "                       - File: process a single video file"
    echo "                       - Device: /dev/video0 etc. character device"
    echo "                       - Index: 0, 1 etc. OpenCV camera index"
    echo "                       - RTSP/HTTP: rtsp://、http(s)://"
    echo "                       - GStreamer: v4l2src/nvarguscamerasrc/rtspsrc 管线"
    echo "  -o, --output PATH    Output directory path (default: $OUTPUT_DIR)"
    echo "  --pattern PATTERN    File name matching pattern (default: process all supported formats)"
    echo "                       Example: --pattern \"DJI_*\" process files starting with DJI_"
    echo "  --skip-existing      Skip existing result files"
    echo "  --bytetrack          Use original ByteTrack algorithm (no ROI constraint)"
    echo "  --enhanced-tracker   Use enhanced tracker (with ROI constraint, default)"
    echo "  --detection-mode MODE  Set detection mode (0-1)"
    echo "                       0: Use global detection (no ROI local detection)"
    echo "                       1: Global+local joint detection (default)"
    echo "  --global-only        Use global detection (short for --detection-mode 0)"
    echo "  --combined-detection Use global+local joint detection (short for --detection-mode 1)"
    echo "  --ros-topics         Show ROS2 topic information"
    echo "  --no-ros             Disable ROS2 publishing (only local processing)"
    echo "  -h, --help           Show this help information"
    echo ""
    echo "ROS2 topics:"
    echo "  Published topics:"
    echo "    /gldt/tracking_result  - Complete tracking detection results"
    echo "    /gldt/tracking_image   - Visualization image"
    echo "Detection mode说明:"
    echo "  Mode 0 (only global detection): Use global detection model, no ROI local detection"
    echo "  Mode 1 (global+local joint detection): Combine global and local detection, dynamically manage ROI"
    echo ""
    echo "Tracking algorithm说明:"
    echo "  Original ByteTrack: Use original ByteTrack tracking algorithm, only depend on IoU matching"
    echo "  Enhanced tracker: Use enhanced algorithm with ROI constraint and memory recovery (default)"
    echo ""
    echo "Supported video formats: ${VIDEO_EXTENSIONS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Process all videos in the default folder"
    echo "  $0 -v /path/to/video.mp4              # Process single video file"
    echo "  $0 -v /path/to/videos/                # Process all videos in the folder"
    echo "  $0 -v /path/to/videos/ --pattern \"DJI_*\" # Process videos starting with DJI_"
    echo "  $0 -v /dev/video0                      # Read USB camera (character device)"
    echo "  $0 -v 0                                # Read camera index 0"
    echo "  $0 -v \"rtsp://user:pass@IP:554/xxx\"   # Read RTSP stream"
    echo "  $0 -v \"v4l2src device=/dev/video0 ! ... ! appsink\" # GStreamer pipeline"
    echo "  $0 --global-only                      # Use global detection mode"
    echo "  $0 --detection-mode 0                 # Use global detection mode"
    echo "  $0 --combined-detection               # Use global+local joint detection mode"
    echo "  $0 --skip-existing                    # Skip existing results"
    echo "  $0 --bytetrack                        # Use original ByteTrack algorithm"
    echo "  $0 --ros-topics                       # Show ROS2 topic information"
    echo "  $0 --no-ros                           # Disable ROS2 publishing"
}

# Check if the file extension is a supported video format
is_video_file() {
    local file="$1"
    local ext=$(echo "${file##*.}" | tr '[:upper:]' '[:lower:]')
    
    for video_ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [[ "$ext" == "$video_ext" ]]; then
            return 0
        fi
    done
    return 1
}

# Get video file list
get_video_files() {
    local path="$1"
    local pattern="$2"
    local files=()
    
    if [[ -f "$path" ]]; then
        # Single file
        if is_video_file "$path"; then
            files=("$path")
        fi
    elif [[ -d "$path" ]]; then
        # Folder
        while IFS= read -r -d '' file; do
            if is_video_file "$file"; then
                files+=("$file")
            fi
        done < <(find "$path" -maxdepth 1 -name "${pattern:-*}" -type f -print0 | sort -z)
    fi
    
    printf '%s\n' "${files[@]}"
}

# Show ROS2 topic information
show_ros_topics() {
    echo "🔍 ROS2 topic information:"
    echo ""
    echo "Published topics:"
    echo "  /gldt/tracking_result  - Tracking detection results (gldt_msgs/msg/TrackingResult)"
    echo "  /gldt/tracking_image   - Visualization image (sensor_msgs/msg/Image)"
    echo ""
    echo "View topic commands:"
    echo "  ros2 topic list"
    echo "  ros2 topic echo /gldt/tracking_result"
    echo "  ros2 topic echo /gldt/tracking_image"
    echo "  ros2 topic hz /gldt/tracking_result"
    echo ""
    echo "View message types:"
    echo "  ros2 interface show gldt_msgs/msg/TrackingResult"
    echo "  ros2 interface show gldt_msgs/msg/Track"
    echo "  ros2 interface show gldt_msgs/msg/Yolov8Inference"
}

# Process single video file
process_single_video() {
    local video_file="$1"
    local video_basename=$(basename "$video_file")
    local video_name="${video_basename%.*}"
    
    echo ""
    echo "========================================"
    echo "Process video: $video_basename"
    echo "========================================"
    
    # Check if the file already exists
    if [[ "$SKIP_EXISTING" == "true" ]]; then
        local result_file="$OUTPUT_DIR/${video_name}_results.txt"
        local video_result_file="$OUTPUT_DIR/${video_name}_multithread_result.mp4"
        
        if [[ -f "$result_file" && -f "$video_result_file" ]]; then
            echo "Skip $video_basename (result file already exists)"
            return 0
        fi
    fi
    
    # Run tracking detection
    echo "Start processing: $video_file"
    echo "Output directory: $OUTPUT_DIR"
    echo "Detection mode: $DETECTION_MODE (0=only global detection, 1=global+local joint detection)"
    echo "Global model: $GLOBAL_MODEL"
    echo "Local model: $LOCAL_MODEL"
    echo "ROS2 Publishing: $([ "$ENABLE_ROS" == "true" ] && echo "Enabled" || echo "Disabled")"
    echo "Processing mode: Deterministic processing (no frame drop, ensure result consistency)"
    echo ""
    
    # Set environment variables
    export USE_ORIGINAL_BYTETRACK="$USE_ORIGINAL_BYTETRACK"
    export DETECTION_MODE="$DETECTION_MODE"
    export ENABLE_ROS_PUBLISHING="$ENABLE_ROS"
    
    # Build command parameters
    echo "Execute command: $EXECUTABLE_PATH \"$GLOBAL_MODEL\" \"$LOCAL_MODEL\" \"$video_file\" \"$OUTPUT_DIR\""
    echo "Use environment variables to control algorithm: Original ByteTrack=$USE_ORIGINAL_BYTETRACK, Detection mode=$DETECTION_MODE, ROS publishing=$ENABLE_ROS"
    
    # Execute program
    "$EXECUTABLE_PATH" "$GLOBAL_MODEL" "$LOCAL_MODEL" "$video_file" "$OUTPUT_DIR"
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ $video_basename processing completed"
        return 0
    else
        echo "❌ $video_basename processing failed (exit code: $exit_code)"
        return $exit_code
    fi
}

# Main program
main() {
    local PATTERN=""
    SKIP_EXISTING=false
    ENABLE_ROS=true
    SHOW_TOPICS=false
    
    # Parse command line parameters
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -g|--global)
                GLOBAL_MODEL="$2"
                shift 2
                ;;
            -l|--local)
                LOCAL_MODEL="$2"
                shift 2
                ;;
            -v|--video)
                VIDEO_PATH="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --pattern)
                PATTERN="$2"
                shift 2
                ;;
            --skip-existing)
                SKIP_EXISTING=true
                shift
                ;;
            --bytetrack)
                USE_ORIGINAL_BYTETRACK=true
                shift
                ;;
            --enhanced-tracker)
                USE_ORIGINAL_BYTETRACK=false
                shift
                ;;
            --detection-mode)
                DETECTION_MODE="$2"
                shift 2
                ;;
            --global-only)
                DETECTION_MODE=0
                shift
                ;;
            --combined-detection)
                DETECTION_MODE=1
                shift
                ;;
            --ros-topics)
                SHOW_TOPICS=true
                shift
                ;;
            --no-ros)
                ENABLE_ROS=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # If only showing topic information, show and exit
    if [[ "$SHOW_TOPICS" == "true" ]]; then
        show_ros_topics
        exit 0
    fi
    
    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"
    
    # Ensure executable file exists
    if [ ! -x "$EXECUTABLE_PATH" ]; then
        echo "❌ Error: executable file does not exist or is not executable: $EXECUTABLE_PATH"
        echo "Please run: cd $SCRIPT_DIR && colcon build"
        exit 1
    fi
    
    # Check ROS2 environment
    if [[ "$ENABLE_ROS" == "true" ]]; then
        if ! command -v ros2 &> /dev/null; then
            echo "❌ Error: ROS2 command not found, please source ROS2 environment"
            echo "Please run: source /opt/ros/foxy/setup.bash"
            exit 1
        fi
        
        # Source ROS2 environment
        source /opt/ros/foxy/setup.bash
        source "$SCRIPT_DIR/install/setup.bash"
        echo "✅ ROS2 environment loaded"
    fi
    
    # 处理视频
    if [ -d "$VIDEO_PATH" ]; then
        echo "🔍 Process video folder: $VIDEO_PATH"
        
        # Get all video files in the directory
        declare -a video_files
        
        # If specified pattern, use pattern matching
        if [[ -n "$PATTERN" ]]; then
            echo "🔍 Use file matching pattern: $PATTERN"
            
            for ext in "${VIDEO_EXTENSIONS[@]}"; do
                while IFS= read -r -d $'\0' file; do
                    video_files+=("$file")
                done < <(find "$VIDEO_PATH" -type f -name "$PATTERN.$ext" -print0 2>/dev/null)
            done
            
            # Also support cases with extensions in the pattern
            while IFS= read -r -d $'\0' file; do
                for ext in "${VIDEO_EXTENSIONS[@]}"; do
                    if [[ "$file" == *".$ext" ]]; then
                        video_files+=("$file")
                        break
                    fi
                done
            done < <(find "$VIDEO_PATH" -type f -name "$PATTERN" -print0 2>/dev/null)
            
        else
            # No pattern, process all supported video files
            for ext in "${VIDEO_EXTENSIONS[@]}"; do
                while IFS= read -r -d $'\0' file; do
                    video_files+=("$file")
                done < <(find "$VIDEO_PATH" -type f -name "*.$ext" -print0 2>/dev/null)
            done
        fi
        
        # Sort file names
        IFS=$'\n' video_files=($(sort <<<"${video_files[*]}"))
        unset IFS
        
        # Process video files
        echo "📋 Found ${#video_files[@]} video files"
        
        if [ ${#video_files[@]} -eq 0 ]; then
            echo "❌ Error: No video files found"
            exit 1
        fi
        
        # Show found video file list
        echo "Video file list:"
        for (( i=0; i<${#video_files[@]}; i++ )); do
            echo "  $((i+1)). $(basename "${video_files[$i]}")"
        done
        echo ""
        
        # Process each video file
        for (( i=0; i<${#video_files[@]}; i++ )); do
            local video_file="${video_files[$i]}"
            echo "Process video $((i+1))/${#video_files[@]}: $(basename "$video_file")"
            process_single_video "$video_file"
        done
        
    elif [ -f "$VIDEO_PATH" ] || [ -c "$VIDEO_PATH" ]; then
        echo "🎥 Process single video source: $VIDEO_PATH"
        process_single_video "$VIDEO_PATH"
    else
        # If not directory/regular file/character device, check if it is index or stream/pipeline string
        if [[ "$VIDEO_PATH" =~ ^[0-9]+$ ]]; then
            echo "🎥 Process camera index: $VIDEO_PATH"
            process_single_video "$VIDEO_PATH"
        elif [[ "$VIDEO_PATH" == rtsp://* || "$VIDEO_PATH" == http://* || "$VIDEO_PATH" == https://* || "$VIDEO_PATH" == v4l2src* || "$VIDEO_PATH" == nvarguscamerasrc* || "$VIDEO_PATH" == rtspsrc* ]]; then
            echo "📡 Process stream/pipeline: $VIDEO_PATH"
            process_single_video "$VIDEO_PATH"
        else
            echo "❌ Error: video path does not exist or is not supported: $VIDEO_PATH"
            exit 1
        fi
    fi
    
    echo "✅ All video processing completed"
    
    # If ROS2 is enabled, show topic information
    if [[ "$ENABLE_ROS" == "true" ]]; then
        echo ""
        echo "🔍 Use the following commands to view ROS2 topics:"
        echo "  ros2 topic list"
        echo "  ros2 topic echo /gldt/tracking_result"
        echo "  ros2 topic hz /gldt/tracking_result"
    fi
}

main "$@"

