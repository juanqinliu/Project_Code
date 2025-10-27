from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """生成启动描述"""
    # 定义所有配置变量
    configs = {
        'weight': LaunchConfiguration('weight'),
        'device': LaunchConfiguration('device'),
        'conf_threshold': LaunchConfiguration('conf_threshold'),
        'iou_threshold': LaunchConfiguration('iou_threshold'),
        'input_type': LaunchConfiguration('input_type'),
        'input_path': LaunchConfiguration('input_path'),
        'camera_id': LaunchConfiguration('camera_id'),
        'width': LaunchConfiguration('width'),
        'height': LaunchConfiguration('height'),
        'fps': LaunchConfiguration('fps'),
        'namespace': LaunchConfiguration('namespace'),
        'display_width': LaunchConfiguration('display_width'),
        'display_height': LaunchConfiguration('display_height'),
        'show_result': LaunchConfiguration('show_result'),
        'save_video': LaunchConfiguration('save_video'),
        'save_original_video': LaunchConfiguration('save_original_video'),
        'save_path': LaunchConfiguration('save_path'),
        'max_queue_size': LaunchConfiguration('max_queue_size'),
        'save_result': LaunchConfiguration('save_result')
    }

    # 定义所有启动参数
    launch_args = [
        # 模型参数
        ('weight', 'weights/yolo11s-p2.engine', 'weight model path or name'),
        ('device', 'cuda:0', 'Device type(GPU|CPU)'),
        ('conf_threshold', '0.5', 'NMS confidence threshold'),
        ('iou_threshold', '0.7', 'NMS IoU threshold'),
        
        # 输入参数
        ('input_type', 'camera', 'Input type: camera, video, image'),
        ('input_path', '', 'Path to video or image file (only used for video or image mode)'),
        ('camera_id', '0', 'Camera device ID, usually 0 or 1 (only used for camera mode)'),
        ('width', '640', 'Camera width (pixels)'),
        ('height', '480', 'Camera height (pixels)'),
        ('fps', '30', 'Frame rate (camera mode) or target frame rate (video mode, 0 means original frame rate)'),
        ('namespace', 'yolo', 'Node namespace'),
        
        # 显示参数
        ('display_width', '640', 'Display window width (pixels)'),
        ('display_height', '480', 'Display window height (pixels)'),
        ('show_result', 'true', 'Show detection result window (true/false)'),
        
        # 视频和结果保存参数
        ('save_video', 'false', 'Enable detection video saving (true/false)'),
        ('save_original_video', 'false', 'Save original input video without detection boxes (true/false)'),
        ('save_result', 'false', 'Enable detection results saving (true/false)'),
        ('save_path', 'results', 'Path to save output files'),
        ('max_queue_size', '100', 'Maximum frame queue size for video saving')
    ]
    
    # 创建启动参数声明
    launch_arguments = [DeclareLaunchArgument(name, default_value=default, description=desc) 
                       for name, default, desc in launch_args]
    # 使用说明
    usage_info = LogInfo(
        msg="\nUsage:\n" +
            "- Camera mode: ros2 launch yolo_detection yolo_detection.launch.py input_type:=camera camera_id:=0\n" +
            "- Video mode: ros2 launch yolo_detection yolo_detection.launch.py input_type:=video input_path:=/path/to/video.mp4\n" +
            "- Image mode: ros2 launch yolo_detection yolo_detection.launch.py input_type:=image input_path:=/path/to/image.jpg\n" +
            "- With video saving: ros2 launch yolo_detection yolo_detection.launch.py save_video:=true save_path:=/path/to/save\n" +
            "- Note: Video saving will preserve ALL detected frames at original input frame rate\n"
    )
    
    # 直接从源码目录读取参数文件
    src_dir = '/home/ljq/ros2_ws/src'  # 源码目录的绝对路径
    params_file = os.path.join(src_dir, 'yolo_detection', 'config', 'default_params.yaml')
    
    # YOLO检测节点
    node_params = [{name: configs[name]} for name, _, _ in launch_args]
    detection_node = Node(
        package='yolo_detection',
        executable='detection_node',
        name='detection_node',
        parameters=[params_file] + node_params,
        output='screen'
    )
    
    return LaunchDescription(launch_arguments + [usage_info, detection_node])
