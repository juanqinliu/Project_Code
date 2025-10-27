import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from yolo_msgs.msg import YoloInferenceMsg, YoloInference, BoundingBox, Mask, Point2D

import cv2
import numpy as np
import signal
import threading
import time
import os
from datetime import datetime
from collections import deque

class YoloNode(Node):
    
    def __init__(self):
        super().__init__('detection_node')
        
        self._class_to_color = {}
        self._shutdown_requested = False
        
        # 声明和获取参数
        self._setup_parameters()
        
        # 初始化组件
        self.cv_bridge = CvBridge()
        self.model = YOLO(self.weight)
        
        # 创建发布者
        self.pub = self.create_publisher(YoloInferenceMsg, '/yolo_inference', 1)
        self.img_pub = self.create_publisher(Image, '/yolo_result', 1)
        
        # 初始化视频处理
        self.cap = None
        self.camera_thread = None
        self.video_writer = None
        self.save_thread = None
        self.frame_queue = deque(maxlen=self.max_queue_size)
        self.save_lock = threading.Lock()
        
        self.initialize_video_saving()
        self.initialize_capture()
        
        self.get_logger().info('YOLO detection node started')
    
    def _setup_parameters(self):
        """设置所有参数"""
        # 模型参数
        param_configs = {
            'weight': ('yolon.pt', 'string'),
            'device': ('cuda:0', 'string'),
            'conf_threshold': (0.5, 'double'),
            'iou_threshold': (0.7, 'double'),
            'input_type': ('camera', 'string'),
            'input_path': ('', 'string'),
            'camera_id': (0, 'integer'),
            'width': (1920, 'integer'),
            'height': (1080, 'integer'),
            'fps': (30, 'integer'),
            'display_width': (640, 'integer'),
            'display_height': (480, 'integer'),
            'show_result': (True, 'bool'),
            'save_video': (False, 'bool'),
            'save_original_video': (False, 'bool'),
            'save_path': ('results', 'string'),
            'max_queue_size': (100, 'integer'),
            'save_result': (True, 'bool')
        }
        
        for param_name, (default_value, param_type) in param_configs.items():
            self.declare_parameter(param_name, default_value)
            param_value = self.get_parameter(param_name).get_parameter_value()
            
            if param_type == 'string':
                setattr(self, param_name, param_value.string_value)
            elif param_type == 'double':
                setattr(self, param_name, param_value.double_value)
            elif param_type == 'integer':
                setattr(self, param_name, param_value.integer_value)
            elif param_type == 'bool':
                setattr(self, param_name, param_value.bool_value)
    
    def initialize_video_saving(self):
        """初始化视频保存功能"""
        if not hasattr(self, 'save_result'):
            self.save_result = False
            
        if not (self.save_video or self.save_original_video or self.save_result):
            self.get_logger().info('Video and result saving disabled')
            return
            
        try:
            os.makedirs(self.save_path, exist_ok=True)
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.input_path))[0] if self.input_type != 'camera' else 'camera'
            
            # 初始化检测结果视频写入器
            if self.save_video:
                self.output_video_path = os.path.join(self.save_path, f"yolo_{base_name}_{timestamp}_detection.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path, fourcc, self.fps, (self.width, self.height)
                )
                
            # 初始化原始视频写入器
            if self.save_original_video:
                self.original_video_path = os.path.join(self.save_path, f"yolo_{base_name}_{timestamp}_original.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.original_video_writer = cv2.VideoWriter(
                    self.original_video_path, fourcc, self.fps, (self.width, self.height)
                )
                
            if self.save_result:
                # 创建检测结果文件
                self.result_file_path = os.path.join(self.save_path, f"yolo_{base_name}_{timestamp}_results.txt")
                self.frame_count = 0  # 添加帧计数器
                # 创建或清空结果文件
                with open(self.result_file_path, 'w') as f:
                    f.write("# Format: frame_id class_name confidence x_center y_center width height\n")
            
            if not self.video_writer.isOpened():
                self.get_logger().error('Cannot create video writer')
                self.save_video = False
                return
            
            # 启动保存线程
            self.save_thread = threading.Thread(target=self.video_save_loop, daemon=True)
            self.save_thread.start()
            
            self.get_logger().info(f'Video saving enabled, output path: {self.output_video_path}')
            
        except Exception as e:
            self.get_logger().error(f'Error initializing video saving: {str(e)}')
            self.save_video = False
    
    def video_save_loop(self):
        """视频保存线程循环"""
        while not self._shutdown_requested and self.save_video:
            try:
                if self.frame_queue:
                    with self.save_lock:
                        frame = self.frame_queue.popleft()
                        if frame is not None and self.video_writer is not None:
                            self.video_writer.write(frame)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f'Error in video saving thread: {str(e)}')
                time.sleep(0.1)
    
    def add_frame_to_queue(self, frame):
        """将帧添加到保存队列"""
        if not (self.save_video and self.video_writer is not None):
            return
            
        try:
            with self.save_lock:
                if len(self.frame_queue) < self.max_queue_size:
                    self.frame_queue.append(frame.copy())
                else:
                    self.get_logger().warn('Frame queue is full, discard current frame')
        except Exception as e:
            self.get_logger().error(f'Error adding frame to queue: {str(e)}')
    
    def initialize_capture(self):
        """初始化视频捕获（相机或视频文件）"""
        try:
            if self.input_type == 'camera':
                self._setup_camera()
            elif self.input_type == 'video':
                self._setup_video()
            elif self.input_type == 'image':
                self._process_single_image()
                return
            else:
                self.get_logger().error(f'Unsupported input type: {self.input_type}')
                return
            
            # 启动视频处理线程
            self.camera_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.camera_thread.start()
            self.get_logger().info('Video processing thread started')
            
        except Exception as e:
            self.get_logger().error(f'Error initializing video capture: {str(e)}')
    
    def _setup_camera(self):
        """设置相机"""
        self.get_logger().info(f'Initializing camera mode, camera ID: {self.camera_id}')
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera ID: {self.camera_id}')
            return
        
        # 设置相机参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 读取实际生效的分辨率
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def _setup_video(self):
        """设置视频文件"""
        if not os.path.exists(self.input_path):
            self.get_logger().error(f'Video file does not exist: {self.input_path}')
            return
        
        self.get_logger().info(f'Initializing video mode, file path: {self.input_path}')
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open video file: {self.input_path}')
            return
        
        # 获取视频属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or self.fps
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.get_logger().info(f'Video information: {self.width}x{self.height}, {self.fps}fps, total {total_frames} frames')
    
    def _process_single_image(self):
        """处理单张图像"""
        if not os.path.exists(self.input_path):
            self.get_logger().error(f'Image file does not exist: {self.input_path}')
            return
        
        self.get_logger().info(f'Initializing single image mode, file path: {self.input_path}')
        img = cv2.imread(self.input_path)
        if img is None:
            self.get_logger().error(f'Cannot read image file: {self.input_path}')
            return
        
        # 使用图像原始分辨率
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.process_frame(img)
        self.get_logger().info('Single image processing completed')
    
    def capture_loop(self):
        """Video capture and processing loop"""
        frame_time = 1.0 / self.fps
        
        while not self._shutdown_requested and rclpy.ok():
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                if self.input_type == 'video':
                    self.get_logger().info('Video playback finished, restarting...')
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.get_logger().warn('Cannot read frame from camera')
                    time.sleep(0.1)
                    continue
            
            self.process_frame(frame)
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def process_frame(self, img):
        """Deal with image frame: keep input original size for inference and save, only used for display"""
        # 首先保存原始帧（如果需要）
        if hasattr(self, 'save_original_video') and self.save_original_video and hasattr(self, 'original_video_writer'):
            self.original_video_writer.write(img.copy())

        # YOLO推理
        results = self.model.predict(
            source=img, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device
        )[0].cpu()
        
        self.get_logger().info(f'YOLO prediction completed. Found {len(results.boxes) if results.boxes else 0} objects')
        
        # 创建推理消息
        inference_msg = self._create_inference_message()
        
        # 处理检测结果
        if not results.boxes or (hasattr(results.boxes, 'shape') and results.boxes.shape[0] == 0):
            self._handle_no_detection(img)
            return
        
        # 处理有检测结果的情况
        detections = self._process_detections(img, results, inference_msg)
        
        # 处理检测结果（绘制检测框等）
        detections = self._process_detections(img, results, inference_msg)
        
        # 保存检测结果到txt文件
        if hasattr(self, 'save_result') and self.save_result:
            self._save_detection_results(detections)
            
        # 保存检测结果视频（带检测框）
        if hasattr(self, 'save_video') and self.save_video and hasattr(self, 'video_writer'):
            self.video_writer.write(img.copy())
        
        # 发布结果
        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(img, 'bgr8'))
        self.pub.publish(inference_msg)
        
        # 显示结果
        self._display_result(img)
    
    def _create_inference_message(self):
        """创建推理消息"""
        inference_msg = YoloInferenceMsg()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera"
        inference_msg.header = header
        return inference_msg
    
    def _handle_no_detection(self, img):
        """处理无检测结果的情况"""
        self.get_logger().info('No objects detected')
        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(img, 'bgr8'))
        
        # 保存检测结果视频（即使没有检测到目标）
        if hasattr(self, 'save_video') and self.save_video and hasattr(self, 'video_writer'):
            self.video_writer.write(img.copy())
            
        self._display_result(img)
    
    def _process_detections(self, img, results, inference_msg):
        """处理检测结果"""
        detections = []
        for i in range(len(results.boxes)):
            inference_result = self._create_detection_result(results, i)
            
            if results.boxes:
                self._draw_bounding_box(img, inference_result)
            
            if results.masks:
                self._draw_mask(img, results, i, inference_result)
            
            inference_msg.yolo_inference.append(inference_result)
            detections.append(inference_result)
        
        return detections
    
    def _create_detection_result(self, results, i):
        """创建检测结果"""
        inference_result = YoloInference()
        bbox_info = results.boxes[i]
        
        # 处理类别和置信度
        try:
            cls_id = int(bbox_info.cls)
            score = float(bbox_info.conf)
        except Exception:
            cls_id = int(bbox_info.cls[0])
            score = float(bbox_info.conf[0])
        
        inference_result.class_id = cls_id
        inference_result.class_name = self.model.names[cls_id]
        inference_result.score = score
        
        # 创建边界框消息
        bbox_msg = BoundingBox()
        bbox = bbox_info.xywh[0]
        bbox_msg.center.x = float(bbox[0])
        bbox_msg.center.y = float(bbox[1])
        bbox_msg.size.x = float(bbox[2])
        bbox_msg.size.y = float(bbox[3])
        inference_result.bbox = bbox_msg
        
        return inference_result
    
    def _draw_bounding_box(self, img, inference_result):
        """绘制边界框"""
        # 获取颜色
        label = inference_result.class_name
        if label not in self._class_to_color:
            import random
            self._class_to_color[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        color = self._class_to_color[label]
        
        # 计算边界框坐标
        bbox = inference_result.bbox
        pt1 = (round(bbox.center.x - bbox.size.x / 2.0), round(bbox.center.y - bbox.size.y / 2.0))
        pt2 = (round(bbox.center.x + bbox.size.x / 2.0), round(bbox.center.y + bbox.size.y / 2.0))
        
        # 绘制边界框和标签
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img, str(inference_result.class_name), 
                   ((pt1[0]+pt2[0])//2-5, pt1[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    
    def _draw_mask(self, img, results, i, inference_result):
        """绘制掩码"""
        mask_info = results.masks[i]
        mask_msg = Mask()
        mask_msg.data = [Point2D(x=float(point[0]), y=float(point[1])) for point in mask_info.xy[0].tolist()]
        mask_msg.height = results.orig_img.shape[0]
        mask_msg.width = results.orig_img.shape[1]
        inference_result.mask = mask_msg
        
        # 获取颜色
        label = inference_result.class_name
        color = self._class_to_color.get(label, (255, 255, 255))
        
        # 绘制掩码
        mask_array = np.array([[int(point.x), int(point.y)] for point in mask_msg.data])
        temp = img.copy()
        temp = cv2.fillPoly(temp, [mask_array], color)
        cv2.addWeighted(img, 0.4, temp, 0.6, 0, img)
        cv2.polylines(img, [mask_array], True, color, 2)
    
    def _display_result(self, img):
        """显示结果"""
        if self._shutdown_requested:
            return
            
        if not hasattr(self, 'show_result') or not self.show_result:
            return
            
        disp = cv2.resize(img, (self.display_width, self.display_height))
        cv2.imshow("YOLO Result", disp)
        cv2.waitKey(1)  # 重要：需要这一行来实际显示窗口和处理窗口事件
        
        if self.input_type == 'image':
            cv2.waitKey(0)  # 如果是图片模式，等待按键
        else:
            pass

    def on_shutdown(self):
        """处理节点关闭时的清理工作"""
        self.get_logger().info('Shutting down YOLO detection node...')
        self._shutdown_requested = True
        
        # 关闭视频捕获
        if self.cap is not None:
            self.cap.release()
            self.get_logger().info('Video capture closed')
        
        # 等待线程结束
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=1.0)
            self.get_logger().info('Video processing thread closed')
        
        # 关闭视频保存
        self._cleanup_video_saving()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 释放模型资源
        self._cleanup_model_resources()
        
        self.get_logger().info('YOLO detection node shutdown complete')
    
    def _save_detection_results(self, detections):
        """保存检测结果到txt文件"""
        if not hasattr(self, 'result_file_path'):
            return
            
        with open(self.result_file_path, 'a') as f:
            for detection in detections:
                # 格式：frame_id class_name confidence x_center y_center width height
                bbox = detection.bbox
                line = f"{self.frame_count:06d} {detection.class_name} {detection.score:.6f} {bbox.center.x:.1f} {bbox.center.y:.1f} {bbox.size.x:.1f} {bbox.size.y:.1f}\n"
                f.write(line)
        self.frame_count += 1
        
    def _cleanup_video_saving(self):
        """清理视频保存资源"""
        # 清理检测结果视频
        if hasattr(self, 'save_video') and self.save_video and hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f'Detection video saved to: {self.output_video_path}')
            
        # 清理原始视频
        if hasattr(self, 'save_original_video') and self.save_original_video and hasattr(self, 'original_video_writer') and self.original_video_writer is not None:
            self.original_video_writer.release()
            self.get_logger().info(f'Original video saved to: {self.original_video_path}')
    
    def _cleanup_model_resources(self):
        """清理模型资源"""
        if not hasattr(self, 'model'):
            return
        
        try:
            del self.model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.get_logger().info('YOLO model resources released')
        except Exception as e:
            self.get_logger().error(f'Error releasing YOLO model: {str(e)}')

def main():
    rclpy.init(args=None)
    node = YoloNode()
    
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        node.get_logger().info('Received shutdown signal')
        node.on_shutdown()
        running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if running:
            node.get_logger().info('Keyboard interrupt detected')
            node.on_shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()