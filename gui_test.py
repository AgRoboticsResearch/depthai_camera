#!/usr/bin/env python3
import os
# 禁用此警告
os.environ['NO_AT_BRIDGE'] = '1'

import cv2
import depthai as dai
import numpy as np
import time
import threading
from datetime import datetime

class OAKCameraApp:
    def __init__(self):
        # 窗口名称
        self.window_name = "OAK Camera Interface"
        
        # 控制变量
        self.running = False
        self.pipeline = None
        self.device = None
        self.thread = None
        self.rgb_image = None
        self.depth_image = None
        self.depth_colormap = None
        self.left_image = None
        self.right_image = None
        self.quit_flag = False
        
        # 保存标志
        self.save_rgb_continuous = False
        self.save_depth_continuous = False
        self.save_lr_continuous = False
        
        # 按钮区域定义 - 英文文字
        self.buttons = {
            "start": {"x1": 20, "y1": 20, "x2": 120, "y2": 60, "text": "Start", "color": (100, 200, 100)},
            "stop": {"x1": 140, "y1": 20, "x2": 240, "y2": 60, "text": "Stop", "color": (100, 100, 200)},
            "save_rgb": {"x1": 260, "y1": 20, "x2": 380, "y2": 60, "text": "Save RGB", "color": (100, 200, 200)},
            "save_depth": {"x1": 400, "y1": 20, "x2": 520, "y2": 60, "text": "Save Depth", "color": (200, 100, 200)},
            "save_lr": {"x1": 540, "y1": 20, "x2": 660, "y2": 60, "text": "Save L+R", "color": (150, 200, 150)},
            "quit": {"x1": 680, "y1": 20, "x2": 780, "y2": 60, "text": "Quit", "color": (200, 100, 100)}
        }
        
        # 保存路径
        self.save_path = "./camera_captures"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # 创建子文件夹
        self.rgb_folder = os.path.join(self.save_path, "rgb")
        self.depth_folder = os.path.join(self.save_path, "depth")
        self.depth_color_folder = os.path.join(self.save_path, "depth_color")
        self.left_folder = os.path.join(self.save_path, "left")
        self.right_folder = os.path.join(self.save_path, "right")
        
        # 确保子文件夹存在
        for folder in [self.rgb_folder, self.depth_folder, self.depth_color_folder, 
                       self.left_folder, self.right_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # 状态消息
        self.status_message = ""
        self.status_time = 0
        
        # 显示模式，默认显示RGB和深度
        self.display_mode = "rgb_depth"  # 可选: "rgb_depth", "left_right", "rgb_only", "depth_only"
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # 设置鼠标回调
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 创建初始界面
        self.create_ui()
    
    def mouse_callback(self, event, x, y, flags, param):
        # 只处理左键点击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at: ({x}, {y})")
            # 检查是否点击了任何按钮
            for button_name, button in self.buttons.items():
                if button["x1"] <= x <= button["x2"] and button["y1"] <= y <= button["y2"]:
                    print(f"Button clicked: {button_name}")
                    # 执行对应的操作
                    if button_name == "start":
                        self.start_camera()
                    elif button_name == "stop":
                        self.stop_camera()
                    elif button_name == "save_rgb":
                        self.toggle_save_rgb()
                    elif button_name == "save_depth":
                        self.toggle_save_depth()
                    elif button_name == "save_lr":
                        self.toggle_save_lr()
                    elif button_name == "quit":
                        self.quit_flag = True
                    return
    
    def create_ui(self):
        # 创建UI界面 - 窗口尺寸: 1280 x 720
        ui = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        
        # 顶部控制面板区域 (高100像素)
        cv2.rectangle(ui, (0, 0), (1280, 100), (230, 230, 230), -1)
        cv2.line(ui, (0, 100), (1280, 100), (200, 200, 200), 2)
        
        # 绘制按钮
        for button_name, button in self.buttons.items():
            # 绘制按钮背景
            button_color = button["color"]
            # 相机运行状态下按钮灰显逻辑
            if button_name == "start" and self.running:
                button_color = (150, 150, 150)
            if not self.running and button_name in ["stop", "save_rgb", "save_depth", "save_lr"]:
                button_color = (150, 150, 150)
            # 活跃状态按钮高亮
            if button_name == "save_rgb" and self.save_rgb_continuous:
                button_color = (0, 255, 255)
            if button_name == "save_depth" and self.save_depth_continuous:
                button_color = (255, 0, 255)
            if button_name == "save_lr" and self.save_lr_continuous:
                button_color = (150, 255, 150)
                
            cv2.rectangle(ui, (button["x1"], button["y1"]), (button["x2"], button["y2"]), button_color, -1)
            cv2.rectangle(ui, (button["x1"], button["y1"]), (button["x2"], button["y2"]), (50, 50, 50), 1)
            
            # 绘制按钮文字
            text_size = cv2.getTextSize(button["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = button["x1"] + (button["x2"] - button["x1"] - text_size[0]) // 2
            text_y = button["y1"] + (button["y2"] - button["y1"] + text_size[1]) // 2
            cv2.putText(ui, button["text"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 显示保存路径和保存状态
        cv2.putText(ui, f"Save Path: {self.save_path}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 1)
        
        # 显示保存状态指示
        status_y = 90
        if self.save_rgb_continuous:
            cv2.putText(ui, "RGB Saving: Active", (300, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 128), 1)
        if self.save_depth_continuous:
            cv2.putText(ui, "Depth Saving: Active", (450, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
        if self.save_lr_continuous:
            cv2.putText(ui, "L+R Saving: Active", (600, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
        
        # 显示状态消息
        if self.status_message and time.time() - self.status_time < 5:
            cv2.putText(ui, self.status_message, (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        
        # 四分屏布局 - 显示所有四个相机视图
        # 左上: RGB, 右上: 深度图, 左下: 左相机, 右下: 右相机
        
        # 绘制分隔线
        cv2.line(ui, (640, 100), (640, 720), (200, 200, 200), 2)  # 垂直分隔线
        cv2.line(ui, (0, 410), (1280, 410), (200, 200, 200), 2)   # 水平分隔线
        
        # 添加标签
        cv2.putText(ui, "RGB Image", (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(ui, "Depth Image", (900, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(ui, "Left Camera", (300, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(ui, "Right Camera", (900, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 图像尺寸调整
        img_width, img_height = 620, 280
        
        # 显示RGB图像 (左上)
        if self.rgb_image is not None:
            rgb_resized = cv2.resize(self.rgb_image, (img_width, img_height))
            ui[130:130+img_height, 10:10+img_width] = rgb_resized
        else:
            cv2.rectangle(ui, (10, 130), (10+img_width, 130+img_height), (200, 200, 200), 2)
            cv2.putText(ui, "RGB Not Available", (260, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 显示深度图像 (右上)
        if self.depth_colormap is not None:
            depth_resized = cv2.resize(self.depth_colormap, (img_width, img_height))
            ui[130:130+img_height, 650:650+img_width] = depth_resized
        else:
            cv2.rectangle(ui, (650, 130), (650+img_width, 130+img_height), (200, 200, 200), 2)
            cv2.putText(ui, "Depth Not Available", (870, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 显示左相机图像 (左下)
        if self.left_image is not None:
            left_resized = cv2.resize(self.left_image, (img_width, img_height))
            ui[440:440+img_height, 10:10+img_width] = cv2.cvtColor(left_resized, cv2.COLOR_GRAY2BGR)
        else:
            cv2.rectangle(ui, (10, 440), (10+img_width, 440+img_height), (200, 200, 200), 2)
            cv2.putText(ui, "Left Camera Not Available", (240, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 显示右相机图像 (右下)
        if self.right_image is not None:
            right_resized = cv2.resize(self.right_image, (img_width, img_height))
            ui[440:440+img_height, 650:650+img_width] = cv2.cvtColor(right_resized, cv2.COLOR_GRAY2BGR)
        else:
            cv2.rectangle(ui, (650, 440), (650+img_width, 440+img_height), (200, 200, 200), 2)
            cv2.putText(ui, "Right Camera Not Available", (870, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 显示UI
        cv2.imshow(self.window_name, ui)
    
    def show_status(self, message):
        """显示状态消息"""
        self.status_message = message
        self.status_time = time.time()
        print(message)  # 控制台打印消息
        self.create_ui()  # 刷新UI以显示状态
    
    def create_pipeline(self):
        """创建DepthAI管线"""
        pipeline = dai.Pipeline()
        
        # 定义RGB相机
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setFps(30)
        camRgb.setInterleaved(False)
        
        # 定义左右单目相机
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setFps(30)
        monoRight.setFps(30)
        
        # 定义立体深度节点
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        
        # 定义输出
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        
        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        
        xoutLeft = pipeline.create(dai.node.XLinkOut)
        xoutLeft.setStreamName("left")
        
        xoutRight = pipeline.create(dai.node.XLinkOut)
        xoutRight.setStreamName("right")
        
        # 连接节点
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        camRgb.preview.link(xoutRgb.input)
        stereo.depth.link(xoutDepth.input)
        
        # 连接左右单目相机输出
        monoLeft.out.link(xoutLeft.input)
        monoRight.out.link(xoutRight.input)
        
        return pipeline
    
    def start_camera(self):
        """启动相机"""
        if self.running:
            self.show_status("Camera already running")
            return
        
        try:
            self.pipeline = self.create_pipeline()
            self.device = dai.Device(self.pipeline)
            
            # 获取输出队列
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            self.left_queue = self.device.getOutputQueue(name="left", maxSize=4, blocking=False)
            self.right_queue = self.device.getOutputQueue(name="right", maxSize=4, blocking=False)
            
            self.running = True
            
            # 显示状态消息
            self.show_status("Camera started successfully")
            
            # 在单独的线程中处理图像
            self.thread = threading.Thread(target=self.process_frames)
            self.thread.daemon = True
            self.thread.start()
            
        except Exception as e:
            self.show_status(f"Error: {str(e)}")
            print(f"Camera start error: {str(e)}")
    
    def stop_camera(self):
        """停止相机"""
        if not self.running:
            self.show_status("Camera not running")
            return
        
        # 同时停止保存图像
        self.save_rgb_continuous = False
        self.save_depth_continuous = False
        self.save_lr_continuous = False
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.device:
            try:
                self.device.close()
            except:
                pass
            self.device = None
        
        self.show_status("Camera stopped")
    
    def process_frames(self):
        """处理从相机获取的帧"""
        # 用于控制连续保存频率的变量
        last_save_time = 0
        save_interval = 0.5  # 每0.5秒保存一次
        
        while self.running:
            try:
                # 获取RGB图像
                rgb_in = self.rgb_queue.tryGet()
                if rgb_in is not None:
                    rgb_frame = rgb_in.getCvFrame()
                    self.rgb_image = rgb_frame.copy()
                
                # 获取深度图像
                depth_in = self.depth_queue.tryGet()
                if depth_in is not None:
                    depth_frame = depth_in.getFrame()
                    self.depth_image = depth_frame.copy()
                    # 创建彩色深度图
                    self.depth_colormap = self.normalize_depth(depth_frame)
                
                # 获取左单目图像
                left_in = self.left_queue.tryGet()
                if left_in is not None:
                    left_frame = left_in.getCvFrame()
                    self.left_image = left_frame.copy()
                
                # 获取右单目图像
                right_in = self.right_queue.tryGet()
                if right_in is not None:
                    right_frame = right_in.getCvFrame()
                    self.right_image = right_frame.copy()
                
                # 连续保存图像
                current_time = time.time()
                if (self.save_rgb_continuous or self.save_depth_continuous or 
                    self.save_lr_continuous) and \
                   current_time - last_save_time >= save_interval:
                    self.save_images_continuous()
                    last_save_time = current_time
                
                # 更新UI
                self.create_ui()
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                
            # 短暂睡眠，减少CPU使用
            time.sleep(0.03)
    
    def normalize_depth(self, depth_frame):
        """归一化深度图像并应用伪彩色映射"""
        try:
            # 缩小图像尺寸以加快处理速度
            depth_downscaled = depth_frame[::2, ::2]
            non_zero_pixels = depth_downscaled[depth_downscaled != 0]
            
            if non_zero_pixels.size > 0:
                min_depth = np.percentile(non_zero_pixels, 1)
                max_depth = np.percentile(non_zero_pixels, 99)
            else:
                min_depth = 0
                max_depth = 1
                
            # 归一化到0-255的范围
            depth_normalized = np.clip((depth_frame - min_depth) / (max_depth - min_depth) * 255.0, 0, 255).astype(np.uint8)
            
            # 应用伪彩色映射
            depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # 将无效深度点设为黑色
            depth_mask = depth_frame == 0
            depth_colorized[depth_mask] = [0, 0, 0]
            
            return depth_colorized
            
        except Exception as e:
            print(f"Depth normalization error: {e}")
            # 返回黑色图像作为备选
            return np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)
    
    def toggle_save_rgb(self):
        """切换RGB图像连续保存状态"""
        if not self.running:
            self.show_status("Camera not running - cannot save images")
            return
            
        self.save_rgb_continuous = not self.save_rgb_continuous
        if self.save_rgb_continuous:
            self.show_status("RGB continuous saving started")
        else:
            self.show_status("RGB continuous saving stopped")
    
    def toggle_save_depth(self):
        """切换深度图像连续保存状态"""
        if not self.running:
            self.show_status("Camera not running - cannot save images")
            return
            
        self.save_depth_continuous = not self.save_depth_continuous
        if self.save_depth_continuous:
            self.show_status("Depth continuous saving started")
        else:
            self.show_status("Depth continuous saving stopped")
    
    def toggle_save_lr(self):
        """切换左右单目图像同时连续保存状态"""
        if not self.running:
            self.show_status("Camera not running - cannot save images")
            return
            
        self.save_lr_continuous = not self.save_lr_continuous
        if self.save_lr_continuous:
            self.show_status("Left+Right continuous saving started")
        else:
            self.show_status("Left+Right continuous saving stopped")
    
    def save_images_continuous(self):
        """连续保存图像"""
        if not self.running:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            
            # 保存RGB图像
            if self.save_rgb_continuous and self.rgb_image is not None:
                rgb_path = os.path.join(self.rgb_folder, f"rgb_{timestamp}.png")
                cv2.imwrite(rgb_path, self.rgb_image)
            
            # 保存深度图像（原始数据和彩色图）
            if self.save_depth_continuous and self.depth_image is not None:
                depth_path = os.path.join(self.depth_folder, f"depth_{timestamp}.png")
                # 将16位深度图转换为可保存格式
                depth_image_uint16 = self.depth_image.astype(np.uint16)
                cv2.imwrite(depth_path, depth_image_uint16)
                
                if self.depth_colormap is not None:
                    depth_color_path = os.path.join(self.depth_color_folder, f"depth_color_{timestamp}.png")
                    cv2.imwrite(depth_color_path, self.depth_colormap)
            
            # 保存左右单目图像（同时）
            if self.save_lr_continuous:
                # 保存左单目图像
                if self.left_image is not None:
                    left_path = os.path.join(self.left_folder, f"left_{timestamp}.png")
                    cv2.imwrite(left_path, self.left_image)
                
                # 保存右单目图像 
                if self.right_image is not None:
                    right_path = os.path.join(self.right_folder, f"right_{timestamp}.png")
                    cv2.imwrite(right_path, self.right_image)
            
        except Exception as e:
            # 出错时停止所有连续保存
            self.save_rgb_continuous = False
            self.save_depth_continuous = False
            self.save_lr_continuous = False
            self.show_status(f"Save error: {str(e)}")
            print(f"Error saving images: {str(e)}")
    
    def run(self):
        """主循环"""
        print("Application started. Press 'ESC' or 'q' to exit")
        
        while not self.quit_flag:
            # 更新UI - 如果自动更新失败
            self.create_ui()
            
            # 等待键盘事件
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'):  # ESC或q键退出
                break
            elif key == ord('s'):  # s键启动相机
                self.start_camera()
            elif key == ord('p'):  # p键停止相机
                self.stop_camera()
            elif key == ord('r'):  # r键保存RGB
                self.toggle_save_rgb()
            elif key == ord('d'):  # d键保存深度
                self.toggle_save_depth()
            elif key == ord('l'):  # l键保存左右图像
                self.toggle_save_lr()
            elif key == ord('v'):  # v键切换视图
                self.display_mode = "left_right" if self.display_mode == "rgb_depth" else "rgb_depth"
                self.show_status(f"Switched to {'Left+Right' if self.display_mode=='left_right' else 'RGB+Depth'} view")
        
        # 退出前清理
        self.quit_flag = True
        if self.running:
            self.stop_camera()
        
        cv2.destroyAllWindows()

def main():
    app = OAKCameraApp()
    try:
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("Application exited")

if __name__ == "__main__":
    main()