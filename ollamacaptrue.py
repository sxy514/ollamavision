import base64
import cv2
import requests
import numpy as np
import threading
import queue
import time
import argparse
from typing import Optional, Tuple, Union

class VideoProcessor:
    def __init__(self, frame_interval: int = 30, show_video: bool = True):
        self.frame_interval = frame_interval
        self.show_video = show_video
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)  # 限制队列大小，避免内存占用过高
        self.result_queue = queue.Queue()
        self.latest_result = ""

    def capture_frames(self, video_source: Union[str, int]):
        """视频捕获线程"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return

        frame_count = 0
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, int):  # 如果是摄像头，继续尝试
                        continue
                    break

                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 按间隔处理帧
                if frame_count % self.frame_interval == 0:
                    if not self.frame_queue.full():  # 如果队列未满，添加帧
                        try:
                            self.frame_queue.put(rgb_frame, block=False)
                        except queue.Full:
                            pass

                # 显示视频
                if self.show_video:
                    cv2.imshow('Video Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                frame_count += 1
                time.sleep(0.03)  # 控制帧率，约30fps

        finally:
            cap.release()
            if self.show_video:
                cv2.destroyAllWindows()

    def process_frames(self):
        """AI处理线程"""
        while self.running:
            try:
                # 从队列中获取帧，非阻塞方式
                frame = self.frame_queue.get(timeout=1)
                
                # 处理帧
                description = self.analyze_frame(frame)
                if description:
                    self.latest_result = description
                    print(f"\n分析结果: {description}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理帧时出错: {str(e)}")
                time.sleep(1)

    def analyze_frame(self, frame: np.ndarray) -> Optional[str]:
        """使用Ollama分析单帧"""
        try:
            # 调整图像大小以加快处理速度
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # 编码为JPEG
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "model": "qwen2.5vl",
                "prompt": "分析这个视频帧中的内容。在画面中找到耳机,然后回复我找到了耳机",
                "images": [base64_image],
                "stream": False
            }
            
            # 设置较短的超时时间
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            return response.json().get("response")
            
        except Exception as e:
            print(f"分析帧时出错: {str(e)}")
            return None

    def run(self, video_source: Union[str, int]):
        """启动视频处理"""
        self.running = True
        
        # 启动捕获线程
        capture_thread = threading.Thread(
            target=self.capture_frames,
            args=(video_source,),
            daemon=True
        )
        
        # 启动处理线程
        process_thread = threading.Thread(
            target=self.process_frames,
            daemon=True
        )
        
        capture_thread.start()
        process_thread.start()
        
        try:
            # 主线程保持运行
            while self.running:
                time.sleep(0.1)
                # 可以在这里添加其他UI更新逻辑
                
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            self.running = False
            capture_thread.join(timeout=1)
            process_thread.join(timeout=1)
            if self.show_video:
                cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='实时视频分析工具')
    parser.add_argument('--source', type=str, default='0',
                       help='视频文件路径或摄像头索引 (默认: 0)')
    parser.add_argument('--interval', type=int, default=30,
                       help='处理帧的间隔 (默认: 30)')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示视频窗口')
    
    args = parser.parse_args()
    
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    processor = VideoProcessor(
        frame_interval=args.interval,
        show_video=not args.no_display
    )
    
    print("开始处理视频，按 'q' 键退出...")
    processor.run(video_source)

if __name__ == "__main__":
    main()