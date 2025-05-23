import base64
import cv2
import requests
import numpy as np
import threading
import queue
import time
import argparse
import os
import json
from typing import Optional, Tuple, Union

class VideoProcessor:
    def __init__(self, frame_interval: int = 30, show_video: bool = True, 
                 api_key: str = None, model: str = "qwen/qwen2.5-vl-72b-instruct:free"):
        self.frame_interval = frame_interval
        self.show_video = show_video
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.api_key = api_key
        self.model = model
        self.latest_result = ""
        self.base_url = "http://localhost:11434/v1"  # OpenRouter API端点

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
                    if isinstance(video_source, int):
                        continue
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame_count % self.frame_interval == 0:
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put((rgb_frame, frame_count), block=False)
                        except queue.Full:
                            pass

                if self.show_video:
                    cv2.imshow('Video Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                frame_count += 1
                time.sleep(0.03)

        finally:
            cap.release()
            if self.show_video:
                cv2.destroyAllWindows()

    def process_frames(self):
        """AI处理线程"""
        while self.running:
            try:
                frame, frame_count = self.frame_queue.get(timeout=1)
                description = self.analyze_frame(frame, frame_count)
                if description:
                    self.latest_result = description
                    print(f"\n分析结果: {description}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理帧时出错: {str(e)}")
                time.sleep(1)

    def analyze_frame(self, frame: np.ndarray, frame_count: int) -> Optional[str]:
        """使用OpenRouter API分析单帧"""
        if not self.api_key:
            print("错误：未提供API密钥")
            return None

        try:
            # 调整图像大小
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # 编码为base64
            _, buffer = cv2.imencode('.jpg', small_frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://your-site-url.com",  # 替换为你的网站
                "X-Title": "Video Analysis App",  # 应用名称
                "Content-Type": "application/json"
            }
            
            # 准备请求体
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请简要描述这个视频帧中的内容。"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # 检查响应
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"分析帧时出错: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            return None

    def run(self, video_source: Union[str, int]):
        """启动视频处理"""
        self.running = True
        
        capture_thread = threading.Thread(
            target=self.capture_frames,
            args=(video_source,),
            daemon=True
        )
        
        process_thread = threading.Thread(
            target=self.process_frames,
            daemon=True
        )
        
        capture_thread.start()
        process_thread.start()
        
        try:
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            self.running = False
            capture_thread.join(timeout=1)
            process_thread.join(timeout=1)
            if self.show_video:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='实时视频分析工具 (使用OpenRouter)')
    parser.add_argument('--source', type=str, default='0',
                      help='视频文件路径或摄像头索引 (默认: 0)')
    parser.add_argument('--interval', type=int, default=30,
                      help='处理帧的间隔 (默认: 30)')
    parser.add_argument('--no-display', action='store_true',
                      help='不显示视频窗口')
    parser.add_argument('--api-key', type=str, default=None,
                      help='OpenRouter API密钥 (或设置OPENROUTER_API_KEY环境变量)')
    parser.add_argument('--model', type=str, default="qwen2.5vl:latest",
                      help='要使用的模型 (默认: qwen2.5vl:latest)')
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = ""
    if not api_key:
        print("错误：请提供OpenRouter API密钥 (通过--api-key参数或设置OPENROUTER_API_KEY环境变量)")
        return

    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    processor = VideoProcessor(
        frame_interval=args.interval,
        show_video=not args.no_display,
        api_key=api_key,
        model=args.model
    )
    
    print(f"开始处理视频 (使用模型: {args.model})，按 'q' 键退出...")
    processor.run(video_source)

if __name__ == "__main__":
    main()
