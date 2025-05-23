import base64
import cv2
import requests
import numpy as np
from typing import Optional, Generator, Tuple
import time

def get_video_frames(video_path: str, frame_interval: int = 30) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract one frame every N frames
        
    Yields:
        Tuple containing the frame and its timestamp in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield rgb_frame, timestamp
            
        frame_count += 1
    
    cap.release()

def process_frame_with_ollama(frame: np.ndarray, prompt: str = "Describe this video frame in detail.") -> Optional[str]:
    """
    Send a video frame to Ollama's qwen2.5vl model for analysis.
    
    Args:
        frame: Input frame as a numpy array (RGB format)
        prompt: The prompt to send to the model
        
    Returns:
        str: The model's response or None if an error occurred
    """
    try:
        # Encode frame to JPEG and then to base64
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "model": "qwen2.5vl",
            "prompt": prompt,
            "images": [base64_image],
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json().get("response")
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

def process_video(video_path: str, frame_interval: int = 30) -> None:
    """
    Process a video file frame by frame using Ollama.
    
    Args:
        video_path: Path to the video file
        frame_interval: Process one frame every N frames
    """
    print(f"Processing video: {video_path}")
    
    for frame, timestamp in get_video_frames(video_path, frame_interval):
        print(f"\n--- Frame at {timestamp:.2f}s ---")
        
        # Process the frame
        description = process_frame_with_ollama(
            frame,
            prompt="Describe what's happening in this video frame in detail."
        )
        
        if description:
            print(f"Analysis: {description}")
        else:
            print("Failed to process frame")
        
        # Optional: Add a small delay to avoid overwhelming the system
        time.sleep(0.5)

if __name__ == "__main__":
    video_path = input("Enter the path to your video file: ")
    process_video(video_path)