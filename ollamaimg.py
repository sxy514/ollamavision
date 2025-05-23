import base64
import requests
from typing import Optional

def recognize_image(image_path: str, api_url: str = "http://localhost:11434/api/generate") -> Optional[str]:
    """
    Send an image to Ollama's qwen2.5vl model for recognition.
    
    Args:
        image_path: Path to the image file
        api_url: Ollama API endpoint (default: http://localhost:11434)
        
    Returns:
        str: The model's response or None if an error occurred
    """
    try:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the request payload
        payload = {
            "model": "qwen2.5vl",
            "prompt": "Describe this image in detail.",
            "images": [base64_image],
            "stream": False
        }
        
        # Send the request to Ollama
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Return the model's response
        return response.json().get("response")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")
    result = recognize_image(image_path)
    if result:
        print("\nRecognition Result:")
        print(result)
    else:
        print("Failed to process the image.")