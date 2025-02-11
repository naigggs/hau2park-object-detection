import cv2
import os
from dotenv import load_dotenv
from roboflow import Roboflow
import supervision as sv
import numpy as np

def load_model(api_key):
    """Load Roboflow model for CPU processing."""
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("parking-detection-nhv0o")
    return project.version(2).model

def process_video(video_path, model):
    """Process video with CPU-friendly optimizations."""
    cap = cv2.VideoCapture(video_path)
    
    # CPU-efficient annotators
    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoxAnnotator()
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * 5
    frame_count = 0
    
    os.makedirs("screenshots", exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frames at specified interval
        if frame_count % frame_interval == 0:
            # Resize frame with efficient interpolation
            resized_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            
            # Perform object detection
            result = model.predict(resized_frame, confidence=40, overlap=30).json()
            
            # Process detections
            labels = [f"{item['class']} ({item['confidence']:.2f})" for item in result["predictions"]]
            detections = sv.Detections.from_inference(result)
            
            # Annotate frame
            annotated_frame = bounding_box_annotator.annotate(
                scene=resized_frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels)
            
            # Save screenshot
            screenshot_path = f"screenshots/frame_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            
            # Display frame
            cv2.imshow('Annotated Screenshot', annotated_frame)
        
        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    # Load model
    model = load_model(api_key)
    
    # Process video
    process_video("assets/videos/test.mp4", model)

if __name__ == "__main__":
    main()