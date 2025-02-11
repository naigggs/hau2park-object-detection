from inference import InferencePipeline
import time
import cv2

# Define your parking spots with their coordinates (x1, y1, x2, y2) in pixel coordinates
PARKING_SPOTS = {
    "P1": (473, 519, 588, 627),
    "P2": (615, 540, 732, 657),
    "P3": (748, 534, 903, 677),
    "P4": (950, 555, 1133, 718),
    "P5": (1182, 577, 1383, 767)
}

last_print_time = 0

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union (IOU) for two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

def print_predictions(predictions, video_frame):
    global last_print_time
    current_time = time.time()
    
    # Get the frame as numpy array
    frame = video_frame.image.copy()
    frame_height, frame_width = frame.shape[:2]
    
    occupied_spots = set()
    
    # Check each detection against our parking spots
    for detection in predictions.get("predictions", []):
        # Convert normalized coordinates to pixel coordinates
        x_center = detection["x"] * frame_width
        y_center = detection["y"] * frame_height
        width = detection["width"] * frame_width
        height = detection["height"] * frame_height
        
        # Calculate detection box in pixel coordinates
        detection_box = (
            x_center - width/2,
            y_center - height/2,
            x_center + width/2,
            y_center + height/2
        )
        
        # Draw detection box (blue)
        cv2.rectangle(frame, 
                     (int(detection_box[0]), int(detection_box[1])),
                     (int(detection_box[2]), int(detection_box[3])),
                     (255, 0, 0), 2)
        
        # Check overlap with parking spots
        for spot_id, spot_coords in PARKING_SPOTS.items():
            iou = calculate_iou(detection_box, spot_coords)
            if iou > 0.2:
                occupied_spots.add(spot_id)
    
    # Draw parking spots (green=available, red=occupied)
    for spot_id, (x1, y1, x2, y2) in PARKING_SPOTS.items():
        color = (0, 0, 255) if spot_id in occupied_spots else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, spot_id, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show the frame
    cv2.imshow('Parking Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.terminate()
    
    # Print status every 5 seconds
    if current_time - last_print_time >= 5:
        print(f"\nParking status at {time.strftime('%H:%M:%S')}:")
        for spot_id in PARKING_SPOTS:
            status = "OCCUPIED" if spot_id in occupied_spots else "EMPTY"
            print(f"{spot_id}: {status}")
        last_print_time = current_time

# Initialize and run the pipeline
pipeline = InferencePipeline.init(
    model_id="parking-detection-nhv0o/2",
    video_reference="assets/videos/test.mp4",
    on_prediction=print_predictions,
    api_key="wGRx0NiZ2Ax1TA7vm1Ew",
)

pipeline.start()
pipeline.join()
cv2.destroyAllWindows()