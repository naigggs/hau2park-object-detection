from inference import InferencePipeline
import time
import cv2
from supabase import create_client, Client
import os

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define parking spots
PARKING_SPOTS = {
    "P1": (615, 540, 732, 657),
    "P2": (748, 534, 903, 677),
    "P3": (950, 555, 1133, 718),
    "P4": (1182, 577, 1383, 767),
    "P5": (1450, 601, 1747, 798)
}

# State management
last_print_time = 0
previous_status = {}
spot_counts = {spot: 0 for spot in PARKING_SPOTS}
total_frames_in_interval = 0

def fetch_initial_status():
    """Fetch current parking status from Supabase to initialize previous_status."""
    global previous_status
    response = supabase.table("parking_spaces").select("name, status").execute()
    
    if response.data:
        previous_status = {entry["name"]: entry["status"] for entry in response.data}
    else:
        previous_status = {spot: "Open" for spot in PARKING_SPOTS}  # Default to Open

    print("Fetched initial parking status:", previous_status)

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

def update_supabase(occupied_spots):
    """Update Supabase only if a parking spot status changes"""
    global previous_status
    for spot_id in PARKING_SPOTS:
        new_status = "Occupied" if spot_id in occupied_spots else "Open"

        if previous_status.get(spot_id) != new_status:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            update_data = {
                "status": new_status,
                "updated_at": timestamp
            }
            
            if new_status == "Open":
                update_data["user"] = "None"

            response = (
                supabase.table("parking_spaces")
                .update(update_data)
                .eq("name", spot_id)
                .execute()
            )
            print(f"Updated {spot_id}: {update_data} - Response: {response}")
            previous_status[spot_id] = new_status

def print_predictions(predictions, video_frame):
    global last_print_time, spot_counts, total_frames_in_interval
    
    current_occupied = set()
    frame = video_frame.image.copy()

    # Process detections
    for detection in predictions.get("predictions", []):
        detection_box = (
            detection["x"] - detection["width"]/2,
            detection["y"] - detection["height"]/2,
            detection["x"] + detection["width"]/2,
            detection["y"] + detection["height"]/2
        )
        
        # Draw detection boxes
        cv2.rectangle(frame, 
                     (int(detection_box[0]), int(detection_box[1])),
                     (int(detection_box[2]), int(detection_box[3])),
                     (255, 0, 0), 2)
        
        # Check against parking spots
        for spot_id, spot_coords in PARKING_SPOTS.items():
            iou = calculate_iou(detection_box, spot_coords)
            if iou > 0.2 and detection["class"] == "occupied":
                current_occupied.add(spot_id)

    # Update counts for this frame
    for spot_id in PARKING_SPOTS:
        if spot_id in current_occupied:
            spot_counts[spot_id] += 1
    total_frames_in_interval += 1

    # Draw parking spot boxes
    for spot_id, (x1, y1, x2, y2) in PARKING_SPOTS.items():
        color = (0, 0, 255) if spot_id in current_occupied else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, spot_id, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Parking Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.terminate()

    # Handle status updates every 5 seconds
    current_time = time.time()
    if current_time - last_print_time >= 5:
        occupied_spots = set()
        
        for spot_id in PARKING_SPOTS:
            if total_frames_in_interval == 0:
                ratio = 0
            else:
                ratio = spot_counts[spot_id] / total_frames_in_interval
            
            current_status = previous_status.get(spot_id, "Open")
            
            # Hysteresis thresholds
            if current_status == "Occupied":
                # Require low confirmation to stay occupied
                if ratio < 0.2:  # 20% detection threshold to mark as open
                    new_status = "Open"
                else:
                    new_status = "Occupied"
            else:
                # Require higher confirmation to become occupied
                if ratio >= 0.4:  # 40% detection threshold to mark as occupied
                    new_status = "Occupied"
                else:
                    new_status = "Open"
            
            if new_status == "Occupied":
                occupied_spots.add(spot_id)

        print(f"\nParking status at {time.strftime('%H:%M:%S')}:")
        for spot_id in PARKING_SPOTS:
            status = "OCCUPIED" if spot_id in occupied_spots else "EMPTY"
            print(f"{spot_id}: {status}")

        update_supabase(occupied_spots)
        
        # Reset counters
        spot_counts = {spot: 0 for spot in PARKING_SPOTS}
        total_frames_in_interval = 0
        last_print_time = current_time

# Fetch initial status and run pipeline
fetch_initial_status()

pipeline = InferencePipeline.init(
    model_id="parking-detection-nhv0o/2",
    video_reference="assets/videos/test-3.mp4",
    on_prediction=print_predictions,
    api_key=API_KEY,
)

pipeline.start()
pipeline.join()