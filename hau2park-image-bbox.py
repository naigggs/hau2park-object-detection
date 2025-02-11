from roboflow import Roboflow
import supervision as sv
import cv2
import os
from dotenv import load_dotenv
import json

load_dotenv()
api_key = os.getenv("API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("parking-detection-jeremykevin")
model = project.version(8).model

predefined_spaces = [
    {"id": "P1", "x_min": 214, "x_max": 340, "y_min": 72, "y_max": 309},
    {"id": "P2", "x_min": 349, "x_max": 473, "y_min": 72, "y_max": 309},
    {"id": "P3", "x_min": 487, "x_max": 617, "y_min": 72, "y_max": 309},
    {"id": "P4", "x_min": 624, "x_max": 756, "y_min": 72, "y_max": 309},
]

result = model.predict("assets/images/parking-lot-1.jpg", confidence=40, overlap=30).json()

labels = [f"{item['class']} ({item['confidence']:.2f})" for item in result["predictions"]]

detections = sv.Detections.from_inference(result)

for prediction in result["predictions"]:
    x = prediction["x"]
    y = prediction["y"]
    status = prediction["class"]
    
    for space in predefined_spaces:
        if space["x_min"] <= x <= space["x_max"] and space["y_min"] <= y <= space["y_max"]:
            space["status"] = status
            space["confidence"] = prediction["confidence"]
            break

with open("parking_spaces.json", "w") as json_file:
    json.dump(predefined_spaces, json_file, indent=4)

image = cv2.imread("assets/images/parking-lot-1.jpg")
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))

print(json.dumps(predefined_spaces, indent=4))
