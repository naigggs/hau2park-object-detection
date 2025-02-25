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

result = model.predict("assets/images/parking-lot-5.png", confidence=40, overlap=30).json()

labels = [f"{item['class']} ({item['confidence']:.2f})" for item in result["predictions"]]

detections = sv.Detections.from_inference(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("assets/images/parking-lot-5.png")

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))

with open("result.json", "w") as json_file:
    json.dump(result, json_file, indent=4)