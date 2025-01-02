from roboflow import Roboflow
import supervision as sv
import cv2
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("parking-detection-jeremykevin")
model = project.version(8).model

result = model.predict("assets/images/parking-lot-1.jpg", confidence=40, overlap=30).json()

labels = [item["class"] for item in result["predictions"]]

detections = sv.Detections.from_inference(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("assets/images/parking-lot-1.jpg")

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))