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

cap = cv2.VideoCapture(0) 

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(frame, confidence=40, overlap=30).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    cv2.imshow('Annotated Video', annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()