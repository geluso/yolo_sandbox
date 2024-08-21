import cv2
import numpy as np
import sys

from ultralytics import YOLO

input_file = sys.argv[1]
print('opening file:', input_file)
cap = cv2.VideoCapture(input_file)
model = YOLO("yolov8m.pt", verbose=False)

RED = (0, 0, 225)
GREEN = (0, 225, 0)
BLUE = (225, 0, 0)

COLORS = {
  "bicycle": RED,
  "person": GREEN,
}


def get_color(name):
  if name in COLORS:
    return COLORS[name]
  return BLUE


while True:
  ret, frame = cap.read()
  if not ret:
    break

  results = model(frame, device="mps", verbose=False)
  result = results[0]
  bboxes = result.boxes.xyxy

  bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
  classes = np.array(result.boxes.cls.cpu(), dtype="int")

  for cls, bbox in zip(classes, bboxes):
    name = result.names[cls]
    color = get_color(name)
    (x, y, x2, y2) = bbox
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

  cv2.imshow("Img", frame)
  key = cv2.waitKey(1)
  if key == 27:
      break
cap.release()
cv2.destroyAllWindows()
